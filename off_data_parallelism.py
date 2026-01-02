import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torchmetrics
from torchmetrics.segmentation import DiceScore
import numpy as np
from tqdm import tqdm
from torchmetrics.segmentation import GeneralizedDiceScore
import random, time, os, gc
from mpi4py import MPI
from torch.utils.data import Subset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()




torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


torch.use_deterministic_algorithms(True)

device = torch.device("cpu") 
metrics_dict = torchmetrics.MetricCollection({
    'dice_micro': DiceScore(num_classes=16, average='micro', input_format='mixed', include_background=False),
    'dice_macro': DiceScore(num_classes=16, average='macro', input_format='mixed', include_background=False),
    'dice': GeneralizedDiceScore(num_classes=16,include_background=False,weight_type='square', input_format='mixed', per_class=True),
    'iou': torchmetrics.JaccardIndex(num_classes=16, average='macro', task='multiclass', ignore_index=0, zero_division=0.0),
    'precision': torchmetrics.Precision(num_classes=16, average='macro', task='multiclass', ignore_index=0),
    'recall': torchmetrics.Recall(num_classes=16, average='macro', task='multiclass', ignore_index=0),
}).to(device)
class_weights = torch.tensor([
    0.1,  # 0
    2.0,  # 1
    1.5,  # 2
    1.5,  # 3
    3.0,  # 4 
    2.5,  # 5
    1.0,  # 6 
    1.2,  # 7
    2.5,  # 8
    2.5,  # 9
    3.0,  # 10
    2.0,  # 11
    3.0,  # 12
    3.0,  # 13
    2.0,  # 14
    2.5,  # 15
    ]).to(device)

def filter_data(folder_img, folder_labels,  min_size=(512,512,80)):
    images = sorted(glob.glob(folder_img + "/*.nii.gz"))
    labels = sorted(glob.glob(folder_labels + "/*.nii.gz"))
    return images, labels


class AMOSDataset(Dataset):
    def __init__(self, folder_img, folder_labels, max_items=10):
        self.images = folder_img[:max_items]
        self.labels = folder_labels[:max_items]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path_img = self.images[idx]
        path_lbl = self.labels[idx]

        img_full = nib.load(path_img).get_fdata().astype("float32")  # (D,H,W)
        img_full = np.clip(img_full, -200, 300)
        img_full = (img_full - img_full.mean()) / img_full.std() + 1e-8

        vol_img = img_full[60:380, 60:380, 5:85].copy()
        #vol_img = img_full[:16, :16, :16].copy()
        del img_full

        lbl_obj = nib.load(path_lbl, mmap=True)
        vol_lbl = np.asanyarray(lbl_obj.dataobj)[60:380, 60:380, 5:85].astype("int64")

        x = torch.tensor(vol_img).unsqueeze(0)
        y = torch.tensor(vol_lbl)

        return x, y


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(num_groups, c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(num_groups, c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DiceLoss(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        tgt = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        pred = pred[:, 1:]
        tgt = tgt[:, 1:]
        inter = (pred * tgt).sum((2, 3, 4))
        union = pred.sum((2, 3, 4)) + tgt.sum((2, 3, 4))
        dice = (2 * inter + 1e-5) / (union + 1e-5)
        return 1 - dice.mean()




class UNet3D(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()

        self.enc1 = ConvBlock(1, 16)
        self.enc2 = ConvBlock(16,32)
        self.enc3 = ConvBlock(32,64)
        self.enc4 = ConvBlock(64,128)

        self.bottleneck = ConvBlock(128, 256)

        self.up1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.up3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.up4 = nn.ConvTranspose3d(32, 16, 2, stride=2)


        self.dec1 = ConvBlock(2*128, 128)
        self.dec2 = ConvBlock(2*64, 64)
        self.dec3 = ConvBlock(2*32, 32)
        self.dec4 = ConvBlock(2*16, 16)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.out_conv = nn.Conv3d(16, 16, 1)  # 16 

    def forward(self, x):
        skip_connections = [] # 16,32,64, 128
        
        x = self.enc1(x) # 16
        skip_connections.append(x)
        x = self.pool(x)

        x = self.enc2(x) # 32
        skip_connections.append(x)
        x = self.pool(x)

        x = self.enc3(x) # 64
        skip_connections.append(x)
        x = self.pool(x)

        x = self.enc4(x) # 128
        skip_connections.append(x)
        x = self.pool(x) #  (D/16,H/16,W/16)


        x = self.bottleneck(x) # 256


        
        x = self.up1(x) #  128
        skip_features = skip_connections[-1] 
        x = torch.cat([x, skip_features], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        skip_features = skip_connections[-2]
        x = torch.cat([x, skip_features], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        skip_features = skip_connections[-3]
        x = torch.cat([x, skip_features], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        skip_features = skip_connections[-4]
        x = torch.cat([x, skip_features], dim=1)
        x = self.dec4(x)

        return self.out_conv(x)


def train(filtered_imgs, filtered_lbls, val_imgs, val_lbls):
    n = 6
    device = torch.device("cpu")
    
    dataset = AMOSDataset(folder_img=filtered_imgs,folder_labels=filtered_lbls,max_items=64)
    index = list(range(len(dataset)))
    rank_index = index[mpi_rank::mpi_size]
    local_dataset = Subset(dataset, rank_index)
    print(f"index: {index[:5]}, rank index: {rank_index[:5]}, local dataset: {len(local_dataset)}\n")

    val_dataset = AMOSDataset(folder_img=val_imgs,folder_labels=val_lbls,max_items=14)
    if mpi_rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        print("VALIDATION FILES")
        for p in val_dataset.images:
            print(os.path.basename(p))
        
    
    model = UNet3D().to(device)
    if mpi_rank == 0:
        print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    
    for p in model.parameters():
        buf = p.data.numpy()
        comm.Bcast(buf, root=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    dice_loss = DiceLoss(num_classes=16)
    
    print(f"Dataset size: {len(dataset)}")
    
    for epoch in range(n):
        epoch_start = time.time()

        train_loader = DataLoader(local_dataset, batch_size=1, shuffle=True, num_workers=0)
        print(f'Epoch {epoch}')
        model.train()
        

        batch_times = []
        forward_times = []
        backward_times = []
        sync_times = []
        
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}'):

            batch_start = time.time()
            
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            

            forward_start = time.time()
            pred = model(x)
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            loss_ce = ce_loss(pred, y)
            loss_dice = dice_loss(pred, y)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            

            backward_start = time.time()
            loss.backward()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)
            

            sync_start = time.time()
            for p in model.parameters():
                if p.grad is not None:
                    grad = p.grad.data.numpy()
                    comm.Allreduce(MPI.IN_PLACE, grad, op=MPI.SUM)
                    grad /= mpi_size
            sync_end = time.time()
            sync_times.append(sync_end - sync_start)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
        

        if mpi_rank == 0:
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            print(f"Epoch {epoch}: {epoch_time:.2f} sec")

        if batch_times:
            mean_batch_time = np.mean(batch_times)
            mean_forward_time = np.mean(forward_times)
            mean_backward_time = np.mean(backward_times)
            mean_sync_time = np.mean(sync_times)
        else:
            mean_batch_time = mean_forward_time = mean_backward_time = mean_sync_time = 0.0
        

        local_times = np.array([mean_batch_time, mean_forward_time, mean_backward_time, mean_sync_time], dtype=np.float32)
        

        if mpi_rank == 0:
            mean_times = np.zeros_like(local_times)
        else:
            mean_times = None
        
        comm.Reduce([local_times, MPI.FLOAT], [mean_times, MPI.FLOAT], op=MPI.SUM, root=0)
        

        if mpi_rank == 0:
            mean_times /= mpi_size
            print(f"\n (Epoch {epoch}):")
            print(f"  Batch time: {mean_times[0]:.4f}s")
            print(f"  Forward pass: {mean_times[1]:.4f}s")
            print(f"  Backward pass: {mean_times[2]:.4f}s")
            print(f"  Sync time: {mean_times[3]:.4f}s")
        

        if mpi_rank == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc='Validation'):
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    
                    loss = 0.5 * ce_loss(pred, y) + 0.5 * dice_loss(pred, y)
                    val_loss += loss.item()
                    
                    metrics_dict.update(F.softmax(pred, dim=1), y)
            
            val_metrics = metrics_dict.compute()
            metrics_dict.reset()
            
            print(f"\n[VAL] Loss: {val_loss/len(val_loader):.4f}")
            dice_class_val = val_metrics['dice']
            for i, d in enumerate(dice_class_val, start=1):
                print(f"Class {i}: Dice = {d.item():.6f}")
            print(f"Dice base: {val_metrics['dice_micro'].item():.6f}")
            print(f"Dice macro: {val_metrics['dice_macro'].item():.8f}")
            print(f"IoU:  {val_metrics['iou'].item():.6f}")
            print(f"Prec: {val_metrics['precision'].item():.6f}")
            print(f"Rec:  {val_metrics['recall'].item():.6f}")
            
            scheduler.step(val_metrics['dice_micro'].item())
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = None
        
        lr = comm.bcast(lr, root=0)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr



        comm.Barrier()
        gc.collect()

if __name__ == "__main__":
    filtered_imgs, filtered_lbls = filter_data(
        "amos_good_samples/imagesTrFil",
        "amos_good_samples/labelsTrFil",
        min_size=(512,512,86))
    val_imgs,val_lbls = filter_data(
        "amos_good_val_samples/imagesVaFil",
        "amos_good_val_samples/labelsVaFil",
        min_size=(512,512,86))
    train(filtered_imgs, filtered_lbls, val_imgs, val_lbls)
