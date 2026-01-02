from mpi4py import MPI
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
torch.manual_seed(42)
np.random.seed(42)

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
torch.set_num_threads(1)
from torch.utils.data import Subset

print(f"rank: {mpi_rank}, size: {mpi_size}")
device = torch.device("cpu") 
metrics_dict = torchmetrics.MetricCollection({
    'dice1': DiceScore(num_classes=16, average='micro', input_format='mixed', include_background=False),
    'dice_macro': DiceScore(num_classes=16, average='macro', input_format='mixed', include_background=False),
    'dice': GeneralizedDiceScore(num_classes=16,include_background=False,weight_type='square', input_format='mixed', per_class=True),
    'iou': torchmetrics.JaccardIndex(num_classes=16, average='macro', task='multiclass', ignore_index=0, zero_division=0.0),
    'precision': torchmetrics.Precision(num_classes=16, average='macro', task='multiclass', ignore_index=0),
    'recall': torchmetrics.Recall(num_classes=16, average='macro', task='multiclass', ignore_index=0),
}).to(device)

class_weights = torch.tensor([
        0.1,   # 0: фон
        2.0,   # 1: селезенка
        2.0,   # 2: правая почка
        2.0,   # 3: левая почка
        2.0,   # 4: желчный пузырь 
        2.0,   # 5: пищевод
        1.0,   # 6: печень 
        1.5,   # 7: желудок
        2.0,   # 8: аорта
        2.0,   # 9: нижняя полая вена
        2.0,   # 10: воротная вена
        1.5,   # 11: поджелудочная
        2.0,   # 12: правый надпочечник 
        2.0,   # 13: левый надпочечник
        2.0,   # 14: 12-перстная кишка
        2.0,   # 15: мочевой пузырь
    ], dtype=torch.float32).to(device)


class AMOSDataset(Dataset):
    def __init__(self, folder_img, folder_labels, max_items=10):
        self.images = folder_img[:max_items]
        self.labels = folder_labels[:max_items]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path_img = self.images[idx]
        path_lbl = self.labels[idx]

        vol_img = nib.load(path_img).get_fdata()  # (D,H,W)
        vol_lbl = nib.load(path_lbl).get_fdata() 

        vol_img = vol_img.astype("float32")
        vol_lbl = vol_lbl.astype("int64")

        
        vol_img = np.clip(vol_img, -200, 300)
        vol_img = (vol_img - vol_img.mean()) / (vol_img.std() + 1e-8)

        vol_img = torch.tensor(vol_img)
        
        vol_img = vol_img[:64, :64, :64]  # D,H,W

        vol_lbl = torch.tensor(vol_lbl)
        vol_lbl = vol_lbl[:64, :64, :64]  # D,H,W


        x = vol_img.unsqueeze(0)  # (1, D, H, W)
        y = vol_lbl

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
    def __init__(self, num_classes=16, ignore_index=0, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (B, C, D, H, W)
        # target: (B, D, H, W)
        pred_soft = F.softmax(pred, dim=1)
        
        
        target_onehot = F.one_hot(target.long(), self.num_classes)  # (B, D, H, W, C)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        
        pred_soft = pred_soft[:, 1:]
        target_onehot = target_onehot[:, 1:]
        
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3, 4))  # (B, C-1)
        union = pred_soft.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
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

        self.out_conv = nn.Conv3d(16, 16, 1)  # 16 классов

    def forward(self, x):
        skip_connections = [] # 16,32,64, 128
        # Encoder
        x = self.enc1(x) #стало 16
        skip_connections.append(x)
        x = self.pool(x)

        x = self.enc2(x) #стало 32
        skip_connections.append(x)
        x = self.pool(x)

        x = self.enc3(x) #стало 64
        skip_connections.append(x)
        x = self.pool(x)

        x = self.enc4(x) #стало 128
        skip_connections.append(x)
        x = self.pool(x) # стало (D/16,H/16,W/16)


        x = self.bottleneck(x) #стало 256


        # Decoder
        x = self.up1(x) # стало 128
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
    

    n = 20
    device = torch.device("cpu")
    
    dataset = AMOSDataset(folder_img=filtered_imgs,folder_labels=filtered_lbls,max_items=50)
    index = list(range(len(dataset)))                   #
    rank_index = index[mpi_rank::mpi_size]              #       
    local_dataset = Subset(dataset, rank_index)         #
    train_loader = DataLoader(local_dataset, batch_size=1, shuffle=True)
    print(f"index: {index[:5]}, rank index: {rank_index[:5]}, local dataset: {len(local_dataset)}\n")

    val_dataset = AMOSDataset(folder_img=val_imgs,folder_labels=val_lbls,max_items=20)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    model = UNet3D().to(device)
    for p in model.parameters():
        comm.Bcast(p.data.numpy(), root = 0)
    print(f"parametrs: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    dice_loss = DiceLoss(num_classes=16, ignore_index=0) 
    

    print(f"Dataset size: {len(dataset)}")

    for epoch in range(n):
        print(f'--- Epoch {epoch} ---')
        model.train()
        start = time.time()
        train_loss, train_ce, train_dice = 0, 0, 0
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}'):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            
            loss_ce = ce_loss(pred, y)
            loss_dice = dice_loss(pred, y)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            
            loss.backward()
            for p in model.parameters():                                        #
                if p.grad is not None:                                          #     
                    grad = p.grad.data.numpy()                                  #
                    comm.Allreduce(MPI.IN_PLACE, grad, op=MPI.SUM)              #
                    grad /= mpi_size                                            #

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
            optimizer.step()
            
            
            train_loss += loss.item()
            train_ce += loss_ce.item()
            train_dice += loss_dice.item()
            
            with torch.no_grad():
                #pred_class = torch.argmax(pred, dim=1)
                metrics_dict.update(F.softmax(pred, dim=1), y)
            

        train_metrics = metrics_dict.compute()
        metrics_dict.reset()
        if mpi_rank == 0:
            print(f"\n[TRAIN] Loss: {train_loss/len(train_loader):.8f} "
                f"(CELoss: {train_ce/len(train_loader):.8f}, DiceLoss: {train_dice/len(train_loader):.8f})")
            #print(f"Dice: {train_metrics['dice'].item():.8f}")
            dice_class = train_metrics['dice']
            for i, d in enumerate(dice_class, start=1):
                print(f"Class {i}: Dice = {d.item():.4f}")
            print(f"Dice base: {train_metrics['dice1'].item():.8f}")
            print(f"Dice macro: {train_metrics['dice_macro'].item():.8f}")
            print(f"IoU:  {train_metrics['iou'].item():.8f}")
            print(f"Precision: {train_metrics['precision'].item():.8f}")
            print(f"Recall:  {train_metrics['recall'].item():.8f}")
            print(f"Time: {time.time() - start}")

        #Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc='Validation'):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                
                loss = 0.5 * ce_loss(pred, y) + 0.5 * dice_loss(pred, y)
                val_loss += loss.item()
                
                #pred_class = torch.argmax(pred, dim=1)
                metrics_dict.update(F.softmax(pred, dim=1), y)

        val_metrics = metrics_dict.compute()
        dice_local = val_metrics['dice'].mean().item()
        dice_global = comm.allreduce(dice_local, op=MPI.SUM) / mpi_size

        metrics_dict.reset()
        
        if mpi_rank == 0:
            print(f"\n[VAL] Loss: {val_loss/len(val_loader):.4f}")
            #print(f"Dice: {val_metrics['dice'].item():.6f}")
            dice_class_val = val_metrics['dice']
            for i, d in enumerate(dice_class_val, start=1):
                print(f"Class {i}: Dice = {d.item():.6f}")
            print(f"Dice base: {val_metrics['dice1'].item():.6f}")
            print(f"Dice macro: {train_metrics['dice_macro'].item():.8f}")
            print(f"IoU:  {val_metrics['iou'].item():.6f}")
            print(f"Prec: {val_metrics['precision'].item():.6f}")
            print(f"Rec:  {val_metrics['recall'].item():.6f}")
            print(f"Dice allreduce: {dice_global:.6f}")

        
        if mpi_rank == 0:
            scheduler.step(val_metrics['dice'].mean().item())
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = None
        lr = comm.bcast(lr, root = 0)
        for dict in optimizer.param_groups:
            dict["lr"] = lr

        comm.Barrier()





def filter_data(folder_img, folder_labels,  min_size=(512,512,86)):
    images = sorted(glob.glob(folder_img + "/*.nii.gz"))
    labels = sorted(glob.glob(folder_labels + "/*.nii.gz"))
    return images, labels
    

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

