print("rabotaem", flush=True)

import random, time, os, gc

import warnings
import numpy as np
import time
import glob
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torchmetrics
from torchmetrics.segmentation import DiceScore

from tqdm import tqdm
from torchmetrics.segmentation import GeneralizedDiceScore

warnings.filterwarnings("ignore", category=UserWarning)


if mpi_rank == 0:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


# torch.use_deterministic_algorithms(True) 

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

        lbl_obj = nib.load(path_lbl)
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
            nn.ReLU(inplace=False),  
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(num_groups, c_out),
            nn.ReLU(inplace=False), 
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


class EncoderPart(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.enc4 = ConvBlock(64, 128)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(128, 256)

    def forward(self, x):
        skips = []

        x = self.enc1(x)
        skips.append(x)
        x = self.pool(x)

        x = self.enc2(x)
        skips.append(x)
        x = self.pool(x)

        x = self.enc3(x)
        skips.append(x)
        x = self.pool(x)

        x = self.enc4(x)
        skips.append(x)
        x = self.pool(x)

        x = self.bottleneck(x)
        return x, skips


class DecoderPart(nn.Module):
    
    def __init__(self, num_classes=16):
        super().__init__()

        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64, 32)

        self.up4 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(32, 16)

        self.out_conv = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, x, skips):
        x = self.up1(x)
        x = torch.cat([x, skips[-1]], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skips[-2]], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skips[-3]], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat([x, skips[-4]], dim=1)
        x = self.dec4(x)

        return self.out_conv(x)



def train(filtered_imgs, filtered_lbls, val_imgs, val_lbls):
    print(" rabotaem ", flush=True)
    n = 10
    device = torch.device("cpu")

    if mpi_size != 2:return

    dataset = AMOSDataset(folder_img=filtered_imgs, folder_labels=filtered_lbls, max_items=64)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

    val_dataset = AMOSDataset(folder_img=val_imgs, folder_labels=val_lbls, max_items=14)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    if mpi_rank == 0:
        print(f"Train size: {len(dataset)}, Val size: {len(val_dataset)}")

    if mpi_rank == 0:
        encoder = EncoderPart(in_channels=1).to(device)
        optimizer_enc = torch.optim.AdamW(encoder.parameters(), lr=8e-4, weight_decay=1e-5)
        print(f"Rank {mpi_rank} encoder params: {sum(p.numel() for p in encoder.parameters())}")

    elif mpi_rank == 1:
        decoder = DecoderPart(num_classes=16).to(device)
        optimizer_dec = torch.optim.AdamW(decoder.parameters(), lr=8e-4, weight_decay=1e-5)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
        dice_loss = DiceLoss(num_classes=16)
        print(f"Rank {mpi_rank} decoder params: {sum(p.numel() for p in decoder.parameters())}")

    print("here")
    if mpi_rank == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dec, mode='max', factor=0.5, patience=2)

    
    for epoch in range(n):
        epoch_start = time.time()

        if mpi_rank == 0:
            encoder.train()
        elif mpi_rank == 1:
            decoder.train()

        if mpi_rank == 0:
            print(f"Epoch {epoch} (rank 0)")
        if mpi_rank == 1:
            print(f"Epoch {epoch} (rank 1)")

        batch_times = []
        forward_times = []
        backward_times = []
        comm_times = []

        
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}'):
            x = x.to(device)    #  [1,1,D,H,W]
            y = y.to(device)    #  [1,D,H,W]

            batch_start = time.time()

            if mpi_rank == 0:
                optimizer_enc.zero_grad()
            elif mpi_rank == 1:
                optimizer_dec.zero_grad()

            
            fwd0 = time.time()
            if mpi_rank == 0:
                x_enc, skips = encoder(x)  
                fwd1 = time.time()

                comm_start = time.time()
                comm.send(x_enc.detach().cpu().numpy(), dest=1, tag=11)
                
                #skips_np = [s.detach().cpu().numpy() for s in skips]
                comm.send(len(skips), dest=1, tag=12) 
                for i, skip_tensor in enumerate(skips):
                    print(f"Skip {i} shape: {skip_tensor.shape}, dtype: {skip_tensor.dtype}")
                    print(f"Skip {i} size in bytes: {skip_tensor.numel() * skip_tensor.element_size()}")

                    comm.send(skip_tensor.detach().cpu().numpy(), dest=1, tag=53+i)
                comm_end = time.time()

                forward_times.append((fwd1 - fwd0))
                comm_times.append(comm_end - comm_start)

                recv_start = time.time()
                grad_enc_np = comm.recv(source=1, tag=21) 
                recv_end = time.time()
                comm_times.append(recv_end - recv_start)

                grad_enc = torch.from_numpy(grad_enc_np).to(device)
                
                bwd_start = time.time()
                x_enc.backward(grad_enc)
                bwd_end = time.time()
                backward_times.append(bwd_end - bwd_start)

                
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optimizer_enc.step()

            elif mpi_rank == 1:

                comm_start = time.time()
                x_enc_np = comm.recv(source=0, tag=11)
                #skips_np = comm.recv(source=0, tag=12)


                num_skips = comm.recv(source=0, tag=12)
                skips_remote = []
                for i in range(num_skips):
                    skip_np = comm.recv(source=0, tag=53+i)
                    skips_remote.append(torch.tensor(skip_np, device=device, dtype=torch.float32))





                comm_end = time.time()
                comm_times.append(comm_end - comm_start)

                fwd1 = time.time()
                x_enc_remote = torch.tensor(x_enc_np, device=device, dtype=torch.float32, requires_grad=True)
                #skips_remote = [torch.tensor(snp, device=device, dtype=torch.float32) for snp in skips_np]

                
                pred = decoder(x_enc_remote, skips_remote)   # [1, C, D, H, W]
                fwd2 = time.time()
                forward_times.append(fwd2 - fwd1)

                
                loss_ce = ce_loss(pred, y)
                loss_dice = dice_loss(pred, y)
                loss = 0.5 * loss_ce + 0.5 * loss_dice

                bwd_start = time.time()
                loss.backward()
                bwd_end = time.time()
                backward_times.append(bwd_end - bwd_start)

                
                send_start = time.time()
                grad_for_enc = x_enc_remote.grad.detach().cpu().numpy()
                comm.send(grad_for_enc, dest=0, tag=21)
                send_end = time.time()
                comm_times.append(send_end - send_start)

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                optimizer_dec.step()

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            
            del x, y
            if mpi_rank == 1:
                del x_enc_remote, skips_remote, pred, loss, loss_ce, loss_dice, grad_for_enc
            if mpi_rank == 0:
                del x_enc, skips, grad_enc

            gc.collect()
        
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"Epoch {epoch} : {epoch_time:.2f} seconds")

        local_times = np.array([
            np.mean(batch_times) if batch_times else 0.0,
            np.mean(forward_times) if forward_times else 0.0,
            np.mean(backward_times) if backward_times else 0.0,
            np.mean(comm_times) if comm_times else 0.0
        ], dtype=np.float32)

        if mpi_rank == 0:
            global_times = np.zeros_like(local_times)
        else:
            global_times = None

        comm.Reduce([local_times, MPI.FLOAT], [global_times, MPI.FLOAT], op=MPI.SUM, root=0)

        if mpi_rank == 0:
            global_times /= mpi_size
            print(f"\n(Epoch {epoch}):")
            print(f" Batch time: {global_times[0]:.4f}s")
            print(f" Forward (local) : {global_times[1]:.4f}s")
            print(f" Backward (local): {global_times[2]:.4f}s")
            print(f" Comm (send/recv): {global_times[3]:.4f}s")

        if mpi_rank == 0:
            encoder.eval()
        elif mpi_rank == 1:
            decoder.eval()

        if mpi_rank == 1:
            metrics_dict.reset()

        val_loss = 0.0
        val_count = 0

        for i, (xv, yv) in enumerate(val_loader):
            xv = xv.to(device)
            yv = yv.to(device)

            if mpi_rank == 0:
                with torch.no_grad():
                    x_enc_val, skips_val = encoder(xv)
                    comm.send(x_enc_val.detach().cpu().numpy(), dest=1, tag=111)
                    skips_np = [s.detach().cpu().numpy() for s in skips_val]
                    comm.send(skips_np, dest=1, tag=112)

                del x_enc_val, skips_val

            elif mpi_rank == 1:
                x_enc_np = comm.recv(source=0, tag=111)
                skips_np = comm.recv(source=0, tag=112)

                x_enc_val = torch.tensor(x_enc_np, device=device, dtype=torch.float32, requires_grad=False)
                skips_val = [torch.tensor(snp, device=device, dtype=torch.float32) for snp in skips_np]

                with torch.no_grad():
                    pred_val = decoder(x_enc_val, skips_val)
                    loss_val = 0.5 * ce_loss(pred_val, yv) + 0.5 * dice_loss(pred_val, yv)
                    val_loss += loss_val.item()
                    val_count += 1

                    metrics_dict.update(F.softmax(pred_val, dim=1), yv)

                del x_enc_val, skips_val, pred_val, loss_val

        if mpi_rank == 1:
            val_metrics = metrics_dict.compute()
            metrics_dict.reset()

            report = {
                'val_loss': float(val_loss / val_count) if val_count else 0.0,
                'dice_micro': float(val_metrics['dice_micro'].item()),
                'dice_macro': float(val_metrics['dice_macro'].item()),
                'iou': float(val_metrics['iou'].item()),
                'precision': float(val_metrics['precision'].item()),
                'recall': float(val_metrics['recall'].item()),
            
                'dice_in_class': val_metrics['dice'].cpu().numpy().tolist()
            }
            comm.send(report, dest=0, tag=200)

        if mpi_rank == 0:
            report = comm.recv(source=1, tag=200)
            print(f"\n[VAL] Loss: {report['val_loss']:.4f}")
            dice_in_class = report['dice_in_class']
            for i, val in enumerate(dice_in_class, start=1):
                print(f"  Class {i}: Dice = {val:.6f}")
            print(f"  Dice micro: {report['dice_micro']:.6f}")
            print(f"  Dice macro: {report['dice_macro']:.6f}")
            print(f"  IoU: {report['iou']:.6f}")
            print(f"  Precision: {report['precision']:.6f}")
            print(f"  Recall: {report['recall']:.6f}")

            lr = None
            lr = comm.recv(source=1, tag=201)
            for pg in optimizer_enc.param_groups:
                pg['lr'] = lr
            print(f"  New LR: {lr:.6g}")

        elif mpi_rank == 1:
            scheduler.step(val_metrics['dice_micro'].item())
            lr_to_send = optimizer_dec.param_groups[0]['lr']
            comm.send(lr_to_send, dest=0, tag=201)


        comm.Barrier()
        gc.collect()


if __name__ == "__main__":
    print("rabotaem ", flush=True)
    filtered_imgs, filtered_lbls = filter_data(
        "amos_good_samples/imagesTrFil",
        "amos_good_samples/labelsTrFil",
        min_size=(512,512,86))
    val_imgs,val_lbls = filter_data(
        "amos_good_val_samples/imagesVaFil",
        "amos_good_val_samples/labelsVaFil",
        min_size=(512,512,86))
    train(filtered_imgs, filtered_lbls, val_imgs, val_lbls)
