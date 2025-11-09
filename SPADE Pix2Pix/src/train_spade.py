"""
train_spade.py
---------------
Main training entrypoint for SPADE fine-tuning.
"""

from pathlib import Path
from tqdm.auto import tqdm
import torch
from torchvision.utils import make_grid
from torchvision import transforms

from dataset import SideBySideDataset
from models import UNetGenerator_SPADE, PatchDiscriminatorSN
from utils import seed_all, d_hinge, g_hinge, l1, perceptual, denorm, compute_fid, build_vgg


def train_spade(G, D, train_loader, val_loader, cfg, device):
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg['LR_G'], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg['LR_D'], betas=(0.5, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=400, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=400, gamma=0.5)
    vgg = build_vgg(device)

    best_fid = float("inf")

    for epoch in range(1, cfg['EPOCHS'] + 1):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['EPOCHS']}")
        epG = epD = epL1 = 0.

        for cond, target, _ in pbar:
            cond, target = cond.to(device), target.to(device)

            # --- Train D ---
            with torch.no_grad():
                fake = G(cond)
            rp = torch.cat([cond, target], 1)
            fp = torch.cat([cond, fake], 1)

            D.zero_grad(set_to_none=True)
            rlog, _ = D(rp)
            flog, _ = D(fp)
            loss_D = d_hinge(rlog, flog)
            loss_D.backward()
            opt_D.step()

            # --- Train G ---
            G.zero_grad(set_to_none=True)
            gen = G(cond)
            gp = torch.cat([cond, gen], 1)
            glog, gf = D(gp)
            loss_G_adv = g_hinge(glog)
            loss_G_L1 = l1(gen, target) * cfg['LAMBDA_L1']

            with torch.no_grad():
                _, rf = D(rp)
            loss_G_FM = sum(l1(g, r) for g, r in zip(gf, rf)) * cfg['LAMBDA_FM']
            loss_G_VGG = perceptual(denorm(gen), denorm(target), vgg) * cfg['LAMBDA_VGG']

            loss_G = loss_G_adv + loss_G_L1 + loss_G_FM + loss_G_VGG
            loss_G.backward()
            opt_G.step()

            epG += loss_G.item(); epD += loss_D.item(); epL1 += loss_G_L1.item()
            pbar.set_postfix(D=f"{loss_D.item():.3f}", G=f"{loss_G.item():.2f}", L1=f"{loss_G_L1.item():.2f}")

        scheduler_G.step(); scheduler_D.step()
        print(f"Epoch {epoch}: G={epG/len(train_loader):.3f}, D={epD/len(train_loader):.3f}, L1={epL1/len(train_loader):.3f}")

        # TODO: Add checkpointing, FID, and sample saving here


if __name__ == "__main__":
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_TRAIN = Path("data/raw/train/train")
    DATA_VAL = Path("data/raw/test/test")

    train_loader = torch.utils.data.DataLoader(
        SideBySideDataset(DATA_TRAIN, True), batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        SideBySideDataset(DATA_VAL, False), batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    G = UNetGenerator_SPADE().to(device)
    D = PatchDiscriminatorSN().to(device)
    print(f"Params G: {sum(p.numel() for p in G.parameters())/1e6:.2f}M, D: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")

    cfg = dict(
        EPOCHS=250,
        LAMBDA_L1=35.0,
        LAMBDA_FM=5.0,
        LAMBDA_VGG=5.0,
        LR_G=2e-4,
        LR_D=2.5e-4
    )

    train_spade(G, D, train_loader, val_loader, cfg, device)
