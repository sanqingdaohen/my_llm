from typing import Any


import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from data.mydataset import MyDataset, set_seed

# 让 utils 目录里的模块可以被正确导入（utils/AdvancedModel.py 内部使用了 from Config import ...）
ROOT_DIR = Path(__file__).resolve().parent
UTILS_DIR = ROOT_DIR / "utils"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(UTILS_DIR))

from utils.AdvancedModel import Advanced_GPT  # noqa: E402
from utils.Config import Advanced_Model_Config  # noqa: E402


def train_epoch(model, optimizer, train_dataloader, scheduler, device):
    model.train()
    total_loss = 0.0

    loop = tqdm(train_dataloader, desc="Training", leave=True)
    for _, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        logits, ce_loss, aux_loss = model(x, targets=y)
        total_batch_loss = ce_loss + aux_loss

        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = total_batch_loss.item()
        total_loss += loss_val
        loop.set_postfix(loss=loss_val)

    return total_loss / len(train_dataloader)


@torch.no_grad()
def eval(model, val_dataloader, device):
    model.eval()
    total_loss = 0.0

    for x, y in tqdm(val_dataloader, desc="Evaluating", leave=True):
        x, y = x.to(device), y.to(device)
        _, ce_loss, aux_loss = model(x, targets=y)
        total_batch_loss = ce_loss + aux_loss
        total_loss += total_batch_loss.item()

    return total_loss / len(val_dataloader)


def train():
    set_seed(42)

    config = Advanced_Model_Config()

    # Advanced_Model_Config 里没有 epochs，这里沿用 base_model_train.py 的习惯。
    epochs = 10

    # 1. 构建数据集和训练集
    data_path = r"./data/pretrain_hq.jsonl"
    dataset = MyDataset(data_path, config.max_seq_len)

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, config.batch_size)

    # 2. 其他配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Advanced_GPT(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # cosine scheduler：按 batch 更新，T_max 设置为总步数
    steps_per_epoch = max(1, len(train_dataloader))
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    model_output_dir = Path("./model_output")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_output_dir / "best_advance_model.pth"
    print(best_model_path)

    for epoch in range(epochs):
        print(f"***********Epoch {epoch + 1}/{epochs}************")
        train_loss = train_epoch(model, optimizer, train_dataloader, scheduler, device)
        val_loss = eval(model, val_dataloader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"新的最佳模型已保存，Val Loss: {best_val_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Advance Training and Validation Loss Over Time")
    plt.savefig(Path(__file__).parent / "loss_curve_advance.png")
    plt.show()


if __name__ == "__main__":
    train()

