from data.mydataset import MyDataset,set_seed
from utils.basemode import GPTconfig,GPT
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import random_split,DataLoader


def train_epoch(model, optimizer, train_dataloader, scheduler, device):
    model.train()
    total_loss = 0
    
    loop = tqdm(train_dataloader, desc="Training", leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        # 前向传播
        logits, loss = model(x, targets = y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        # 可选：在进度条上实时显示当前批次的损失
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def eval(model, val_dataloader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_dataloader, desc='Evaluating', leave=True):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x,targets =  y)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    return avg_val_loss


def train():
    set_seed(42)
    config = GPTconfig()
    # 1.构建数据集和训练集
    data_path = r'./data/pretrain_hq.jsonl'
    dataset = MyDataset(data_path,config.max_seq_len)

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset,config.batch_size,shuffle=True)
    val_dataloader = DataLoader(val_dataset,config.batch_size)

    # 2.其他配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(config).to(device)
    # 打印模型一共有多少参数

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # 设置 cosine 学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    # 初始化用于保存loss的列表
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # --- 确保模型保存目录存在 ---
    model_output_dir = Path('./model_output')
    model_output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_output_dir / 'best_model.pth'
    print(best_model_path)

    
    for epoch in range(config.epochs):
        print(f'***********Epoch {epoch+1}/{config.epochs}************')
        train_loss = train_epoch(model, optimizer, train_dataloader, scheduler, device)
        val_loss = eval(model, val_dataloader, device)
        
        # 保存当前epoch的loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 检查是否需要保存当前的最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"新的最佳模型已保存，Val Loss: {best_val_loss:.4f}")
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, config.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.savefig(Path(__file__).parent / 'loss_curve.png')
    plt.show()

if __name__ == "__main__":
    train()


