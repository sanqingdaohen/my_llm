from torch.utils.data import DataLoader, Dataset
import json
import tiktoken
import torch
from pathlib import Path
import numpy as np
import random

def set_seed(seed=42):
    """
    固定所有相关的随机数种子，确保实验可复现。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作是确定性的
    torch.backends.cudnn.benchmark = False    # 关闭cuDNN的自动寻找最优算法，因为它可能不是确定性的

class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        super().__init__()
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        # 获取特殊标记的 token ID
        eos_ids = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )
        if not eos_ids:
            raise ValueError("Failed to resolve eos token id for '<|endoftext|>'. Please check tiktoken version/tokenizer.")
        self.eos_token_id = eos_ids[0]
        
        # 获取 <|im_end|> 对应的 token id 序列（可能不止一个 token）
        encoded_im_end = self.enc.encode(
            "<|im_end|>",
            allowed_special={"<|im_end|>"}
        )
        self.im_end_token_ids = encoded_im_end if encoded_im_end else None

        self.encoded_data = []
        self.max_lines = 20000

        # 1. 读取并解析 JSONL 文件
        all_conversations = []
        error_count = 0
        first_errors = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    obj = json.loads(line.strip())
                    # 你的数据里通常是 'text' 字段；额外提供 'content' 兜底
                    text = obj.get('text', obj.get('content', None))
                    if not text:
                        continue
                    
                    # 2. 按 <|im_end|> 分割对话
                    parts = text.split("<|im_end|>")
                    
                    # 3. 将分割后的各部分编码，并在它们之间插入 <|im_end|> 的token ID
                    encoded_conv = []
                    for j, part in enumerate(parts):
                        if part.strip(): # 只处理非空部分
                            encoded_part = self.enc.encode(part.strip())
                            encoded_conv.extend(encoded_part)
                            # 在每个部分后（除了最后一部分）添加 <|im_end|> token 序列
                            if j < len(parts) - 1 and self.im_end_token_ids is not None:
                                encoded_conv.extend(self.im_end_token_ids)
                    
                    if encoded_conv: # 如果编码后的序列不为空
                        all_conversations.append(encoded_conv)
                        
                except json.JSONDecodeError:
                    error_count += 1
                    if len(first_errors) < 5:
                        first_errors.append((i, "JSONDecodeError"))
                    continue
                except Exception as e:
                    error_count += 1
                    if len(first_errors) < 5:
                        first_errors.append((i, str(e)))
                    continue

        # 4. 将所有对话拼接成一个大的 token 序列
        full_encoded = []
        for conv in all_conversations:
            full_encoded.extend(conv)
            # 在每个对话单元后加上 EOS 标记，帮助模型识别边界
            full_encoded.append(self.eos_token_id)

        if not full_encoded:
            raise ValueError(
                f"No tokens found from dataset file: {path}. "
                f"all_conversations={len(all_conversations)} error_count={error_count} first_errors={first_errors}"
            )

        # 5. 将长序列切分成 block_size+1 长度的块，用于训练
        # 关键修复：覆盖到最后一段数据，避免漏掉最后一个切块起点。
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1]
            # 确保块的长度是 block_size + 1，否则用 eos_token_id 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token_id] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        # x 是输入序列，y 是目标序列 (x 向右移动一位)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)

if __name__ == "__main__":
    set_seed(42) # 你可以换成任意你喜欢的数字
    data_path = Path(__file__).parent
    dataset = MyDataset(data_path/'pretrain_hq.jsonl')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset, batch_size=64,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)

    sample_batch_x, sample_batch_y = next(iter(train_dataloader))

    print(f"一个批次的输入 x 形状: {sample_batch_x.shape}") # 应该是 [64, block_size]
    print(f"一个批次的目标 y 形状: {sample_batch_y.shape}") # 应该是 [64, block_size]
    print("-" * 20)
    print("一个批次的输入 x (第一个样本): ", sample_batch_x[0][:10].tolist()) # 打印第一个样本的前10个token ID
    print("一个批次的目标 y (第一个样本): ", sample_batch_y[0][:10].tolist()) # 打印目标的前10个token ID