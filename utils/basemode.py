import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import math
from dataclasses import dataclass
import json
import tiktoken


@dataclass
class GPTconfig:
    n_embed: int =768 #embedding维度
    n_layers: int = 12 
    n_heads : int = 12
    max_seq_len : int =256 # 最大上下文
    batch_size : int = 32
    head_dim : int = n_embed//n_heads #768//12 = 64
    dropout: float = 0.1
    vocab_size = 50257 # 词表
    epochs = 5

#print(GPTconfig())

class SingleHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.q = nn.Linear(config.n_embed,config.head_dim)
        self.k = nn.Linear(config.n_embed,config.head_dim)
        self.v = nn.Linear(config.n_embed,config.head_dim)

        self.register_buffer(
            'attention_mask',
            torch.tril(
                torch.ones(config.max_seq_len,config.max_seq_len,dtype=torch.bool)
            )
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        batch_size,seq_len,hidden_dim = x.size() #[b,s,n_embed]

        q = self.q(x) #[b,s,h_dim]
        k = self.k(x) #[b,s,h_dim]
        v = self.v(x) #[b,s,h_dim]
        # 注意这里除的是k.size(-1)而不是x.size(-1)
        weights = q@k.transpose(-2,-1)/math.sqrt(k.size(-1))#[b,s,s]

        # 尝试学习新的写法，attention_mask 通过 register_buffer 注册
        # 因为不用计算 梯度，所以节约内存和显存，速度也更快
        weights = weights.masked_fill(
            # 卡在这里很久了
            self.attention_mask[:seq_len,:seq_len]==0,
            float('-inf')
        )
        
        socres = F.softmax(weights,dim=-1)#[b,s,s]
        socres = self.dropout(socres)
        output = socres@v #[b,s,h_dim]
        return output

class MultiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 因为要对每一个单头的数据拼接，这个过程使用到了单头的数据
        # 所以使用ModuleList（列表）而不用Sequential（需要解码）
        self.heads = nn.ModuleList(
            [
                SingleHead(config)
                for _ in range(config.n_heads)
            ]
        )
        self.proj = nn.Linear(config.n_embed,config.n_embed,)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        weights = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )#[b,s,n_embed]
        output = self.proj(weights)#[b,s,n_embed]
        return self.dropout(output)

# #########验证代码#########   
# x = torch.rand(4,12,768)
# model = MultiHead(GPTconfig())
# y = model(x)
# print(y.shape)
# #########验证代码#########   

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed,4*config.n_embed),
            nn.GELU(),
            nn.Linear(4*config.n_embed,config.n_embed),
            nn.Dropout(config.dropout)
        )
        
    def forward(self,x):
        return self.net(x) #[b,s,n_embed]

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.atte = MultiHead(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ffn = FeedForward(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
    def forward(self,x):
        x = x + self.atte(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
       
        self.embedding = nn.Embedding(config.vocab_size,config.n_embed)
        self.pos = nn.Embedding(config.max_seq_len,config.n_embed)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layers)]
        )

        self.layernormal = nn.LayerNorm(config.n_embed)
        self.lin = nn.Linear(config.n_embed,config.vocab_size,bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 这里使用的是正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self,idx,targets=None):
        batch,seq_len = idx.size() #[b,s]
        token_emb = self.embedding(idx) #[b,s.n_embed]
        
        # 注意这里传递的不是idx，而是seq_len
        pos_idx = torch.arange(seq_len, device=idx.device) # [0, 1, 2, ..., seq_len-1]
        token_pos = self.pos(pos_idx) # [s, n_embed]

        x = token_emb+token_pos #[b,s,n_embed]

        x = self.blocks(x) #[b,s,n_embed]
        x = self.layernormal(x)
        logits = self.lin(x) # shape is (batch, seq_len, vocab_size)

        if targets is None:
            loss = None
        else:
            batch,seq_len,vocab_size = logits.size()
            logits = logits.view(batch*seq_len,vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits,loss
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后 block_size 个token
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # 获取预测
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTconfig()
    model = GPT(config).to(device)
    # 测试输入
    B, T = 2, 10  # 小批量和短序列，方便测试
    dummy_input = torch.randint(0, config.vocab_size, (B, T)) # 随机生成 token IDs
    dummy_targets = torch.randint(0, config.vocab_size, (B, T)) # 随机生成目标 IDs

    print("--- 1. 模型基本结构检查 ---")
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n--- 2. 推理模式 (targets=None) ---")
    try:
        logits, loss = model(dummy_input)
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  输出 logits 形状: {logits.shape}")
        print(f"  损失 (loss): {loss}")
        print("  推理模式: OK")
    except Exception as e:
        print(f"  推理模式: FAILED - {e}")

    print("\n--- 3. 训练模式 (targets=targets) ---")
    try:
        logits, loss = model(dummy_input, dummy_targets)
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  目标形状: {dummy_targets.shape}")
        print(f"  输出 logits 形状: {logits.shape}")
        print(f"  损失 (loss): {loss.item():.4f}") # .item() 获取具体的数值
        print("  训练模式: OK")
    except Exception as e:
        print(f"  训练模式: FAILED - {e}")