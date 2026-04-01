import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from Config import Advanced_Model_Config
from Share_Moe import SharedMoE

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_group_num = self.n_heads // self.n_kv_heads

        # Q 投影
        self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        
        # K,V 投影 (合并优化)
        kv_dim = self.n_kv_heads * self.head_dim
        self.kv_proj = nn.Linear(config.n_embed, 2 * kv_dim, bias=False)
        
        # 输出投影
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=True)
        
        self.atten_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, input_pos=None, cache_k=None, cache_v=None):
        B, T, C = x.size()  # [Batch, Seq_Len, Embed_Dim]

        # 1. 线性投影
        q = self.q_proj(x)          # [B, T, C]
        kv = self.kv_proj(x)        # [B, T, 2 * kv_dim]=[B,T,2*kv_heads*head_dim]
        
        # 2. 拆分 K 和 V
        k, v = kv.split(self.n_kv_heads * self.head_dim, dim=-1) 
        # tensor.split(份数,dim)就是对tensor的dim维度上按照每一份的份数来分开
        # k: [B, T, kv_dim], v: [B, T, kv_dim]

        # 3. 重塑多头形状 (View + Transpose)
        # Q: [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # K, V: [B, T, n_kv_heads, head_dim] -> [B, n_kv_heads, T, head_dim]
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 4. KV Cache 更新 (仅在推理/解码阶段)
        if input_pos is not None:
            if cache_k is None or cache_v is None:
                raise ValueError("In inference mode, cache_k and cache_v must be provided.")
            
            # cache_k shape: (B, n_kv_heads, max_seq_len, head_dim)
            # input_pos shape: (B, T) or (T,) - 通常包含当前生成的位置索引
            
            # 将当前的 k, v 复制到 cache 的指定位置
            # index_copy_(dim, index, source)
            cache_k.index_copy_(2, input_pos, k)
            cache_v.index_copy_(2, input_pos, v)
            #tonsor.index_cpoy_(dim,index,source) 就是把source塞到tensor dim维度的第index位置上面去
            #a = torch.zeors(3,3)   b = torch.tensor([1,2,3])
            #a.index_copy_(0,1,b) ->a=[[0,0,0],[1,2,3],[0,0,0]]
            #
            # 【关键逻辑修正】: 获取从 0 到当前位置的所有历史
            # input_pos[-1] 获取当前批次中最大的位置索引 (假设是递增的)
            # 注意：如果 input_pos 不是排序的或者 batch 内位置不一致，这里需要更复杂的处理
            # 通常对于单步推理 (T=1)，input_pos 是一个标量或 [pos]
            current_max_pos = input_pos.max().item() + 1
            k = cache_k[:, :, :current_max_pos, :] # [B,kv_heads,history_len,head_dim]
            v = cache_v[:, :, :current_max_pos, :] # [B,kv_heads,history_len,head_dim]

        # 5. GQA: 重复 K, V 以匹配 Q 的头数
        if self.n_kv_heads != self.n_heads:
            # ✅ 修正了这里的参数错误
            k = k.repeat_interleave(self.kv_group_num, dim=1)
            v = v.repeat_interleave(self.kv_group_num, dim=1)
            # 现在 k, v 的形状变为: [B, n_heads, T_effective, head_dim]

        # 6. 注意力计算
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.atten_dropout.p if self.training else 0.0,
            is_causal=(input_pos is None) # 训练时因果掩码，推理时靠 Cache 保证因果
        ) 
        # y shape: [B, n_heads, T, head_dim]

        # 7. 重组并输出
        y = y.transpose(1, 2).contiguous().view(B, T, C) # [B, T, C]
        
        return self.resid_dropout(self.c_proj(y))
    
#######预测代码#########


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化 RMSNorm
        
        Args:
            dim: 输入特征的维度 (例如 config.n_embed)
            eps: 防止除以零的小常数，通常设为 1e-5 或 1e-6
        """
        super().__init__()
        self.eps = eps
        # weight 是可学习参数 gamma，形状为 (dim,)，初始化为 1
        self.weight = nn.Parameter(torch.ones(dim)) #[n_embed]

    def _norm(self, x: torch.Tensor):
        """
        计算 RMS 并归一化
        公式: x / sqrt(mean(x^2) + eps)
        """
        # 1. 计算平方和的均值 (沿最后一个特征维度)
        # x.pow(2) -> 每个元素平方
        # .mean(-1, keepdim=True) -> 对最后一维求平均，保持维度以便广播
        rms = x.pow(2).mean(-1, keepdim=True) #这是对最后一个维度求mean，所以输出的形状是[batch_size,seq_len,1]
        
        # 2. 开根号并加上 epsilon
        rms = torch.sqrt(rms + self.eps)
        
        # 3. 归一化
        return x / rms #由于广播机制，最后输出的形状就是[B，T，H]

    def forward(self, x: torch.Tensor):
        # 1. 归一化 (输出形状与输入相同)
        # x.shape  [B,T,C]
        output = self._norm(x.float())  # 使用 float32 计算以保证稳定性
        
        # 2. 应用可学习缩放参数 (weight)
        # self.weight 会自动广播到所有维度
        output = output * self.weight
        
        # 3. 如果输入是 float16/bfloat16，转回原类型
        return output.type_as(x)
    
class FeedForward(nn.Module):
    def __init__(self, config:Advanced_Model_Config):
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
    def __init__(self, config:Advanced_Model_Config):
        super().__init__()
        # RMSNorm代替LayerNorm
        self.ln1 = RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embed)
        self.use_moe = config.use_moe
        # 动态选择FFN或者MOE
        if config.use_moe:
            self.ffn = SharedMoE(config)
        else:
            self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None, 
                cache_k: Optional[torch.Tensor] = None, cache_v: Optional[torch.Tensor] = None):
        # Pre-Norm Architecture
        x = x + self.attn(self.ln1(x), input_pos=input_pos, cache_k=cache_k, cache_v=cache_v)
        layer_aux_loss = None
        if self.use_moe:
            moe_out,layer_aux_loss = self.ffn(self.ln2(x))
            x= x+moe_out
        else:
            x = x + self.ffn(self.ln2(x))
            layer_aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return x,layer_aux_loss




class Advanced_GPT(nn.Module):
    def __init__(self,config:Advanced_Model_Config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        
        self.token_embedding = nn.Embedding(config.vocab_size,config.n_embed)
        self.position_embedding = nn.Embedding(config.max_seq_len,config.n_embed)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False)

        # 优化，节省内存
        self.lm_head.weight = self.token_embedding.weight 

        self.apply(self._init_weights)

        # 预分配 KV Cache (用于推理)
        # 形状: [Batch, n_kv_heads, max_seq_len, head_dim]
        self.register_buffer('cache_k', torch.zeros(
            config.batch_size, config.n_kv_heads, config.max_seq_len, config.head_dim
        ))
        self.register_buffer('cache_v', torch.zeros(
            config.batch_size, config.n_kv_heads, config.max_seq_len, config.head_dim
        ))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                input_pos: Optional[torch.Tensor] = None):
        """
        Args:
            idx: Input token IDs [B, T]
            targets: Target token IDs [B, T] (for training loss)
            input_pos: Position indices [B, T] or [T] (for inference with cache)
        """
        B, T = idx.size()
        device = idx.device
        
        # 1. Embeddings
        tok_emb = self.token_embedding(idx)
        
        # 位置编码处理
        if input_pos is None:
            # 训练模式 / 预填充模式：使用 0 到 T-1 的位置
            pos_ids = torch.arange(T, device=device)
            pos_emb = self.position_embedding(pos_ids)
        else:
            # 推理模式 (单步)：使用传入的具体位置
            # input_pos 可能是 [B, 1] 或 [1]，需要扩展以匹配 B
            if input_pos.dim() == 1:
                input_pos = input_pos.unsqueeze(0) # [1, T] -> [B, T] if B=1
            # 确保 pos_emb 能广播或正确选取
            # 这里简单处理：直接选取对应位置的 embedding
            pos_emb = self.position_embedding(input_pos) # [B, T, D]
            
        x = tok_emb + pos_emb

        # 2. Transformer Blocks
        # 如果是推理模式且使用了 cache，我们需要传入当前的 cache 切片
        # 注意：为了简化，这里我们在每个 block 内部处理 cache 的 index_copy
        # 但我们需要确保 cache 是干净的或者正确管理的。
        # 在实际推理循环中，通常会在外部管理 cache 的清零或偏移。
        # 这里为了演示，我们假设 cache_k/v 是类成员变量，并在 forward 中传递它们。
        # *重要*：在真实多步推理中，cache 需要在每一步之间保持状态。
        
        cache_k = self.cache_k[:B, :, :, :] if input_pos is not None else None
        cache_v = self.cache_v[:B, :, :, :] if input_pos is not None else None

        # 【新增】用于累加所有层的 Aux Loss
        total_aux_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for block in self.blocks:
            x,layer_aux_loss = block(x, input_pos=input_pos, cache_k=cache_k, cache_v=cache_v)
            total_aux_loss+=layer_aux_loss
        
        x = self.ln_f(x)
        
        # 3. Language Model Head
        logits = self.lm_head(x) # [B, T, Vocab]
        
        # 4. Loss Calculation
        loss = None
        if targets is not None:
            logits_view = logits.view(-1, self.config.vocab_size)
            targets_view = targets.view(-1)
            loss = F.cross_entropy(logits_view, targets_view, ignore_index=-1)
            
        return logits, loss,total_aux_loss
    def generate(self,x):
        pass


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Advanced_Model_Config(use_moe=True)
    model = Advanced_GPT(config).to(device)
    # 测试输入
    B, T = 2, 10  # 小批量和短序列，方便测试
    dummy_input = torch.randint(0, config.vocab_size, (B, T)) # 随机生成 token IDs
    dummy_targets = torch.randint(0, config.vocab_size, (B, T)) # 随机生成目标 IDs

    print("--- 1. 模型基本结构检查 ---")
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters())/1e6:.4f}M")

    print("\n--- 2. 推理模式 (targets=None) ---")
    try:
        logits, loss,aux_loss = model(dummy_input)
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  输出 logits 形状: {logits.shape}")
        print(f"  损失 (loss): {loss}")
        print("  推理模式: OK")
    except Exception as e:
        print(f"  推理模式: FAILED - {e}")

    print("\n--- 3. 训练模式 (targets=targets) ---")
    try:
        logits, loss,aux_loss = model(dummy_input, dummy_targets)
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  目标形状: {dummy_targets.shape}")
        print(f"  输出 logits 形状: {logits.shape}")
        print(f"  损失 (loss): {loss.item():.4f}") # .item() 获取具体的数值
        print("  训练模式: OK")
    except Exception as e:
        print(f"  训练模式: FAILED - {e}")

        print("\n" + "="*50)


    print("🔍 开始 MoE 专项诊断：负载平衡与专家利用率")
    print("="*50)

    # 使用你刚才成功的配置 (假设变量名保持一致)
    # 注意：这里需要确保你的模型能暴露出 "router_probs" 或类似信息
    # 如果你的模型内部没有记录，我们需要稍微修改一下 forward 或者利用钩子

    model.eval() # 先用评估模式看初始分布，或者用 train 模式看带噪声的分布
    model.train() # 为了看到真实的训练时路由情况，建议用 train 模式 (如果有噪声)

    # 构造一批随机数据 (模拟真实分布)
    B, T = 4, 32 # 稍大一点的序列，以便统计
    dummy_input = torch.randint(0, 50257, (B, T))
    dummy_targets = dummy_input.clone()

    # --- 核心检测逻辑 ---
    # 由于不同实现方式不同，这里有两种检测方案：

    # 方案 A: 如果你的 Forward 返回了 aux_loss 或 router_probs
    # 假设你的 model.forward 返回 (logits, loss, aux_loss, router_stats) 
    # 如果没有，请看方案 B (使用 Hook)

    try:
        # 尝试调用带有详细返回值的 forward (根据你的具体实现调整)
        # 这里假设你可能有一个方法可以获取最后的专家选择情况
        # 如果不确定，我们直接用 Hook 来抓取
        
        expert_counts = torch.zeros(8) # 假设有 8 个路由专家
        total_tokens = 0
        
        # 注册钩子来捕获 Router 的输出
        # 假设你的 MoE 层里有一个叫 'gate' 或 'router' 的模块，输出是 (B*T, num_experts)
        # 这里需要你根据实际类名调整，比如 'blocks.0.moe.gate'
        
        hooks = []
        captured_probs = []

        def get_hook(layer_name):
            def hook(module, input, output):
                # 假设 output 是 (tokens, num_experts) 的 logits 或 probs
                # 需要根据你的具体实现调整解析逻辑
                # 这里做一个通用假设：output 包含路由权重
                if isinstance(output, tuple):
                    probs = output[0] # 假设第一个是概率
                else:
                    probs = output
                
                # 如果是 logits，转 softmax
                if probs.dim() == 2 and probs.shape[-1] == 8: # 假设有8个专家
                    if probs.max().item() > 1.0: # 看起来像 logits
                        probs = F.softmax(probs, dim=-1)
                    captured_probs.append(probs.detach())
            return hook

        # ⚠️ 重要：你需要找到你代码中 MoE 路由模块的具体路径
        # 例如：model.blocks[0].moe.router
        # 下面是一个示例路径，请根据你的实际代码修改！
        # 如果不知道路径，可以打印 model 结构查找包含 'gate' 或 'router' 的层
        
        print("⚠️ 提示：为了精确统计，请在代码中找到 MoE 的 Router 层并注册钩子。")
        print("   如果你暂时无法定位，我们可以做一个简化的 '黑盒' 测试：")
        print("   观察训练时 Aux Loss 是否下降。")
        
        # --- 简化版黑盒测试：直接跑几步训练，看 Loss 组成 ---
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        print(f"\n开始运行 5 步训练，观察 Loss 变化...")
        for i in range(5):
            optimizer.zero_grad()
            # 假设你的 forward 返回 (logits, total_loss, aux_loss) 
            # 如果只返回 (logits, loss)，则无法直接分离，需检查代码
            try:
                out = model(dummy_input, dummy_targets)
                if isinstance(out, tuple) and len(out) >= 3:
                    logits, total_loss, aux_loss = out[0], out[1], out[2]
                    print(f"Step {i}: Total Loss = {total_loss.item():.4f}, Aux Loss = {aux_loss.item():.4f}")
                    if aux_loss.item() > 0:
                        print("   ✅ 检测到辅助损失 (Aux Loss) 正在工作！")
                    else:
                        print("   ⚠️ 辅助损失为 0，请检查是否启用了负载均衡损失。")
                else:
                    print(f"Step {i}: Loss = {out[1].item():.4f} (未检测到独立的 Aux Loss 返回)")
                    print("   💡 建议：修改 forward 函数，返回 (logits, ce_loss, aux_loss) 以便监控。")
                    
            except Exception as e:
                print(f"运行出错: {e}")
                break
                
        print("\n" + "="*50)
        print("📝 诊断建议:")
        print("1. 如果模型能跑通且 Loss 正常，下一步务必加入 '专家利用率监控'。")
        print("2. 在训练循环中，定期打印每个专家被选中的 Token 比例。")
        print("   理想情况：8 个专家，每个约占 12.5% (允许一定波动)。")
        print("   危险信号：某个专家 > 50%，其他专家 < 1%。")
        print("="*50)

    except Exception as e:
        print(f"诊断脚本报错: {e}")