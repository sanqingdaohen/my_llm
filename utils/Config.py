from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Advanced_Model_Config:
    # --- 基础架构参数 ---
    n_embed: int = 768          # Embedding 维度 (Hidden Size)
    n_layers: int = 12          # Transformer 层数
    n_heads: int = 12           # Query 头数量 (Total Query Heads)
    max_seq_len: int = 256      # 最大上下文长度 (用于初始化 KV Cache)
    batch_size: int = 32        # 批次大小
    vocab_size: int = 50257     # 词表大小
    dropout: float = 0.1        # Dropout 比例
    bias: bool = True           # 线性层是否使用 bias
    
    # 自动计算 Head 维度 (Head Dim = Embed / Heads)
    # 注意：在 dataclass 中默认值不能直接依赖同类的其他字段进行复杂计算，
    # 通常建议在 __post_init__ 中处理，或者这里直接写死逻辑表达式（如果字段顺序允许）
    # 这里为了保持简洁，我们假设 n_embed 能被 n_heads 整除
    head_dim: int = field(init=False) 
    
    # --- GQA (Grouped Query Attention) 配置 ---
    # KV 头数量。如果等于 n_heads，则是标准 MHA；如果为 1，则是 MQA；如果在中间，则是 GQA
    n_kv_heads: int = 4         # 例如：12 个 Q 头，4 个 KV 头 -> 每 3 个 Q 共享 1 组 KV
    
    # --- MOE (Mixture of Experts) 配置 ---
    use_moe: bool = True        # 是否启用 MoE
    num_experts: int = 8        # 路由专家总数 (Routed Experts)
    top_k: int = 2              # 每个 token 激活的路由专家数量
    
    # --- Shared MoE (共享专家) 配置 ---
    # 共享专家数量。如果 > 0，则启用 Shared MoE 架构 (如 DeepSeek-V2/V3)
    # 这些专家对所有 token 始终激活，不经过门控路由
    n_shared_experts: int = 2   
    
    # 共享专家的中间层维度 (可选)
    # 如果为 None，则默认与路由专家相同 (通常是 4 * n_embed)
    # DeepSeek-V3 中共享专家往往更宽，以增强通用能力
    shared_expert_intermediate_size: Optional[int] = None 

    def __post_init__(self):
        # 在初始化后自动计算 head_dim
        self.head_dim = self.n_embed // self.n_heads
        
        # 基本校验
        assert self.n_embed % self.n_heads == 0, "n_embed must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
        
        if self.use_moe:
            assert self.top_k <= self.num_experts, "top_k cannot be greater than num_experts"
            if self.n_shared_experts > 0:
                print(f"[Config] Shared MoE Enabled: {self.n_shared_experts} shared experts + {self.num_experts} routed experts (Top-{self.top_k})")
            else:
                print(f"[Config] Standard MoE Enabled: {self.num_experts} routed experts (Top-{self.top_k})")
        else:
            print("[Config] Dense Model (No MoE)")

# --- 使用示例 ---
if __name__ == "__main__":
    # 实例化配置 (默认即开启了 Shared MoE 和 GQA)
    config = Advanced_Model_Config(
        n_embed=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,       # GQA: 4 组 KV
        use_moe=True,
        num_experts=8,
        top_k=2,
        n_shared_experts=2  # Shared MoE: 2 个共享专家
    )
    
    print(f"Head Dim: {config.head_dim}")
    print(f"KV Groups: {config.n_heads // config.n_kv_heads}")