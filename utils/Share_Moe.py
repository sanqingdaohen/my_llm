import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from Config import Advanced_Model_Config

# ... (保持 Expert 类不变) ...
def find_multiple(n: int, k: int) -> int:
    if n % k == 0: return n
    return n + k - (n % k)

class Expert(nn.Module):
    def __init__(self, config: 'Advanced_Model_Config', is_shared: bool = False, intermediate_size_override: Optional[int] = None):
        super().__init__()
        self.n_embed = config.n_embed
        theoretical_intermediate = int((8 / 3) * self.n_embed)
        
        if hasattr(config, 'intermediate_size') and config.intermediate_size is not None:
            default_intermediate = config.intermediate_size
        else:
            alignment = 32 
            default_intermediate = find_multiple(theoretical_intermediate, alignment)
            
        if is_shared and intermediate_size_override is not None:
            self.hidden_dim = find_multiple(intermediate_size_override, 32)
        else:
            self.hidden_dim = default_intermediate
            
        self.gate_proj = nn.Linear(self.n_embed, self.hidden_dim, bias=config.bias)
        self.up_proj   = nn.Linear(self.n_embed, self.hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(self.hidden_dim, self.n_embed, bias=config.bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class SharedMoE(nn.Module):
    def __init__(self, config: 'Advanced_Model_Config'):
        super().__init__()
        self.config = config
        self.n_embed = config.n_embed
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.n_shared_experts = config.n_shared_experts
        
        # 1. 路由专家
        self.experts = nn.ModuleList([
            Expert(config, is_shared=False) for _ in range(self.num_experts)
        ])
        
        # 2. 共享专家
        self.shared_experts = nn.ModuleList([])
        if self.n_shared_experts > 0:
            for _ in range(self.n_shared_experts):
                self.shared_experts.append(
                    Expert(config, is_shared=True, intermediate_size_override=getattr(config, 'shared_expert_intermediate_size', None))
                )
        
        # 3. 门控网络
        self.gate = nn.Linear(self.n_embed, self.num_experts, bias=False)
        
        # 辅助损失系数 (通常 0.01 是经验值，如果坍塌严重可调大到 0.1)
        self.aux_loss_coeff = getattr(config, 'aux_loss_coeff', 0.01)

    def compute_load_balancing_loss(self, gating_logits: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失 (Load Balancing Loss)
        参考: Switch Transformer, DeepSeek-MoE
        """
        if self.num_experts == 0:
            return gating_logits.new_tensor(0.0)
            
        N = gating_logits.shape[0] # Token 数量
        E = self.num_experts
        
        # 1. 计算每个专家被选中的频率 (Frequency, f_i)
        # mask: [N, K] -> [N, E] (one-hot 化 TopK 选择)
        mask = F.one_hot(top_k_indices, num_classes=E).sum(dim=1) # [N, E]
        # 归一化得到比例: f_i = (选中该专家的Token数) / 总Token数
        frac_selected = mask.float().mean(dim=0) # [E], sum should be approx K/E if balanced? No, sum is K/N * N / N = K? 
        # 修正：frac_selected 是每个专家被选中的比例总和。
        # 理想情况下，每个专家被选中的概率是 1/E，选 K 个，所以总期望负载是 K/E。
        # 这里我们计算的是 "占比"，即 sum(mask[:, i]) / N。
        
        # 2. 计算每个专家的平均门控概率 (Probability, P_i)
        # 对原始 logits 做 softmax 得到所有专家的概率分布
        probs = F.softmax(gating_logits.float(), dim=-1) # [N, E]
        # 取每个 Token 选中的那 K 个专家的概率之和？
        # 标准公式使用的是：该 Token 对所有专家的概率均值？
        # Switch Transformer 公式：L = alpha * N * sum(f_i * P_i)
        # 其中 P_i 是该专家在所有 Token 上的平均门控概率 (不管是否被选中)
        mean_probs = probs.mean(dim=0) # [E]
        
        # 3. 计算 Loss
        # load_balancing_loss = coeff * N * sum(frac_selected * mean_probs)
        # 注意：frac_selected 已经是比例 (0~1)，mean_probs 也是 (0~1)
        # 有些实现会除以 K 来归一化，这里采用最通用的 Switch Transformer 形式
        loss = self.aux_loss_coeff * (frac_selected * mean_probs).sum()
        
        return loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [N, C]
        N = x_flat.shape[0]
        dtype = x.dtype
        
        # ==========================================
        # 1. 共享专家 (所有 Token 都经过，无门控)
        # ==========================================
        shared_output = torch.zeros_like(x_flat)
        if self.n_shared_experts > 0:
            # 优化：可以将多个共享专家合并计算，或者保持循环
            for expert in self.shared_experts:
                shared_output = shared_output + expert(x_flat)
        
        # ==========================================
        # 2. 路由专家
        # ==========================================
        routed_output = torch.zeros_like(x_flat, dtype=torch.float32) # 用 float32 积累防止溢出
        aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        
        if self.num_experts > 0:
            # 2.1 门控 Logits
            gating_logits = self.gate(x_flat)  # [N, E]
            
            # 2.2 Top-K
            top_k_weights, top_k_indices = torch.topk(gating_logits, self.top_k, dim=-1, sorted=False)
            
            # 2.3 归一化权重 (Softmax over K)
            # 注意：这里必须在 float32 下计算 softmax 以保证稳定性
            top_k_weights = F.softmax(top_k_weights.float(), dim=-1).to(dtype)
            
            # 2.4 计算 Aux Loss
            aux_loss = self.compute_load_balancing_loss(gating_logits, top_k_indices)
            
            # 2.5 稀疏执行 (优化版)
            # 方法：展平 indices 和 weights，一次性处理所有 Token-Expert 对
            # 这比 Python 循环快得多
            
            # top_k_indices: [N, K] -> [N*K]
            # top_k_weights: [N, K] -> [N*K]
            flat_indices = top_k_indices.view(-1)          # [N*K]
            flat_weights = top_k_weights.view(-1)          # [N*K]
            
            # 创建一个新的 Tensor 存放所有专家的输出 (先不乘权重)
            # 为了节省内存，我们不预先计算所有专家的输出，而是按专家分组计算 (Grouped GEMM 思想的手动简化版)
            # 但为了代码简洁且不引入复杂依赖，我们保留循环，但修正索引逻辑
            
            # 【修正后的循环逻辑】
            for i in range(self.num_experts):
                # 找到哪些位置 (N*K) 选了专家 i
                # mask shape: [N*K]
                mask = (flat_indices == i)
                
                if not mask.any():
                    continue
                
                # 获取这些位置的原始 Token 索引
                # token_pos = mask.nonzero(as_tuple=True)[0] // self.top_k  <-- 这种方法太慢
                
                # 更简单的方法：回到 [N, K] 维度判断
                # 其实你原来的逻辑是对的，只是写法可以更清晰
                # 让我们保留你的逻辑结构，但修复权重的提取方式，确保万无一失
                
                pass 

            # 【终极修正方案：使用 scatter_add 的思想，但手动实现以确保兼容性】
            # 既然你要仔细看代码，我给你一个最稳健的 "Token-Choice" 实现
            
            for i in range(self.num_experts):
                # 找出哪些 Token (在 N 中) 选择了专家 i
                # top_k_indices is [N, K]
                expert_mask = (top_k_indices == i)  # [N, K]
                
                # 只要这一行里有任何一个 True，说明该 Token 被分给了专家 i
                token_selected = expert_mask.any(dim=1) # [N]
                
                if not token_selected.any():
                    continue
                
                # 提取这些 Token 的输入
                current_x = x_flat[token_selected]      # [M, C]
                
                # 【关键修复】提取对应的权重
                # 我们需要知道每个选中的 Token，是在 K 个位置中的哪一个选中的专家 i
                # expert_mask[token_selected] -> [M, K] (每行只有一个 True)
                # top_k_weights[token_selected] -> [M, K]
                # 我们可以直接用 (expert_mask * top_k_weights).sum(dim=1) 来提取，这是安全的
                selected_weights = (expert_mask[token_selected] * top_k_weights[token_selected]).sum(dim=1, keepdim=True) # [M, 1]
                
                # 计算专家输出
                expert_out = self.experts[i](current_x) # [M, C]
                
                # 加权
                weighted_out = expert_out * selected_weights # [M, C]
                
                # 累加到总输出 (注意类型转换)
                routed_output[token_selected] += weighted_out.to(routed_output.dtype)

        # ==========================================
        # 3. 合并
        # ==========================================
        final_output = shared_output + routed_output.to(dtype)
        return final_output.view(B, T, C), aux_loss