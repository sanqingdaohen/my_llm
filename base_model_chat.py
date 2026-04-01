import torch
import torch.nn.functional as F
import tiktoken
import os
import sys
from utils.basemode import GPT,GPTconfig 


# ======================
# 2. 配置与加载模型
# ======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r'./model_output/best_model.pth'
#MODEL_PATH = r'./model_output/best_advance_model.pth'
if not os.path.exists(MODEL_PATH):
    print(f"❌ 模型文件 '{MODEL_PATH}' 不存在，请检查路径。")
    sys.exit(1)

# ⚠️ 必须与训练时完全一致！
config = GPTconfig()

print("🧠 正在加载模型...")
model = GPT(config).to(DEVICE)
try:
    # weights_only=True 更安全（PyTorch >= 2.4）
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    sys.exit(1)

model.eval()
print(f"✅ 模型已加载到 {DEVICE}，准备就绪！\n")

# ======================
# 3. Tokenizer
# ======================
enc = tiktoken.get_encoding("gpt2")

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)

# ======================
# 4. 生成函数（带基础采样）
# ======================
@torch.no_grad()
def generate_response(prompt: str, max_new_tokens: int = 60, temperature: float = 0.7) -> str:
    """
    从给定 prompt 生成回复
    """
    tokens = encode(prompt)
    if len(tokens) > config.max_seq_len - max_new_tokens:
        tokens = tokens[-(config.max_seq_len - max_new_tokens):]
    
    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    
    # 使用你模型自带的 generate 方法（需确保它支持 temperature）
    # 如果没有，我们在这里实现一个简单版本
    for _ in range(max_new_tokens):
        if x.size(1) >= config.max_seq_len:
            break
        logits, _ = model(x)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
    
    new_tokens = x[0, len(tokens):].tolist()
    response = decode(new_tokens)
    
    # 简单截断：遇到换行或句号停止
    for stop in ["\n", ".", "?", "!"]:
        if stop in response:
            response = response.split(stop)[0] + stop
            break
    
    return response.strip()

# ======================
# 5. 聊天主循环
# ======================
def main():
    print("💬 欢迎使用自研聊天机器人！")
    print("📌 提示：本模型为基础语言模型，非专用对话模型，回答可能不完美。")
    print("⌨️  输入 'quit' 或 'exit' 退出。\n")
    
    history = []  # 存储对话历史
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit']:
                print("👋 再见！")
                break
            
            # 构建 prompt（简单 Human/AI 格式）
            prompt = ""
            for msg in history[-4:]:  # 只保留最近2轮（4条消息）避免超长
                role = "Human" if msg["role"] == "user" else "AI"
                prompt += f"{role}: {msg['content']}\n"
            prompt += f"Human: {user_input}\nAI:"
            
            print("AI: ", end="", flush=True)
            response = generate_response(prompt, max_new_tokens=80, temperature=0.8)
            print(response)
            
            # 保存历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            break

if __name__ == "__main__":
    main()