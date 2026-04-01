import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

# 让 utils/ 下的 Config.py/Share_Moe.py 能被 AdvancedModel.py 内部的
# `from Config import ...` 正确找到
ROOT_DIR = Path(__file__).resolve().parent
UTILS_DIR = ROOT_DIR / "utils"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(UTILS_DIR))

from utils.AdvancedModel import Advanced_GPT  # noqa: E402
from utils.Config import Advanced_Model_Config  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"./model_output/best_advance_model.pth"
if not os.path.exists(MODEL_PATH):
    print(f"❌ 模型文件 '{MODEL_PATH}' 不存在，请检查路径。")
    sys.exit(1)

config = Advanced_Model_Config()

print("🧠 正在加载高级模型...")
model = Advanced_GPT(config).to(DEVICE)
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    sys.exit(1)

model.eval()
print(f"✅ 模型已加载到 {DEVICE}，准备就绪！\n")

enc = tiktoken.get_encoding("gpt2")


def encode(text: str):
    return enc.encode(text)


def decode(tokens):
    return enc.decode(tokens)


@torch.no_grad()
def generate_response(prompt: str, max_new_tokens: int = 60, temperature: float = 0.7) -> str:
    tokens = encode(prompt)

    # 防止上下文过长：只保留最后一段
    max_context = config.max_seq_len - max_new_tokens
    if len(tokens) > max_context:
        tokens = tokens[-max_context:]

    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        if x.size(1) >= config.max_seq_len:
            break

        logits, _, _ = model(x, targets=None)  # (logits, loss(None), aux_loss)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

    new_tokens = x[0, len(tokens) :].tolist()
    response = decode(new_tokens)

    # 简单截断：遇到常见结束符停止
    for stop in ["\n", ".", "?", "!"]:
        if stop in response:
            response = response.split(stop)[0] + stop
            break

    return response.strip()


def main():
    print("💬 欢迎使用自研高级聊天机器人！")
    print("📌 提示：该模型是语言建模形式（非专用指令对齐），回答可能不稳定。\n")
    print("输入 'quit' 或 'exit' 退出。\n")

    history = []  # [{"role": "user"/"assistant", "content": "..."}]

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("👋 再见！")
                break

            # 构建 prompt：只保留最近两轮（4 条消息）
            prompt = ""
            for msg in history[-4:]:
                role = "Human" if msg["role"] == "user" else "AI"
                prompt += f"{role}: {msg['content']}\n"
            prompt += f"Human: {user_input}\nAI:"

            print("AI: ", end="", flush=True)
            response = generate_response(prompt, max_new_tokens=80, temperature=0.8)
            print(response)

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

