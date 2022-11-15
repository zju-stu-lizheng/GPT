import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
 
def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index
  
# 自动加载预训练模型（权重）
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# 手动加载词表文件：merges.txt vocab.json
tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt-2') 
 
# 编码输入
text = "Yesterday, a man named Jack said he saw an alien,"
indexed_tokens = tokenizer.encode(text)
print("输入语句为：",tokenizer.decode(indexed_tokens))
tokens_tensor = torch.tensor([indexed_tokens])  # 将输入语句转换为张量
 
# 自动加载预训练模型（权重）
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# 手动加载：配置文件config.json 与 权重文件pytorch_model.bin
model = GPT2LMHeadModel.from_pretrained('./models/gpt-2/pytorch_model.bin',config='./models/gpt-2/config.json')
 
# 将模型设置为评估模式
model.eval()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokens_tensor = tokens_tensor.to(DEVICE)
model.to(DEVICE)

total_predicted_text = text
n = 100  # 预测过程的循环次数
for _ in range(n):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
 
    predicted_index = select_top_k(predictions, k=10)
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    total_predicted_text += tokenizer.decode(predicted_index)
 
    if '<|endoftext|>' in total_predicted_text:
        # 如果出现文本结束标志，就结束文本生成
        break
 
    indexed_tokens += [predicted_index]
    tokens_tensor = torch.tensor([indexed_tokens]).to(DEVICE)

print("输出语句为： "+total_predicted_text)
 
# # 预测所有标记
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     predictions = outputs[0]
 
# # 得到预测的下一词
# predicted_index = torch.argmax(predictions[0, -1, :]).item()
# predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
# print("输出语句为：",predicted_text) # GPT-2模型没有为输入文本添加特殊词。
 