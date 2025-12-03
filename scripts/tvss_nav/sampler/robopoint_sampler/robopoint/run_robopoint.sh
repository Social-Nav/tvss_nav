#!/usr/bin/env bash
# 安装依赖（如还没装过）
pip install llava transformers==4.37.2 accelerate==0.21.0 sentencepiece==0.1.99 tokenizers==0.15.1

# 写入并执行 Python 脚本
python3 - << 'PYCODE'
import llava                             # 确保注册 custom model
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 1. 加载 config/tokenizer
cfg = AutoConfig.from_pretrained("wentao-yuan/robopoint-v1-vicuna-v1.5-13b", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("wentao-yuan/robopoint-v1-vicuna-v1.5-13b", trust_remote_code=True)

# 2. 加载模型（自动分片，不手动 .to()）
model = AutoModelForCausalLM.from_pretrained(
    "wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
    config=cfg,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# 3. 准备输入
prompt = "Hello world"
inputs = tok(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs.get("attention_mask", None)
if attention_mask is not None:
    attention_mask = attention_mask.to(model.device)

# 4. 生成：**注意把 input_ids 作为第一个位置参数**传给 generate
output_ids = model.generate(
    input_ids,                     # 位置参数
    attention_mask=attention_mask, # 关键词参数
    do_sample=False,
    max_new_tokens=100,
)

# 5. 打印
print(tok.decode(output_ids[0], skip_special_tokens=True))
PYCODE

