#!/usr/bin/env bash
# 安装必要依赖（如没装过请取消注释）
# pip install llava transformers==4.37.2 accelerate==0.21.0 sentencepiece==0.1.99 tokenizers==0.15.1

python3 - << 'PYCODE'
import llava                                 # 注册 llava_llama
import torch, transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# —— 一些环境调试信息 —— 
print(">>> 环境检查 <<<")
print("transformers version:", transformers.__version__)
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "  devices:", torch.cuda.device_count())
print("当前 CUDA 设备:", torch.cuda.current_device(), torch.cuda.get_device_name(0))
print()

# —— 加载模型 —— 
print(">>> 加载模型 <<<")
cfg = AutoConfig.from_pretrained("wentao-yuan/robopoint-v1-vicuna-v1.5-13b", trust_remote_code=True)
print("config.model_type  =", cfg.model_type)
tok = AutoTokenizer.from_pretrained("wentao-yuan/robopoint-v1-vicuna-v1.5-13b", trust_remote_code=True)
print("tokenizer 用到 vocab_size=", tok.vocab_size)
model = AutoModelForCausalLM.from_pretrained(
    "wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
    config=cfg,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
# hf_device_map 存在的话也打印
if hasattr(model, "hf_device_map"):
    print("模型分布到设备:", model.hf_device_map)
print("模型总参数量(亿):", sum(p.numel() for p in model.parameters())/1e8)
print()

# —— 构造输入 —— 
prompt = "Hello world"
print(">>> 构造输入 <<<")
print("原始 prompt:", prompt)
inputs = tok(prompt, return_tensors="pt")
print("inputs keys:", list(inputs.keys()))
input_ids = inputs["input_ids"].to(model.device)
print("input_ids shape:", input_ids.shape, " dtype:", input_ids.dtype, " device:", input_ids.device)
attention_mask = inputs.get("attention_mask", None)
if attention_mask is not None:
    attention_mask = attention_mask.to(model.device)
    print("attention_mask shape:", attention_mask.shape)
print()

# —— 生成推理 —— 
print(">>> 开始生成 <<<")
try:
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=128,
    )
    print("生成完成，output_ids shape:", output_ids.shape)
    decoded = tok.decode(output_ids[0], skip_special_tokens=True)
    print(">>> 解码结果 <<<")
    print(decoded)
except Exception as e:
    print("!!! 生成出错:", type(e).__name__, e)
    import traceback; traceback.print_exc()

PYCODE
