#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 加载 Vicuna（纯文本）模型
    tok = AutoTokenizer.from_pretrained(
        "lvwerra/vicuna-13b-v1.3",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "lvwerra/vicuna-13b-v1.3",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 输入与生成
    inputs = tok("Hello world", return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )
    print(tok.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
