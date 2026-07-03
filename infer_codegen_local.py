"""
Generate code with a locally fine-tuned (LoRA/QLoRA) small model.

Pairs with train_lora_local.py trained on a codegen dataset produced by the
Fine-tune tab (db_store.export_codegen_jsonl). Designed for small GPUs
(e.g. RTX 3050 Ti, 4GB) — pass --load_4bit to load the base in 4-bit.

Example:
  python infer_codegen_local.py \
    --adapter_dir finetuned_adapters/codegen_qwen05/adapter \
    --prompt "FastAPI 의존성 주입을 사용하는 엔드포인트 예시를 작성해줘" \
    --load_4bit
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CODEGEN_SYSTEM_PROMPT = (
    "You are a coding assistant that writes small, correct code units "
    "(functions, classes, config, usage snippets) in the style of the "
    "referenced open-source codebases. Output only the code in a fenced block."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--system", default=CODEGEN_SYSTEM_PROMPT)
    parser.add_argument("--load_4bit", action="store_true", help="Load base in 4-bit (for <=4GB VRAM).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to("cuda")

    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = (
            f"<|system|>\n{args.system}\n\n"
            f"<|user|>\n{args.prompt}\n\n"
            f"<|assistant|>\n"
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)


if __name__ == "__main__":
    main()
