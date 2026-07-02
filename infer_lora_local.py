import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--language", default="Korean", help="Language for the tutor persona/answer.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
    ).to("cuda")

    model.eval()

    messages = [
        {
            "role": "system",
            "content": f"너는 오픈소스 코드베이스를 {args.language} 언어로 설명하는 기술 튜터다.",
        },
        {
            "role": "user",
            "content": args.prompt,
        },
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        text = (
            f"<|system|>\n{messages[0]['content']}\n\n"
            f"<|user|>\n{args.prompt}\n\n"
            f"<|assistant|>\n"
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print(decoded)


if __name__ == "__main__":
    main()
