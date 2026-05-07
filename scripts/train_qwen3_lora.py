"""Kaggle script for Qwen3:8B LoRA fine-tuning.

This file documents the notebook experiment used in the thesis. It is intended
to be copied into a Kaggle notebook with GPU T4 x2 enabled. Paths correspond to
the uploaded Kaggle dataset with qwen3_rag_lora_train.jsonl and
qwen3_rag_lora_valid.jsonl.
"""

# Kaggle setup cell:
# !pip install -U transformers datasets peft trl accelerate bitsandbytes

from __future__ import annotations

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch


MODEL_ID = "Qwen/Qwen3-8B"
TRAIN_PATH = "/kaggle/input/datasets/ffuxkyouu/rdfrag-lora/qwen3_rag_lora_train.jsonl"
VALID_PATH = "/kaggle/input/datasets/ffuxkyouu/rdfrag-lora/qwen3_rag_lora_valid.jsonl"
OUTPUT_DIR = "/kaggle/working/qwen3-rdfrag-lora"


def main() -> None:
    print("CUDA:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_PATH,
            "validation": VALID_PATH,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def to_text(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(to_text)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
