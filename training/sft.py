"""
Step 7: SFT with QLoRA. Run on Kaggle/Colab GPU (T4).
pip install -U transformers datasets trl peft bitsandbytes accelerate
NOTE: TRL's API changes between versions. If an argument errors, check the
latest TRL SFTTrainer docs; the structure stays the same.
"""
import json, sys, torch
sys.path.insert(0, ".")
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from scorer.run_eval import SYSTEM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # small and good at multilingual; swap if you want

def to_chat(r):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": r["query"]},
        {"role": "assistant", "content": json.dumps(r["gold"], ensure_ascii=False)},
    ]}

rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
ds = Dataset.from_list([to_chat(r) for r in rows])

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16),
    device_map="auto",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    train_dataset=ds,
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules="all-linear", task_type="CAUSAL_LM"),
    args=SFTConfig(
        output_dir="outputs/sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        fp16=True,
        report_to="none",
    ),
)
trainer.train()
trainer.save_model("outputs/sft")   # saves LoRA adapter
print("Saved to outputs/sft")
