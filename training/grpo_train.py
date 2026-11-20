"""
Step 8: GRPO. Your scorer IS the reward function.
Run on Kaggle/Colab after SFT. Keep it small: few generations, short outputs.
Check the latest TRL GRPOTrainer docs if arguments differ in your version.
"""
import json, sys, torch
sys.path.insert(0, ".")
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
from scorer.score import score, parse
from scorer.run_eval import SYSTEM

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER = "outputs/sft"

rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
ds = Dataset.from_list([{
    "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["query"]}],
    "gold": json.dumps(r["gold"], ensure_ascii=False),
} for r in rows])

def _text(c):
    return c[0]["content"] if isinstance(c, list) else c

# --- reward functions: TRL passes extra dataset columns (gold) as kwargs ---
def reward_correct(completions, gold, **kw):
    return [score(_text(c), json.loads(g))["total"] * 2.0 for c, g in zip(completions, gold)]

def reward_valid_json(completions, **kw):
    return [0.5 if parse(_text(c)) is not None else 0.0 for c in completions]

def penalty_extra_text(completions, **kw):
    """Penalise text outside the JSON (we want JSON only)."""
    return [0.0 if _text(c).strip().startswith("{") and _text(c).strip().endswith("}") else -0.3
            for c in completions]

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto", torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16))
model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)

trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    reward_funcs=[reward_correct, reward_valid_json, penalty_extra_text],
    train_dataset=ds,
    args=GRPOConfig(
        output_dir="outputs/grpo",
        num_generations=4,                 # keep small for free GPU
        max_completion_length=128,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        num_train_epochs=1,
        logging_steps=5,
        fp16=True,
        report_to="none",
    ),
)
trainer.train()
trainer.save_model("outputs/grpo")
