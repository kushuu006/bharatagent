import json, sys, os, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from scorer.score import evaluate
from scorer.run_eval import SYSTEM

model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
adapter = sys.argv[2] if len(sys.argv) > 2 else None

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
model.eval()

def generate(query):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

rows = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
outs = [generate(r["query"]) for r in rows]
golds = [r["gold"] for r in rows]

print("Overall:", evaluate(outs, golds))
for lang in sorted({r["lang"] for r in rows}):
    idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
    print(lang, evaluate([outs[i] for i in idx], [golds[i] for i in idx]))

os.makedirs("results", exist_ok=True)
tag = (adapter or model_name).replace("/", "_")
json.dump([{"query": r["query"], "gold": r["gold"], "output": o} for r, o in zip(rows, outs)],
          open(f"results/{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Saved results/" + tag + ".json")
