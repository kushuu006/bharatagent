"""
Evaluate any model on data/test.jsonl.
Edit `generate()` to call your model (HF model, vLLM, or API).
Run: python scorer/run_eval.py
"""
import json, sys
sys.path.insert(0, ".")
from scorer.score import evaluate
from tools.tools import SCHEMAS

SYSTEM = (
    "You are an assistant that can call tools. Available tools:\n"
    + json.dumps(SCHEMAS, ensure_ascii=False)
    + '\nReply with ONLY JSON in one of these formats:\n'
    '{"calls":[{"tool":"name","args":{...}}]}  to call tools,\n'
    '{"ask":"question"}  if information is missing,\n'
    '{"answer":"text"}  if no tool is needed.\n'
    "Never guess missing arguments."
)

def generate(query):
    """TODO: replace with real model call. Dummy baseline for testing the pipeline."""
    return '{"answer": "ok"}'

if __name__ == "__main__":
    rows = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
    outs = [generate(r["query"]) for r in rows]
    golds = [r["gold"] for r in rows]
    print("Overall:", evaluate(outs, golds))
    for lang in sorted({r["lang"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        print(lang, evaluate([outs[i] for i in idx], [golds[i] for i in idx]))
