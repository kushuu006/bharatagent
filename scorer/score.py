"""
Model output format (JSON only):
  {"calls": [{"tool": "get_weather", "args": {"city": "Udaipur"}}]}   # use tools
  {"ask": "Kisko aur kitne paise bhejne hain?"}                      # missing info
  {"answer": "..."}                                                   # no tool needed
Gold has the same format. Score is 0.0 to 1.0.
"""
import json, re

def parse(text):
    """Extract JSON object from model text. Returns dict or None."""
    if isinstance(text, dict):
        return text
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def norm(v):
    return v.strip().lower() if isinstance(v, str) else v

def norm_args(a):
    return {k: norm(v) for k, v in a.items()}

def kind(d):
    return "calls" if d.get("calls") else "ask" if "ask" in d else "answer"

def score(output, gold):
    """Returns dict with total score and parts."""
    out = parse(output)
    r = {"valid_json": 0.0, "action": 0.0, "tools": 0.0, "args": 0.0, "total": 0.0}
    if not isinstance(out, dict):
        return r
    r["valid_json"] = 1.0
    if kind(out) != kind(gold):
        r["total"] = 0.1          # valid JSON but wrong behaviour
        return r
    r["action"] = 1.0
    if kind(gold) != "calls":
        r["tools"] = r["args"] = r["total"] = 1.0
        return r
    oc, gc = out["calls"], gold["calls"]
    if len(oc) != len(gc):
        r["total"] = 0.3
        return r
    tool_ok = [o.get("tool") == g["tool"] for o, g in zip(oc, gc)]
    arg_ok = [t and norm_args(o.get("args", {})) == norm_args(g["args"]) for t, o, g in zip(tool_ok, oc, gc)]
    r["tools"] = sum(tool_ok) / len(gc)
    r["args"] = sum(arg_ok) / len(gc)
    r["total"] = round(0.2 + 0.3 * r["tools"] + 0.5 * r["args"], 3)
    return r

def evaluate(outputs, golds):
    rows = [score(o, g) for o, g in zip(outputs, golds)]
    n = len(rows)
    res = {k: round(sum(r[k] for r in rows) / n, 4) for k in rows[0]}
    res["exact_match"] = round(sum(r["total"] == 1.0 for r in rows) / n, 4)
    return res

if __name__ == "__main__":
    gold = {"calls": [{"tool": "get_weather", "args": {"city": "Udaipur"}}]}
    print(score('{"calls":[{"tool":"get_weather","args":{"city":"udaipur"}}]}', gold))
    print(score('not json', gold))
    print(score('{"ask":"kaunsa shehar?"}', gold))
