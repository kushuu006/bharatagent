"""Fake (mock) tools. Same input -> same type of output, so testing is fair."""
import json, ast, operator as op
from pathlib import Path

SCHEMAS = json.loads((Path(__file__).parent / "schemas.json").read_text())
SCHEMA_BY_NAME = {s["name"]: s for s in SCHEMAS}

def get_weather(city):
    return {"city": city, "temp_c": 28, "condition": "sunny"}

def send_upi(to, amount):
    return {"status": "success", "to": to, "amount": amount, "txn_id": "TXN123456"}

def check_train(from_city, to_city, date):
    return {"trains": [{"name": "Express 12991", "from": from_city, "to": to_city, "date": date, "seats": 42}]}

def scheme_lookup(scheme_name):
    return {"scheme": scheme_name, "summary": "Mock summary of the scheme.", "eligibility": "Indian citizens"}

def set_reminder(text, time):
    return {"status": "set", "text": text, "time": time}

RATES = {("USD", "INR"): 83.0, ("INR", "USD"): 0.012, ("EUR", "INR"): 90.0, ("INR", "EUR"): 0.011}
def currency_convert(amount, from_cur, to_cur):
    rate = RATES.get((from_cur.upper(), to_cur.upper()), 1.0)
    return {"result": round(amount * rate, 2), "currency": to_cur.upper()}

def search_news(topic):
    return {"topic": topic, "headlines": [f"Latest update on {topic}", f"Experts discuss {topic}"]}

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsafe expression")

def calculator(expression):
    return {"result": _eval(ast.parse(expression, mode="eval").body)}

REGISTRY = {f.__name__: f for f in
            [get_weather, send_upi, check_train, scheme_lookup, set_reminder,
             currency_convert, search_news, calculator]}

def execute(tool, args):
    """Run a tool safely. Returns result dict or {'error': ...}."""
    if tool not in REGISTRY:
        return {"error": f"unknown tool {tool}"}
    missing = [r for r in SCHEMA_BY_NAME[tool]["required"] if r not in args]
    if missing:
        return {"error": f"missing arguments: {missing}"}
    try:
        return REGISTRY[tool](**args)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print(execute("get_weather", {"city": "Udaipur"}))
    print(execute("calculator", {"expression": "12*5+3"}))
    print(execute("send_upi", {"to": "Ravi"}))
