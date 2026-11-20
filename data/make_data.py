"""
Make a starter dataset from templates (no API needed).
Run:  python data/make_data.py
Output: data/train.jsonl, data/test.jsonl
IMPORTANT: this is only a SEED set. Use a strong LLM to paraphrase/expand it
(more styles, typos, Rajasthani, multi-step) and hand-check your test set.
"""
import json, random
random.seed(42)

CITIES = ["Udaipur", "Jaipur", "Delhi", "Mumbai", "Kota", "Jodhpur", "Pune", "Chennai", "Kolkata", "Bhopal"]
NAMES = ["Ravi", "Priya", "Amit", "Sunita", "Rahul", "Neha", "Mohan", "Kavita"]
SCHEMES = ["PM Kisan", "Ayushman Bharat", "Ujjwala Yojana", "Mudra Loan", "Atal Pension Yojana"]
TOPICS = ["cricket", "election", "stock market", "ISRO", "weather alert"]
DATES = ["2026-12-05", "2026-12-10", "2027-01-02", "2026-12-25"]
DATE_WORDS = {"2026-12-05": "5 December", "2026-12-10": "10 December", "2027-01-02": "2 January", "2026-12-25": "25 December"}

def call(tool, **args):
    return {"calls": [{"tool": tool, "args": args}]}

def make():
    rows = []
    def add(q, gold, lang, kind):
        rows.append({"query": q, "gold": gold, "lang": lang, "kind": kind})

    for c in CITIES:
        g = call("get_weather", city=c)
        add(f"{c} ka mausam kaisa hai?", g, "hinglish", "single")
        add(f"{c} में आज मौसम कैसा है?", g, "hindi", "single")
        add(f"What's the weather in {c}?", g, "english", "single")

    for n in NAMES:
        amt = random.choice([100, 250, 500, 1000, 2000])
        g = call("send_upi", to=n, amount=amt)
        add(f"{n} ko {amt} rupaye bhej do", g, "hinglish", "single")
        add(f"{n} को {amt} रुपये भेजो", g, "hindi", "single")
        add(f"Send {amt} rupees to {n} on UPI", g, "english", "single")
        # missing info -> should ask
        add(f"{n} ko paise bhej do", {"ask": "Kitne rupaye bhejne hain?"}, "hinglish", "missing")
        add("Paise bhej do", {"ask": "Kisko aur kitne rupaye bhejne hain?"}, "hinglish", "missing")

    for a in CITIES[:5]:
        for b in CITIES[5:]:
            d = random.choice(DATES)
            g = call("check_train", from_city=a, to_city=b, date=d)
            add(f"{d} ko {a} se {b} ki train batao", g, "hinglish", "single")
            add(f"Trains from {a} to {b} on {DATE_WORDS[d]}", g, "english", "single")
            add(f"{a} se {b} ki train chahiye", {"ask": "Kis tarikh ke liye?"}, "hinglish", "missing")

    for s in SCHEMES:
        g = call("scheme_lookup", scheme_name=s)
        add(f"{s} ke baare mein batao", g, "hinglish", "single")
        add(f"{s} योजना की जानकारी दो", g, "hindi", "single")
        add(f"Tell me about {s}", g, "english", "single")

    for t in TOPICS:
        g = call("search_news", topic=t)
        add(f"{t} ki latest khabar kya hai?", g, "hinglish", "single")
        add(f"Latest news on {t}", g, "english", "single")

    for x, y in [(12, 5), (45, 3), (100, 7), (250, 4)]:
        add(f"{x} guna {y} kitna hota hai?", call("calculator", expression=f"{x}*{y}"), "hinglish", "single")
        add(f"What is {x} plus {y}?", call("calculator", expression=f"{x}+{y}"), "english", "single")

    for amt in [10, 50, 100, 500]:
        add(f"{amt} dollar ko rupaye mein badlo", call("currency_convert", amount=amt, from_cur="USD", to_cur="INR"), "hinglish", "single")
        add(f"Convert {amt} USD to INR", call("currency_convert", amount=amt, from_cur="USD", to_cur="INR"), "english", "single")

    for text, tm in [("dawai leni hai", "08:00"), ("meeting hai", "10:30"), ("mummy ko call karna", "18:00")]:
        add(f"Mujhe {tm} baje yaad dilana: {text}", call("set_reminder", text=text, time=tm), "hinglish", "single")

    # multi-step
    for c in CITIES[:5]:
        g = {"calls": [{"tool": "get_weather", "args": {"city": c}},
                       {"tool": "set_reminder", "args": {"text": "umbrella lena", "time": "08:00"}}]}
        add(f"{c} ka mausam check karo aur 08:00 baje umbrella lene ka reminder lagao", g, "hinglish", "multi")

    # no tool needed
    for q in ["Namaste, kaise ho?", "Tum kaun ho?", "Thank you!", "Dhanyavaad", "Hello"]:
        add(q, {"answer": "Namaste! Main aapki kaise madad kar sakta hoon?"}, "hinglish", "no_tool")

    return rows

if __name__ == "__main__":
    rows = make()
    random.shuffle(rows)
    n_test = max(30, len(rows) // 6)
    test, train = rows[:n_test], rows[n_test:]
    for name, data in [("train", train), ("test", test)]:
        with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} test={len(test)}")
