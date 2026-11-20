# BharatAgent
Small Hindi/Hinglish tool-calling agent: benchmark + SFT + GRPO.

## Run order
```
pip install -r requirements.txt
python tools/tools.py              # test fake tools
python scorer/score.py             # test scorer
python data/make_data.py           # make seed data -> data/train.jsonl, data/test.jsonl
python scorer/run_eval.py          # baseline pipeline (edit generate() to call a real model)
# on Kaggle/Colab GPU:
python training/sft.py
python training/grpo_train.py
```
