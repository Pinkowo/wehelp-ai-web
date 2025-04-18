# app/model.py
import random

LABELS = [
    "Baseball",
    "Boy-Girl",
    "C_Chat",
    "HatePolitics",
    "Lifesmoney",
    "Military",
    "PC_Shopping",
    "Stock",
    "Tech_Job",
]


def predict(text: str) -> str:
    # TODO: 換成真正模型
    return random.choice(LABELS)
