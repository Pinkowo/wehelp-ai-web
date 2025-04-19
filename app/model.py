from pathlib import Path
import json
import torch
from gensim.models.doc2vec import Doc2Vec

from .model_arch import Classifier  # 確保 app/model_arch.py 內有相同定義

# === 路徑設定 ===============================================================
BASE_DIR = (
    Path(__file__).parent / "model_files"
)  # classifier.pt、label2idx.json、doc2vec_model.bin

# === 讀取標籤對照 ===========================================================
with (BASE_DIR / "label2idx.json").open("r", encoding="utf-8") as f:
    LABEL2IDX = json.load(f)
IDX2LABEL = {int(v): k for k, v in LABEL2IDX.items()}

# === 讀取 Doc2Vec ===========================================================
DOC2VEC_PATH = BASE_DIR / "doc2vec_model.bin"
_d2v = Doc2Vec.load(str(DOC2VEC_PATH))
VECTOR_SIZE = _d2v.vector_size

# === 還原分類器 ============================================================
HIDDEN_DIM = 64
_classifier = Classifier(
    input_dim=VECTOR_SIZE, hidden_dim=HIDDEN_DIM, num_classes=len(IDX2LABEL)
)
_classifier.load_state_dict(torch.load(BASE_DIR / "classifier.pt", map_location="cpu"))
_classifier.eval()

# ---------------------------------------------------------------------------


def _vectorize(text: str):
    """將輸入文字轉成 Doc2Vec 向量。"""
    tokens = text.split()
    return _d2v.infer_vector(tokens, epochs=50)


def predict(text: str) -> str:
    """回傳預測的 PTT 看板名稱。"""
    vec = _vectorize(text)
    with torch.no_grad():
        logits = _classifier(torch.tensor(vec, dtype=torch.float).unsqueeze(0))
        pred_idx = int(torch.argmax(logits, dim=1).item())
    return IDX2LABEL[pred_idx]
