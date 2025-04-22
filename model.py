from pathlib import Path
import json
import torch
from gensim.models.doc2vec import Doc2Vec

from model_arch import Classifier  # 確保 model_arch.py 內有相同定義

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


def predict(text: str) -> dict:
    """回傳所有分類的機率分布（由高到低排序）"""
    vec = _vectorize(text)
    with torch.no_grad():
        logits = _classifier(torch.tensor(vec, dtype=torch.float).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    label_probs = [
        {"label": label, "score": probs[idx]} for idx, label in IDX2LABEL.items()
    ]
    label_probs.sort(key=lambda x: x["score"], reverse=True)
    return label_probs
