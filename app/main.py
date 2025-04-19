from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.model import predict
from fastapi import Form
from pathlib import Path
import csv

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
FEEDBACK_FILE = Path("user-labeled-titles.csv")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/model/prediction")
async def prediction_api(title: str):
    label = predict(title)
    return {"label": label}


@app.post("/api/model/feedback")
async def feedback(title: str = Form(...), label: str = Form(...)):
    if not FEEDBACK_FILE.exists():
        with FEEDBACK_FILE.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["board", "title"])
    with FEEDBACK_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, title])
    return {"message": "OK"}
