import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.generation.generate_transformer import (
    generate_single,
    load_checkpoint,
    load_model_tokenizer_from_transformer_checkpoint,
)
from src.utils.device import get_device

BASE_DIR = Path(__file__).parent.parent.parent
UI_DIR = Path(__file__).parent  # <-- src/ui/
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args, _ = parser.parse_known_args()

MODEL_PATH = Path(args.model)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / "models" / MODEL_PATH

device = get_device()
checkpoint = load_checkpoint(MODEL_PATH, device)
model, tokenizer = load_model_tokenizer_from_transformer_checkpoint(checkpoint, device)

app = FastAPI()
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")


class GenerateRequest(BaseModel):
    synopsis: str
    rating: float
    liked: bool


@app.post("/generate")
def generate(req: GenerateRequest):
    review = generate_single(model, tokenizer, device, prompt=req.synopsis, length=200)
    return {"review": review}


if __name__ == "__main__":
    uvicorn.run("src.ui.server:app", host="0.0.0.0", port=8000, reload=True)
