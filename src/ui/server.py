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
parser.add_argument(
    "-l",
    "--level",
    type=int,
    choices=[1, 2],
    required=True,
    help="LeveL 1 = continue review, 2 = write review to given synopsis",
)
args, _ = parser.parse_known_args()
LEVEL = args.level

MODEL_PATH = Path(args.model)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / "models" / MODEL_PATH

device = get_device()
checkpoint = load_checkpoint(MODEL_PATH, device)
model, tokenizer = load_model_tokenizer_from_transformer_checkpoint(checkpoint, device)

app = FastAPI()
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


def extract_review(text: str) -> str:
    if "<REV>" in text:
        return text.split("<REV>", 1)[1].strip()
    return text.strip()


@app.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")


class GenerateRequest(BaseModel):
    synopsis: str
    rating: float
    liked: bool


@app.post("/generate")
def generate(req: GenerateRequest):
    match LEVEL:
        case 1:
            prompt = req.synopsis
        case 2:
            prompt = f"<SYN> {req.synopsis} <REV> "
        case _:
            raise ValueError("Invalid level")

    raw_output = generate_single(model, tokenizer, device, prompt=prompt, length=200)
    review = extract_review(raw_output)
    return {"review": review}


if __name__ == "__main__":
    uvicorn.run("src.ui.server:app", host="0.0.0.0", port=8000, reload=True)
