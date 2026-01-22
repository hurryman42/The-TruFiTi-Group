import uvicorn
from pathlib import Path
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
from src.enums.types import SpecialTokensEnum

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, required=True)
# parser.add_argument(
#    "-l",
#    "--level",
#    type=int,
#    choices=[1, 2],
#    required=True,
#    help="LeveL 1 = continue review, 2 = write review to given synopsis",
# )
# args, _ = parser.parse_known_args()
# LEVEL = args.level

BASE_DIR = Path(__file__).parent.parent.parent
UI_DIR = Path(__file__).parent  # <-- src/ui/
MODELS_DIR = BASE_DIR / "models"
# MODEL_PATH = Path(args.model)
# if not MODEL_PATH.is_absolute():
#    MODEL_PATH = MODELS_DIR / MODEL_PATH

app = FastAPI()
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

device = get_device()
# checkpoint = load_checkpoint(MODEL_PATH, device)
# model, tokenizer = load_model_tokenizer_from_transformer_checkpoint(checkpoint, device)


def load_selected_model(model_name: str):
    model_path = MODELS_DIR / model_name
    checkpoint = load_checkpoint(model_path, device)
    return load_model_tokenizer_from_transformer_checkpoint(checkpoint, device)


def get_level_from_filename(filename: str) -> int:
    name = filename.upper()
    if "L3" in name:
        return 3
    if "L2" in name:
        return 2
    if "L1" in name:
        return 1
    print(f"[WARN] Could not detect LEVEL from model filename '{filename}'. Defaulting to LEVEL 1.")
    return 1


def extract_review(prompt, raw_output):
    raw = raw_output.lstrip()
    # return raw[len(prompt):].lstrip()

    prompt_decoded = prompt.replace(str(SpecialTokensEnum.SYN), "").replace(str(SpecialTokensEnum.REV), "").strip()

    # if raw begins with the decoded prompt, trim exactly that
    if raw.startswith(prompt_decoded):
        return raw[len(prompt_decoded) :].lstrip()

    # fallback: smallest overlap match
    if prompt_decoded in raw:
        idx = raw.index(prompt_decoded) + len(prompt_decoded)
        return raw[idx:].lstrip()

    # last fallback: don't destroy content
    return raw


@app.get("/models")
def list_models():
    return [f.name for f in MODELS_DIR.iterdir() if f.suffix == ".pt"]


@app.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")


class GenerateRequest(BaseModel):
    synopsis: str
    rating: float
    liked: bool
    model: str


@app.post("/generate")
def generate(req: GenerateRequest):
    model, tokenizer = load_selected_model(req.model)
    level = get_level_from_filename(req.model)
    print(f"LEVEL: {level}")
    match level:
        case 1:
            prompt = req.synopsis
        case 2:
            prompt = f"{SpecialTokensEnum.SYN} {req.synopsis} {SpecialTokensEnum.REV}"
        case _:
            raise ValueError("Invalid level")

    raw_output = generate_single(model, tokenizer, device, prompt=prompt, length=200)
    if level == 2:
        review = extract_review(prompt, raw_output)
    else:
        review = raw_output
    # review = review + "\n\n\n\n\n" + raw_output # DEBUG
    return {"review": review.strip()}


if __name__ == "__main__":
    uvicorn.run("src.ui.server:app", host="0.0.0.0", port=8000, reload=True)
