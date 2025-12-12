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

# ---- Load model ----
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "transformer_sinusoidal_6.8M.pt"

device = get_device()
checkpoint = load_checkpoint(MODEL_PATH, device)
model, tokenizer = load_model_tokenizer_from_transformer_checkpoint(checkpoint, device)


# ---- FastAPI ----
app = FastAPI()

UI_DIR = Path(__file__).parent  # <-- src/ui/


# Serve static files (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


# Serve index.html at "/"
@app.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")


# Request body schema
class GenerateRequest(BaseModel):
    synopsis: str
    rating: float
    liked: bool


# Generation endpoint
@app.post("/generate")
def generate(req: GenerateRequest):
    prompt = req.synopsis

    review = generate_single(model, tokenizer, device, prompt=prompt, length=200)
    return {"review": review}


# Run with uv
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.ui.server:app", host="0.0.0.0", port=8000, reload=True)
