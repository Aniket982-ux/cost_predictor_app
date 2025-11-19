# app.py
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import torch
import numpy as np

# Import your embedding functions and inference pipeline
from text_embed import embed_long_text
from image_embed import get_embedding
from inference import load_embedding_model, load_lgbm_model, predict_price_from_embedding  # your imports

app = FastAPI(title="Multimodal Price Prediction API")
embedding_model = load_embedding_model()
lgbm_model = load_lgbm_model()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def async_embed_text(text: str):
    return embed_long_text(text)


async def async_embed_image(image_file: UploadFile):
    import tempfile
    import os
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(await image_file.read())
    temp.close()
    try:
        emb_arr = get_embedding(temp.name)       # numpy array
    finally:
        os.unlink(temp.name)
    emb_tensor = torch.tensor(emb_arr, dtype=torch.float32).unsqueeze(0)  # [1, dim]
    return emb_tensor

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "device": str(device)}

@app.post("/predict")
async def predict(text: str = Form(...), image: UploadFile = File(...)):
    # Run text and image embedding async in parallel
    text_task = asyncio.create_task(async_embed_text(text))
    image_task = asyncio.create_task(async_embed_image(image))
    text_emb = await text_task  # torch.Tensor shape [1, hidden_dim]
    image_emb = await image_task  # torch.Tensor shape [1, hidden_dim]

    # Convert embeddings to CPU numpy arrays and concatenate
    combined_emb = torch.cat([text_emb.to(device), image_emb.to(device)], dim=1).squeeze(0).cpu().numpy()

    # Pass combined embedding through your inference pipeline for refined embedding + price prediction
    predicted_price = predict_price_from_embedding(combined_emb, embedding_model, lgbm_model)

    return {"predicted_price": predicted_price}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

