# text_embed.py

import logging
from transformers import AutoTokenizer, AutoModel
import torch

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'sentence-transformers/all-mpnet-base-v2'  # Or your preferred model

logging.info(f"Loading tokenizer for model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

logging.info(f"Loading pretrained model: {model_name}")
model = AutoModel.from_pretrained(model_name)
logging.info("Model loaded successfully, moving to device...")
model = model.to(device)
logging.info(f"Model moved to device: {device}")
model.eval()  # Set model to eval mode for inference

max_length = 512
stride = 128

def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on token embeddings based on attention mask.
    """
    token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask  # [batch_size, hidden_dim]

def embed_long_text(text: str) -> torch.Tensor:
    """
    Encodes long text into a single embedding vector by chunking and mean pooling.

    Args:
      text (str): Input text to embed.
      
    Returns:
      torch.Tensor: [1, hidden_dim] embedding tensor on CPU.
    """
    logging.info("Encoding input text for embedding...")
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )
    logging.info(f"Number of chunks generated: {len(encoded['input_ids'])}")
    chunk_embeddings = []
    for i in range(len(encoded['input_ids'])):
        inputs = {k: v[i].unsqueeze(0).to(device) for k, v in encoded.items() if k != 'overflow_to_sample_mapping'}
        with torch.no_grad():
            outputs = model(**inputs)
            emb = mean_pooling(outputs, inputs['attention_mask'])  # [1, hidden_dim]
        chunk_embeddings.append(emb.cpu())
    logging.info("Chunk embeddings computed, combining...")
    item_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
    logging.info("Combined embedding ready")
    return item_embedding  # [1, hidden_dim]


