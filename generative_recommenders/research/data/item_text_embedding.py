import os
import torch
from tqdm import tqdm
import tiktoken
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel


def generate_text_embeddings(texts, model_name, device="cuda"):
    """
    Generate embeddings for input texts using either 'blair', 'llama', or 'chatgpt'
    """
    if model_name.lower() == "blair":
        model_id = 'hyp1231/blair-roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        return _transformers_batch_encode(texts, tokenizer, model, device)
    elif model_name.lower() == "openai":
        return _openai_embed(texts)
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")


def _transformers_batch_encode(texts, tokenizer, model, device):
    all_embeddings = []
    batch_size = 8

    for pr in tqdm(range(0, len(texts), batch_size), desc="Start embedding"):
        batch = texts[pr: pr + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif hasattr(outputs, "hidden_states"):
            embeddings = outputs.hidden_states[-1][:, 0, :]
        else:
            raise RuntimeError("Model output does not contain expected embedding fields.")

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # [num_texts, hidden_dim]


def _openai_embed(texts, model="text-embedding-3-large", max_tokens=8192, output_dim=768):
    enc = tiktoken.encoding_for_model(model)
    client = OpenAI()
    embeddings = []

    def tokenize(text):
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            print(f"[WARN] Truncating input with {len(tokens)} tokens (limit {max_tokens}): {text[:30]}...")
            tokens = tokens[:max_tokens]
        return enc.decode(tokens)

    # Step 1: Token-safe batching (based on token count after truncation)
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        safe_text = tokenize(text)
        n_tokens = len(enc.encode(safe_text))

        if current_tokens + n_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = [safe_text]
            current_tokens = n_tokens
        else:
            current_batch.append(safe_text)
            current_tokens += n_tokens

    if current_batch:
        batches.append(current_batch)

    # Step 2: Batch embedding with dimension control
    for batch in tqdm(batches, desc="Embedding with OpenAI"):
        try:
            response = client.embeddings.create(
                model=model,
                input=batch,
                dimensions=output_dim
            )
            batch_embeddings = [
                torch.tensor(item.embedding, dtype=torch.float32)
                for item in response.data
            ]
        except Exception as e:
            print(f"[ERROR] Failed batch: {e}")
            batch_embeddings = [torch.zeros(output_dim, dtype=torch.float32) for _ in batch]

        embeddings.extend(batch_embeddings)

    return torch.stack(embeddings)  # [num_texts, output_dim]



def embed_and_save_item_texts(sorted_text, output_path, model_name="blair"):
    """
    Generate and save item text embeddings with selected model as a .pt file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = generate_text_embeddings(sorted_text, model_name=model_name, device=device)

    # Add a zero-padding embedding at index 0
    pad0_embedding = torch.zeros((1, embeddings.shape[1]), dtype=embeddings.dtype)
    embeddings = torch.cat([pad0_embedding, embeddings], dim=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"[INFO] Item embeddings saved to {output_path} with shape {embeddings.shape}")
