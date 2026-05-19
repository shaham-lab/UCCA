"""
Encode the COCO 2017 val subset (2K images) into two embedding tensors:
  - encoded1.pt  : image features via DINOv2-ViT-S/14   (N x 384)
  - encoded2.pt  : text  features via all-MiniLM-L6-v2   (N x 384)
  - labels.pt    : object-category label per image       (N,)

Dataset: riddhimanrana/coco-fastvlm-2k-val2017
 - 'image' field is a PIL image (decoded automatically by HF datasets)
 - 'conversations' contains human/gpt turns; gpt value = structured caption

Run from the repo root:
  python datasets/coco/download.py
"""

import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

OUT_DIR = Path("datasets/coco")


def get_text_embedding(texts, tokenizer, model, device, batch_size=128):
    """Mean-pool token embeddings to get a sentence vector."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**encoded)
        token_embs = out[0]
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)


def extract_label(human_val: str) -> str:
    """Extract the first detected object class from the human prompt."""
    m = re.search(
        r"objects detected[^\n]*\n[•\-\*]?\s*([A-Za-z ]+?)(?:\s*\(|\n|$)",
        human_val, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    # Fallback: look for bullet list items
    lines = human_val.splitlines()
    for line in lines:
        line = line.strip().lstrip("•*- ")
        if line and not line.startswith("<") and not line.startswith("The"):
            # First short non-sentence line is likely a class name
            tok = line.split("(")[0].strip()
            if tok and len(tok.split()) <= 3:
                return tok
    return "unknown"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load HF dataset -----------------------------------------------
    print("Loading riddhimanrana/coco-fastvlm-2k-val2017 ...")
    raw = load_dataset("riddhimanrana/coco-fastvlm-2k-val2017")
    split_name = list(raw.keys())[0]
    ds = raw[split_name]
    print(f"Split='{split_name}', N={len(ds)}")
    print(f"Features: {ds.features}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Image model: DINOv2-ViT-S/14 ---------------------------------
    print("Loading DINOv2 (dinov2_vits14)...")
    v_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    v_model.eval()

    v_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ---- Text model: all-MiniLM-L6-v2 ---------------------------------
    from transformers import AutoTokenizer, AutoModel
    print("Loading text model (all-MiniLM-L6-v2)...")
    t_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    t_tokenizer = AutoTokenizer.from_pretrained(t_model_name)
    t_model = AutoModel.from_pretrained(t_model_name).to(device)
    t_model.eval()

    # ---- Encode --------------------------------------------------------
    img_features = []
    txt_features = []
    label_list   = []

    batch_size = 32
    n = len(ds)
    print(f"Processing {n} samples in batches of {batch_size}...")

    for i in tqdm(range(0, n, batch_size)):
        batch = ds[i : i + batch_size]

        # ---- Images (already PIL) ----------------------------------------
        imgs_pil = [img.convert("RGB") for img in batch["image"]]
        imgs_t   = torch.stack([v_transform(img) for img in imgs_pil]).to(device)
        with torch.no_grad():
            img_emb = v_model(imgs_t).cpu()
        img_features.append(img_emb)

        # ---- Text: GPT structured caption --------------------------------
        captions = []
        for convs in batch["conversations"]:
            gpt_text = ""
            for turn in convs:
                if isinstance(turn, dict) and turn.get("from") == "gpt":
                    gpt_text = turn.get("value", "")
                    break
            captions.append(gpt_text)

        txt_emb = get_text_embedding(captions, t_tokenizer, t_model, device)
        txt_features.append(txt_emb)

        # ---- Label: first detected object from human prompt --------------
        for convs in batch["conversations"]:
            lbl = "unknown"
            for turn in convs:
                if isinstance(turn, dict) and turn.get("from") == "human":
                    lbl = extract_label(turn.get("value", ""))
                    break
            label_list.append(lbl)

    img_features = torch.cat(img_features, dim=0)
    txt_features = torch.cat(txt_features, dim=0)

    # Map string labels → integer indices
    unique_cats = sorted(set(label_list))
    cat2idx     = {c: i for i, c in enumerate(unique_cats)}
    labels      = torch.tensor([cat2idx[l] for l in label_list], dtype=torch.long)

    # ---- Save ----------------------------------------------------------
    torch.save(img_features, OUT_DIR / "encoded1.pt")
    torch.save(txt_features, OUT_DIR / "encoded2.pt")
    torch.save(labels,       OUT_DIR / "labels.pt")

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  encoded1.pt (images) : {img_features.shape}")
    print(f"  encoded2.pt (text)   : {txt_features.shape}")
    print(f"  labels.pt            : {labels.shape}  ({len(unique_cats)} unique classes)")


if __name__ == "__main__":
    main()
