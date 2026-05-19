import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os
from torchvision import transforms

def main():
    print("Loading dataset jxie/flickr8k...")
    ds = load_dataset("jxie/flickr8k")
    train_ds = ds['train']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Image features (DINOv2)
    print("Loading DINOv2 model...")
    v_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    v_model.eval()
    
    v_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 2. Text features (SentenceTransformer or CLIP)
    # Using a simple one from transformers to avoid extra dependencies if possible
    from transformers import AutoTokenizer, AutoModel
    print("Loading text model (SentenceTransformer style)...")
    t_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    t_tokenizer = AutoTokenizer.from_pretrained(t_model_name)
    t_model = AutoModel.from_pretrained(t_model_name).to(device)
    t_model.eval()
    
    def get_text_embedding(texts):
        encoded_input = t_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = t_model(**encoded_input)
            # Mean pooling
            token_embeddings = model_output[0]
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

    img_features = []
    txt_features = []
    
    # Process in batches
    batch_size = 32
    print(f"Processing {len(train_ds)} samples...")
    for i in tqdm(range(0, len(train_ds), batch_size)):
        batch = train_ds[i:i+batch_size]
        
        # Images
        imgs = [v_transform(img.convert("RGB")) for img in batch['image']]
        imgs_t = torch.stack(imgs).to(device)
        with torch.no_grad():
            v_feat = v_model(imgs_t).cpu()
            img_features.append(v_feat)
            
        # Text (using caption_0 for simplicity, or we could average all 5)
        # Taking caption_0 for now
        captions = batch['caption_0']
        t_feat = get_text_embedding(captions).cpu()
        txt_features.append(t_feat)
        
    img_features = torch.cat(img_features, dim=0)
    txt_features = torch.cat(txt_features, dim=0)
    
    os.makedirs("datasets/flickr8", exist_ok=True)
    torch.save(img_features, "datasets/flickr8/encoded1.pt")
    torch.save(txt_features, "datasets/flickr8/encoded2.pt")
    
    print(f"Saved features to datasets/flickr8/")
    print(f"Images shape: {img_features.shape}")
    print(f"Text shape: {txt_features.shape}")

if __name__ == "__main__":
    main()
