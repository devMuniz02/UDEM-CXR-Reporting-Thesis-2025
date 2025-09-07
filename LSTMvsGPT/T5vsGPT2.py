# --- Imports ---
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Import CheXpertDataset and DINOEncoder ---
from GPT import CheXpertDataset, dino_image_transform, build_valid_df, TEXT_COL, IMG_ROOT, CSV_PATH

# --- GPT2 Decoder ---
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class DINOv3Encoder(nn.Module):
    def __init__(self, model_id="facebook/dinov3-vits16-pretrain-lvd1689m", out_dim=384, tokens=8, freeze=True):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.model.config.hidden_size, out_dim)
        self.tokens = tokens

    @torch.no_grad()
    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        tokens = out.last_hidden_state  # [B, N, D]
        # Use CLS + first N patch tokens
        prefix = torch.cat([tokens[:, :1, :], tokens[:, 5:5+self.tokens, :]], dim=1)
        prefix = self.proj(prefix)
        return prefix  # [B, tokens+1, out_dim]

class VisualPrefixGPT2(nn.Module):
    def __init__(self, gpt2_name="gpt2", vis_dim=384, num_prefix_tokens=8):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_name)
        self.num_prefix_tokens = num_prefix_tokens
        self.vis_to_prefix = nn.Linear(vis_dim, self.model.config.n_embd)

    def encode_text(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    def forward(self, visual_tokens, input_ids, attention_mask=None, labels=None):
        B, Tvis, Dv = visual_tokens.shape
        prefix = self.vis_to_prefix(visual_tokens)
        inputs_embeds = self.model.transformer.wte(input_ids)
        full_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        vis_mask = torch.ones((B, Tvis), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)
        # Fix: pad labels with -100 for prefix tokens
        if labels is not None:
            pad_labels = torch.full((B, Tvis), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([pad_labels, labels], dim=1)
        else:
            full_labels = None
        return self.model(inputs_embeds=full_embeds, attention_mask=full_mask, labels=full_labels)

# --- T5 Decoder ---
from transformers import T5Tokenizer, T5ForConditionalGeneration

class DinoT5(nn.Module):
    def __init__(self, t5_name="t5-base", vis_dim=384):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(t5_name)
        self.model = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.vis_proj = nn.Linear(vis_dim, self.model.config.d_model)

    def encode_visual(self, vtokens):
        return self.vis_proj(vtokens)

    def forward(self, vtokens, target_texts):
        enc_vis = self.encode_visual(vtokens)
        out = self.model(
            encoder_outputs=(enc_vis, ),
            labels=self.tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).input_ids,
            return_dict=True,
        )
        return out

# --- Collate ---
class CaptionCollate:
    def __init__(self, tokenizer, pad_id):
        self.tokenizer = tokenizer
        self.pad_id = pad_id

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        texts = [b["label"] for b in batch]
        return images, texts

# --- Main ---
def main(option="A", device="cuda"):
    valid_df = build_valid_df(CSV_PATH, IMG_ROOT)
    labels_as_str = valid_df[TEXT_COL].astype(str).tolist()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if option == "A" else T5Tokenizer.from_pretrained("t5-base")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    tf = dino_image_transform(img_size=1024)
    ds = CheXpertDataset(img_root=IMG_ROOT, csv=valid_df, transform=tf, text_col=TEXT_COL)
    collate_fn = CaptionCollate(tokenizer, pad_id)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    if option == "A":
        dino = DINOv3Encoder(out_dim=384, tokens=8).to(device)
        dec = VisualPrefixGPT2(gpt2_name="gpt2", vis_dim=384, num_prefix_tokens=9).to(device)
    else:
        dino = DINOv3Encoder(out_dim=384, tokens=8).to(device)
        dec = DinoT5(t5_name="t5-base", vis_dim=384).to(device)

    optim = torch.optim.AdamW(dec.parameters(), lr=2e-5)

    from tqdm import tqdm

    # Training loop
    for epoch in range(2):
        num_batches = 0
        for images, texts in tqdm(loader, desc=f"Epoch {epoch}"):
            num_batches += 1
            images = images.to(device)
            vis_tokens = dino(images)
            if option == "A":
                tok = dec.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                input_ids = tok.input_ids.to(device)
                labels = input_ids.clone()
                out = dec(vis_tokens, input_ids=input_ids, labels=labels)
                loss = out.loss
            else:
                out = dec(vis_tokens, texts)
                loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            if num_batches == 1000:
                break
        print(f"Epoch {epoch} loss: {loss.item():.4f}")

    with torch.no_grad():
        image, target_text = next(iter(loader))
        image = image.to(device)
        vis_tokens = dino(image)
        if option == "A":
            tok = dec.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)
            input_ids = tok.input_ids.to(device)
            labels = input_ids.clone()
            out = dec(vis_tokens, input_ids=input_ids, labels=labels)
            loss = out.loss
            print(f"Sample loss (GPT2): {loss.item():.4f}")

            # Generate text from visual prefix
            prefix_embeds = dec.vis_to_prefix(vis_tokens)
            text_embeds = dec.model.transformer.wte(input_ids)
            full_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
            # Attention mask for prefix + text
            full_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=full_embeds.device)
            gen_ids = dec.model.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                max_new_tokens=512,  # Number of tokens to generate
                pad_token_id=dec.tokenizer.eos_token_id,
                do_sample=False
            )
            # Decode and print for each sample in batch
            for i in range(gen_ids.shape[0]):
                generated_text = dec.tokenizer.decode(gen_ids[i], skip_special_tokens=True)
                target_text_decoded = dec.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                print(f"Length of generated ids: {len(gen_ids[i])}")
                print(f"Length of target ids: {len(input_ids[i])}")
                print(f"Generated [{i}]:", generated_text)
                print(f"Target   [{i}]:", target_text_decoded)
        else:
            out = dec(vis_tokens, target_text)
            loss = out.loss
            print(f"Sample loss (T5): {loss.item():.4f}")
            gen_ids = dec.model.generate(inputs_embeds=dec.encode_visual(vis_tokens), max_new_tokens=64)
            for i in range(gen_ids.shape[0]):
                generated_text = dec.tokenizer.decode(gen_ids[i], skip_special_tokens=True)
                print(f"Generated [{i}]:", generated_text)
                print(f"Target   [{i}]:", target_text[i])

if __name__ == "__main__":
    main(option="A", device="cuda" if torch.cuda.is_available() else "cpu")
