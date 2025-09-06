# --------------------------------------------
# DINO → (patch tokens, CLS)  +  LSTM+Bahdanau
# --------------------------------------------

# ========== Imports ==========
import os
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageFile
import torchvision.transforms as T
from tqdm import tqdm
from itertools import islice

try:
    from transformers import AutoModel, AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ========== Config ==========
CSV_PATH = r"C:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\CheXpertPlus\df_chexpert_plus_240401.csv"
IMG_ROOT = r"C:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\CheXpertPlus\PNG"
TEXT_COL = "section_impression"
PATH_COL = "path_to_image"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========== Data Loading ==========
class CheXpertDataset(Dataset):
    def __init__(self, img_root, csv, transform=None, text_col="section_impression", enforce_exists=True):
        self.img_root = os.path.abspath(img_root)
        self.df = csv.reset_index(drop=True)
        self.transform = transform
        self.text_col = text_col
        self.enforce_exists = enforce_exists

    def __len__(self):
        return len(self.df)

    def _full_png_path(self, rel):
        p = rel.replace("\\", "/")
        return os.path.join(self.img_root, p).replace(".jpg", ".png")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = row[PATH_COL]
        full_path = self._full_png_path(rel)
        if self.enforce_exists and not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        with Image.open(full_path) as im:
            im = im.convert("RGB")
            image = self.transform(im) if self.transform else im
        findings = row[self.text_col]
        findings = "" if pd.isna(findings) else str(findings)
        return {"image": image, "label": findings, "path": full_path}

def dino_image_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class SimpleWordTokenizer:
    def __init__(self, texts, bos="<bos>", eos="<eos>", pad="<pad>", unk="<unk>"):
        from collections import Counter
        self.bos, self.eos, self.pad, self.unk = bos, eos, pad, unk
        cnt = Counter()
        for t in texts:
            if isinstance(t, str):
                cnt.update(t.strip().lower().split())
        self.itos = [pad, bos, eos, unk] + sorted([w for w in cnt])
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_token_id = self.stoi[pad]
        self.bos_token_id = self.stoi[bos]
        self.eos_token_id = self.stoi[eos]
        self.unk_token_id = self.stoi[unk]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text, add_special_tokens=True):
        toks = [self.stoi.get(w, self.unk_token_id) for w in str(text).strip().lower().split()]
        return [self.bos_token_id] + toks + [self.eos_token_id] if add_special_tokens else toks

    def decode(self, ids):
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return " ".join(self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids if i not in specials)

def build_tokenizer_from_labels(labels):
    if HF_AVAILABLE:
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": "[PAD]"})
        class Wrap:
            def __init__(self, t):
                self.t = t
                self.pad_token_id = t.pad_token_id
                self.bos_token_id = t.cls_token_id
                self.eos_token_id = t.sep_token_id
            @property
            def vocab_size(self): return self.t.vocab_size
            def encode(self, text, add_special_tokens=True):
                ids = self.t.encode(str(text), add_special_tokens=False)
                return [self.bos_token_id] + ids + [self.eos_token_id] if add_special_tokens else ids
            def decode(self, ids): return self.t.decode(ids, skip_special_tokens=True)
        return Wrap(tok)
    else:
        return SimpleWordTokenizer(labels)

class CaptionCollate:
    def __init__(self, tokenizer, pad_id):
        self.tokenizer = tokenizer
        self.pad_id = pad_id

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        ids = [torch.tensor(self.tokenizer.encode(b["label"], add_special_tokens=True), dtype=torch.long) for b in batch]
        targets = pad_sequence(ids, batch_first=True, padding_value=self.pad_id)
        paths = [b["path"] for b in batch]
        labels = [b["label"] for b in batch]
        return images, targets, paths, labels

def build_valid_df(csv_path, img_root):
    df = pd.read_csv(csv_path)
    keep_idx = []
    for i, rel in enumerate(df[PATH_COL].tolist()):
        p = os.path.join(img_root, rel.replace("\\", "/")).replace(".jpg", ".png")
        if os.path.exists(p):
            keep_idx.append(i)
    valid_df = df.iloc[keep_idx].reset_index(drop=True)
    print(f"[INFO] Kept {len(valid_df)}/{len(df)} rows with existing PNGs under {img_root}")
    return valid_df

# ========== Model ==========
class DINOEncoder(nn.Module):
    def __init__(self, model_id="facebook/dinov3-vits16-pretrain-lvd1689m", freeze=True):
        super().__init__()
        if not HF_AVAILABLE:
            raise RuntimeError("Hugging Face transformers not available in this environment.")
        self.model = AutoModel.from_pretrained(model_id)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        tokens = out.last_hidden_state
        cls = tokens[:, 0, :]
        patches = tokens[:, 5:, :]
        return cls, patches

class BahdanauAttention(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.W_h = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.v = nn.Linear(d_h, 1, bias=False)

    def forward(self, h_t, keys):
        B, N, d_h = keys.size()
        energy = torch.tanh(self.W_h(h_t).unsqueeze(1) + self.W_k(keys))
        scores = self.v(energy).squeeze(-1)
        alpha = torch.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), keys).squeeze(1)
        return context, alpha

class LSTMAttnDecoder(nn.Module):
    def __init__(self, vocab_size, d_img, d_h=512, pad_id=0, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_h = d_h
        self.pad_id = pad_id
        self.kv_proj = nn.Linear(d_img, d_h)
        self.init_h = nn.Linear(d_img, d_h)
        self.init_c = nn.Linear(d_img, d_h)
        self.emb = nn.Embedding(vocab_size, d_h, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_h)
        self.lstm = nn.LSTMCell(d_h * 2, d_h)
        self.attn = BahdanauAttention(d_h)
        self.out = nn.Linear(d_h, vocab_size)

    def forward(self, patches, cls, tgt_ids):
        B, T = tgt_ids.size()
        device = tgt_ids.device
        keys = self.kv_proj(patches)
        h = self.init_h(cls)
        c = self.init_c(cls)
        logits = []
        x = self.emb(tgt_ids[:, 0])
        for t in range(T):
            ctx, _ = self.attn(h, keys)
            lstm_in = torch.cat([x, ctx], dim=-1)
            h, c = self.lstm(lstm_in, (h, c))
            h = self.ln(self.dropout(h))
            step_logits = self.out(h)
            logits.append(step_logits.unsqueeze(1))
            if t + 1 < T:
                x = self.emb(tgt_ids[:, t + 1])
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def generate(self, patches, cls, bos_id, eos_id, max_new_tokens=30, top_p=0.9, temperature=1.0, greedy=False):
        B = cls.size(0)
        device = cls.device
        keys = self.kv_proj(patches)
        h = self.init_h(cls)
        c = self.init_c(cls)
        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        x = self.emb(seq[:, -1])
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            ctx, _ = self.attn(h, keys)
            h, c = self.lstm(torch.cat([x, ctx], dim=-1), (h, c))
            h = self.ln(h)
            logits = self.out(h) / max(1e-6, temperature)
            logits[:, bos_id] = -1e9
            min_len = 2
            if seq.size(1) < min_len:
                logits[:, eos_id] = -1e9
            probs = torch.softmax(logits, dim=-1)
            if greedy:
                next_tokens = probs.argmax(dim=-1)
            else:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumprobs > top_p
                mask[:, 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_pos = torch.multinomial(sorted_probs, num_samples=1).squeeze(1)
                next_tokens = sorted_idx.gather(dim=1, index=next_pos.unsqueeze(1)).squeeze(1)
            seq = torch.cat([seq, next_tokens.unsqueeze(1)], dim=1)
            x = self.emb(next_tokens)
            finished = finished | (next_tokens == eos_id)
            if finished.all():
                break
        return seq

class DinoLSTMAttnCaptioner(nn.Module):
    def __init__(self, vocab_size, d_img, d_h=512, pad_id=0, dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m", freeze_dino=True):
        super().__init__()
        self.encoder = DINOEncoder(dino_model_id, freeze=freeze_dino)
        self.decoder = LSTMAttnDecoder(vocab_size=vocab_size, d_img=d_img, d_h=d_h, pad_id=pad_id)

    def forward(self, pixel_values, tgt_ids):
        cls, patches = self.encoder(pixel_values)
        logits = self.decoder(patches, cls, tgt_ids)
        return logits

    @torch.no_grad()
    def encode(self, pixel_values):
        return self.encoder(pixel_values)

    @torch.no_grad()
    def generate(self, pixel_values, bos_id, eos_id, max_new_tokens=30, top_p=0.9, temperature=1.0, greedy=False):
        cls, patches = self.encoder(pixel_values)
        return self.decoder.generate(
            patches=patches,
            cls=cls,
            bos_id=bos_id,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            greedy=greedy
        )

# ========== Training Utils ==========
def sequence_ce_loss(logits, targets, pad_id):
    B, T, V = logits.size()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    logits_next = logits[:, :-1, :].contiguous().view(-1, V)
    targets_next = targets[:, 1:].contiguous().view(-1)
    return loss_fn(logits_next, targets_next)

@torch.no_grad()
def batch_perplexity(logits, targets, pad_id):
    loss = sequence_ce_loss(logits, targets, pad_id)
    return float(math.exp(min(loss.item(), 20.0)))

def train_one_epoch(model, loader, optimizer, device, pad_id, num_batches, grad_clip=1.0):
    model.train()
    total_loss, total_pp, steps = 0.0, 0.0, 0
    for pixel_values, tgt_ids, *_ in tqdm(loader, desc="Training", total=num_batches):
        pixel_values = pixel_values.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values, tgt_ids)
        loss = sequence_ce_loss(logits, tgt_ids, pad_id)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        with torch.no_grad():
            ppl = batch_perplexity(logits, tgt_ids, pad_id)
        total_loss += loss.item()
        total_pp += ppl
        steps += 1
        # Free batch memory
        del pixel_values, tgt_ids, logits, loss, ppl
        torch.cuda.empty_cache()

    return {"loss": total_loss / steps, "ppl": total_pp / steps}

@torch.no_grad()
def evaluate(model, loader, device, pad_id, num_batches):
    model.eval()
    total_loss, total_pp, steps = 0.0, 0.0, 0
    for pixel_values, tgt_ids, *_ in tqdm(loader, desc="Evaluating", total=num_batches):
        pixel_values = pixel_values.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)
        logits = model(pixel_values, tgt_ids)
        loss = sequence_ce_loss(logits, tgt_ids, pad_id)
        ppl = batch_perplexity(logits, tgt_ids, pad_id)
        total_loss += loss.item()
        total_pp += ppl
        steps += 1
        # Free batch memory
        del pixel_values, tgt_ids, logits, loss, ppl
        torch.cuda.empty_cache()

    return {"val_loss": total_loss / steps, "val_ppl": total_pp / steps}

# ========== Main ==========
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#     valid_df = build_valid_df(CSV_PATH, IMG_ROOT)
#     if valid_df.empty:
#         print("[WARN] No valid rows found; check paths and PNG conversion.")
#         return
#     labels_as_str = valid_df[TEXT_COL].astype(str).tolist()
#     tokenizer = build_tokenizer_from_labels(labels_as_str)
#     pad_id = getattr(tokenizer, "pad_token_id", 0)
#     bos_id = getattr(tokenizer, "bos_token_id", 1)
#     eos_id = getattr(tokenizer, "eos_token_id", 2)
#     tf = dino_image_transform(img_size=1024)
#     ds = CheXpertDataset(img_root=IMG_ROOT, csv=valid_df, transform=tf, text_col=TEXT_COL)
#     collate_fn = CaptionCollate(tokenizer, pad_id)
#     is_windows = os.name == "nt"
#     num_workers = 0 if is_windows else 2
#     persistent_workers = False if num_workers == 0 else True
#     loader = DataLoader(
#         ds,
#         batch_size=8,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         persistent_workers=persistent_workers,
#         collate_fn=collate_fn
#     )
#     train_ds = torch.utils.data.Subset(ds, range(0, int(len(ds)*.8)))
#     valid_ds = torch.utils.data.Subset(ds, range(int(len(ds)*.8), len(ds)))
#     train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
#     valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
#     D_IMG = 384
#     model = DinoLSTMAttnCaptioner(
#         vocab_size=tokenizer.vocab_size,
#         d_img=D_IMG,
#         d_h=512,
#         pad_id=pad_id,
#         dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
#         freeze_dino=True,
#     ).to(device)
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2
#     )
#     for epoch in range(20):
#         slice_train_loader = islice(train_loader, 10)
#         slice_valid_loader = islice(valid_loader, 10)
#         train_stats = train_one_epoch(model, slice_train_loader, optimizer, device, pad_id, num_batches=10, grad_clip=1.0)
#         val_stats = evaluate(model, slice_valid_loader, device, pad_id, num_batches=10)
#         print(f"Epoch {epoch + 1}: Train Loss={train_stats['loss']:.4f}, PPL={train_stats['ppl']:.2f} | "
#               f"Val Loss={val_stats['val_loss']:.4f}, Val PPL={val_stats['val_ppl']:.2f}")
#     test_loader_sliced = iter(valid_loader)
#     with torch.no_grad():
#         for batch in test_loader_sliced:
#             pixel_values, ids_loader, paths, raw_labels = batch
#             pixel_values = pixel_values.to(device)
#             gen_ids = model.generate(
#                 pixel_values=pixel_values,
#                 bos_id=bos_id, eos_id=eos_id,
#                 max_new_tokens=50, top_p=0.9, temperature=0.9, greedy=True
#             )
#             print("Predictions test:")
#             for i in range(gen_ids.size(0)):
#                 print(f"\nTEST GEN {i+1}:", tokenizer.decode(gen_ids[i].tolist()))
#                 print(f"TEST TARGET {i+1}:", tokenizer.decode(ids_loader[i].tolist()))
#             # Free batch memory
#             del pixel_values, ids_loader, paths, raw_labels, gen_ids
#             torch.cuda.empty_cache()
#             break

# if __name__ == "__main__":
#     main()