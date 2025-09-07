# --------------------------------------------
# DINO → (patch tokens, CLS)  +  GPT-like Transformer (visual prefix)
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

from torch.utils.data import Dataset
import json
from typing import List, Dict, Any
import torchvision.transforms as transforms

class PadChestGRDataset(Dataset):
    """
    Minimal, fast dataset for PadChest-GR report generation.
    - Precomputes image paths.
    - Pre-tokenizes reports once with BioBERT (or any HF tokenizer).
    - No logging, no prints.
    """
    def __init__(
        self,
        dataframe,                 # pandas DataFrame with column 'ImageID'
        root_dir: str,
        json_file: str,            # grounded_reports_*.json
        max_txt_len: int = 64,
        image_size: int = 1024,
        normalize: bool = True,
        transform=None,
        return_paths: bool = False,
        sentence_key: str = "sentence_en",
    ):
        self.root_dir = root_dir
        self.img_ids: List[str] = dataframe["ImageID"].tolist()
        self.img_paths: List[str] = [os.path.join(root_dir, x) for x in self.img_ids]
        self.return_paths = return_paths

        # Build per-ImageID text in the same order as dataframe
        # (join all sentence_en found in JSON for that ImageID)
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        by_id: Dict[str, List[Dict[str, Any]]] = {
            d["ImageID"]: d.get("findings", []) for d in data
        }
        texts: List[str] = []
        for img_id in self.img_ids:
            findings = by_id.get(img_id, [])
            joined = " ".join(
                (f.get(sentence_key) or "").strip()
                for f in findings if f.get(sentence_key)
            ).strip()
            texts.append(joined)

        # Image transforms (DINO-friendly normalization by default)
        tfs = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        if normalize:
            tfs.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]))
        self.transform = transform or transforms.Compose(tfs)

        self.texts = texts  # keep raw if you need it for eval/debug

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        with Image.open(self.img_paths[idx]).convert("RGB") as im:
            image = self.transform(im)
        findings = self.texts[idx]
        full_path = self.img_paths[idx]
        # return item
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
    def __init__(self, tokenizer, pad_id, max_len: int | None = 256):
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.max_len = max_len  # max total tokens including BOS/EOS

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])

        ids = []
        for b in batch:
            seq = self.tokenizer.encode(b["label"], add_special_tokens=True)
            if self.max_len is not None and len(seq) > self.max_len:
                # keep BOS + (max_len-2) middle tokens + EOS
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if bos_id is not None and eos_id is not None and len(seq) >= 2:
                    seq = [bos_id] + seq[1:self.max_len-1] + [eos_id]
                else:
                    seq = seq[:self.max_len]
            ids.append(torch.tensor(seq, dtype=torch.long))

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

# ========== DINO Encoder ==========
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
        tokens = out.last_hidden_state           # (B, 1 + num_patches [+ extra], D_img)
        cls = tokens[:, 0, :]                    # (B, D_img)
        patches = tokens[:, 5:, :]               # skip some early tokens if present
        return cls, patches                      

# ========== GPT-like Decoder with Visual Prefix ==========
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, D)
        B, L, D = x.size()
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        def shape(t):
            return t.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, L, Hd)

        q = shape(q)
        k = shape(k)
        v = shape(v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)

        # ---- FIXED: broadcast masks to (1,1,L,L) and (B,1,1,L) ----
        if attn_mask is not None:
            # attn_mask expected as (L, L) bool
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]  # (1,1,L,L)
            att = att.masked_fill(~attn_mask.to(att.device), float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: (B, L) True = mask keys
            kpm = key_padding_mask[:, None, None, :]      # (B,1,1,L)
            att = att.masked_fill(kpm.to(att.device), float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, H, L, Hd)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expansion * d_model)
        self.fc2 = nn.Linear(expansion * d_model, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio=4, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, attn_dropout, resid_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, resid_dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class VisualPrefixGPTDecoder(nn.Module):
    """
    Decoder-only transformer (GPT-like). It receives:
      - visual prefix tokens derived from DINO (CLS + first N patch tokens after projection),
      - text token embeddings,
    concatenated and fed through causal self-attention.

    Loss is computed only on text positions (standard next-token).
    """
    def __init__(
        self,
        vocab_size,
        d_img,
        d_model=512,
        n_layer=8,
        n_head=8,
        n_prefix=8,           # number of visual prefix tokens (includes CLS)
        pad_id=0,
        max_seq_len=256,
        dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_prefix = n_prefix
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

        # Text embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len + n_prefix + 8, d_model)  # a bit extra room

        # Project image features to model dim
        self.proj_img = nn.Linear(d_img, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_head, mlp_ratio=4, attn_dropout=0.0, resid_dropout=dropout)
        for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def _build_visual_prefix(self, patches, cls):
        """
        patches: (B, Np, D_img)
        cls: (B, D_img)
        Returns: prefix embeddings (B, P, D) with P = n_prefix
        """
        B, Np, _ = patches.size()
        cls_proj = self.proj_img(cls).unsqueeze(1)  # (B, 1, D)
        patches_proj = self.proj_img(patches)       # (B, Np, D)

        if self.n_prefix <= 1:
            return cls_proj

        if Np >= (self.n_prefix - 1):
            take = patches_proj[:, :self.n_prefix - 1, :]
        else:
            # pad with zeros if not enough patches
            pad_len = self.n_prefix - 1 - Np
            pad = torch.zeros(B, pad_len, patches_proj.size(-1), device=patches_proj.device, dtype=patches_proj.dtype)
            take = torch.cat([patches_proj, pad], dim=1)
        prefix = torch.cat([cls_proj, take], dim=1)  # (B, n_prefix, D)
        return prefix

    def forward(self, patches, cls, input_ids):
        """
        input_ids: (B, T) text tokens used as inputs (teacher forcing)
        Returns logits for text positions: (B, T, V)
        """
        B, T = input_ids.size()
        device = input_ids.device

        # Build prefix (B, P, D)
        prefix = self._build_visual_prefix(patches, cls)
        P = prefix.size(1)

        # Embeddings
        tok = self.tok_emb(input_ids)                     # (B, T, D)
        # Positional embeddings for the concatenated sequence [prefix, text]
        pos_idx = torch.arange(P + T, device=device).unsqueeze(0).expand(B, P + T)
        pos = self.pos_emb(pos_idx)                       # (B, P+T, D)

        x = torch.cat([prefix, tok], dim=1) + pos         # (B, P+T, D)

        # Causal mask (allow attending only to previous or same positions)
        causal = torch.tril(torch.ones(P + T, P + T, device=device, dtype=torch.bool))
        # Key padding mask: no padding for prefix; padding where input_ids == pad_id for text
        pad_mask_text = (input_ids == self.pad_id)        # (B, T)
        key_padding_mask = torch.cat([torch.zeros(B, P, device=device, dtype=torch.bool), pad_mask_text], dim=1)  # (B, P+T)

        for blk in self.blocks:
            x = blk(x, attn_mask=causal, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)
        # Take only the text positions for LM head
        x_text = x[:, P:, :]                               # (B, T, D)
        logits = self.lm_head(x_text)                      # (B, T, V)
        return logits

    @torch.no_grad()
    def generate(self, patches, cls, bos_id, eos_id, max_new_tokens=50, top_p=0.9, temperature=1.0, greedy=False):
        """
        Autoregressive generation (no KV-cache for simplicity).
        """
        B = cls.size(0)
        device = cls.device

        prefix = self._build_visual_prefix(patches, cls)   # (B, P, D)
        P = prefix.size(1)

        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            T = seq.size(1)
            # embeddings
            tok = self.tok_emb(seq)                        # (B, T, D)
            pos_idx = torch.arange(P + T, device=device).unsqueeze(0).expand(B, P + T)
            pos = self.pos_emb(pos_idx)
            x = torch.cat([prefix, tok], dim=1) + pos

            # masks
            causal = torch.tril(torch.ones(P + T, P + T, device=device, dtype=torch.bool))
            pad_mask_text = (seq == self.pad_id)
            key_padding_mask = torch.cat([torch.zeros(B, P, device=device, dtype=torch.bool), pad_mask_text], dim=1)

            for blk in self.blocks:
                x = blk(x, attn_mask=causal, key_padding_mask=key_padding_mask)
            x = self.ln_f(x)
            logits = self.lm_head(x[:, -1, :]) / max(1e-6, temperature)  # (B, V)

            # avoid producing BOS; force minimal length before EOS
            logits[:, bos_id] = -1e9
            if seq.size(1) < 2:
                logits[:, eos_id] = -1e9

            probs = torch.softmax(logits, dim=-1)
            if greedy:
                next_tok = probs.argmax(dim=-1)
            else:
                # nucleus (top-p) sampling
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumprobs > top_p
                cutoff[:, 0] = False
                sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_pos = torch.multinomial(sorted_probs, num_samples=1).squeeze(1)
                next_tok = sorted_idx.gather(1, next_pos.unsqueeze(1)).squeeze(1)

            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
            if torch.all(next_tok == eos_id):
                break

        return seq

# ========== Full Captioner ==========
class DinoGPTCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_img,
        pad_id=0,
        d_model=512,
        n_layer=8,
        n_head=8,
        n_prefix=8,
        max_seq_len=256,
        dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_dino=True
    ):
        super().__init__()
        self.encoder = DINOEncoder(dino_model_id, freeze=freeze_dino)
        self.decoder = VisualPrefixGPTDecoder(
            vocab_size=vocab_size,
            d_img=d_img,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            n_prefix=n_prefix,
            pad_id=pad_id,
            max_seq_len=max_seq_len,
            dropout=0.1
        )

    def forward(self, pixel_values, input_ids):
        cls, patches = self.encoder(pixel_values)
        logits = self.decoder(patches, cls, input_ids)
        return logits

    @torch.no_grad()
    def generate(self, pixel_values, bos_id, eos_id, **gen_kwargs):
        cls, patches = self.encoder(pixel_values)
        return self.decoder.generate(patches, cls, bos_id=bos_id, eos_id=eos_id, **gen_kwargs)

# ========== Training Utils ==========
def sequence_ce_loss(logits, labels, pad_id):
    """
    logits: (B, T, V) — corresponds to input_ids[:, :] positions
    labels: (B, T) — next tokens; pad ignored
    """
    B, T, V = logits.size()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    return loss_fn(logits.reshape(B * T, V), labels.reshape(B * T))

@torch.no_grad()
def batch_perplexity(logits, labels, pad_id):
    loss = sequence_ce_loss(logits, labels, pad_id)
    return float(math.exp(min(loss.item(), 20.0)))

def train_one_epoch(model, loader, optimizer, device, pad_id, num_batches, grad_clip=1.0):
    model.train()
    total_loss, total_pp, steps = 0.0, 0.0, 0
    for pixel_values, tgt_ids, *_ in tqdm(loader, desc="Training", total=num_batches):
        pixel_values = pixel_values.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)

        # Shift: inputs are tokens[:-1], labels are tokens[1:]
        input_ids = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values, input_ids)
        loss = sequence_ce_loss(logits, labels, pad_id)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            ppl = batch_perplexity(logits, labels, pad_id)
        total_loss += loss.item()
        total_pp += ppl
        steps += 1
        del pixel_values, tgt_ids, input_ids, labels, logits, loss, ppl
        torch.cuda.empty_cache()
        if steps > num_batches:
            break
    return {"loss": total_loss / steps, "ppl": total_pp / steps}

@torch.no_grad()
def evaluate(model, loader, device, pad_id, num_batches):
    model.eval()
    total_loss, total_pp, steps = 0.0, 0.0, 0
    for pixel_values, tgt_ids, *_ in tqdm(loader, desc="Evaluating", total=num_batches):
        pixel_values = pixel_values.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)
        input_ids = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]
        logits = model(pixel_values, input_ids)
        loss = sequence_ce_loss(logits, labels, pad_id)
        ppl = batch_perplexity(logits, labels, pad_id)
        total_loss += loss.item()
        total_pp += ppl
        steps += 1
        
        del pixel_values, tgt_ids, input_ids, labels, logits, loss, ppl
        torch.cuda.empty_cache()
        if steps > num_batches:
            break
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

#     # DINO expects 224 or 518 square; 224 is fine here
#     IMG_SIZE = 1024
#     tf = dino_image_transform(img_size=IMG_SIZE)
#     ds = CheXpertDataset(img_root=IMG_ROOT, csv=valid_df, transform=tf, text_col=TEXT_COL)
#     collate_fn = CaptionCollate(tokenizer, pad_id)

#     is_windows = os.name == "nt"
#     num_workers = 0 if is_windows else 2
#     persistent_workers = False if num_workers == 0 else True

#     # Full loader (used to sample subsets below)
#     full_loader = DataLoader(
#         ds,
#         batch_size=8,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         persistent_workers=persistent_workers,
#         collate_fn=collate_fn
#     )

#     # Simple split
#     n_total = len(ds)
#     n_train = int(n_total * 0.8)
#     indices = torch.randperm(n_total).tolist()
#     train_idx, valid_idx = indices[:n_train], indices[n_train:]
#     train_ds = torch.utils.data.Subset(ds, train_idx)
#     valid_ds = torch.utils.data.Subset(ds, valid_idx)
#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
#     valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

#     # DINO ViT-S/16 hidden size is 384 (for this checkpoint); adjust if you change encoder
#     D_IMG = 384
#     N_PREFIX = 1 #(IMG_SIZE // 16) ** 2  # number of visual prefix tokens (including CLS)
#     model = DinoGPTCaptioner(
#         vocab_size=tokenizer.vocab_size,
#         d_img=D_IMG,
#         pad_id=pad_id,
#         d_model=512,
#         n_layer=8,
#         n_head=8,
#         n_prefix=N_PREFIX,           # number of visual prefix tokens
#         max_seq_len=256,
#         dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
#         freeze_dino=True,
#     ).to(device)

#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2
#     )

#     # ---- Train a few slices just to validate wiring ----
#     for epoch in range(20):
#         slice_train_loader = islice(train_loader, 20)
#         slice_valid_loader = islice(valid_loader, 20)
#         train_stats = train_one_epoch(model, slice_train_loader, optimizer, device, pad_id, num_batches=20, grad_clip=1.0)
#         val_stats = evaluate(model, slice_valid_loader, device, pad_id, num_batches=20)
#         print(f"Epoch {epoch + 1}: Train Loss={train_stats['loss']:.4f}, PPL={train_stats['ppl']:.2f} | "
#               f"Val Loss={val_stats['val_loss']:.4f}, Val PPL={val_stats['val_ppl']:.2f}")

#     # ---- Quick generation sanity check ----
#     with torch.no_grad():
#         for pixel_values, ids_loader, paths, raw_labels in valid_loader:
#             pixel_values = pixel_values.to(device)
#             gen_ids = model.generate(
#                 pixel_values=pixel_values,
#                 bos_id=bos_id, eos_id=eos_id,
#                 max_new_tokens=256, top_p=0.9, temperature=0.9, greedy=True
#             )
#             print("Predictions (first batch):")
#             for i in range(min(gen_ids.size(0), 8)):
#                 print(f"\nGEN {i+1}:", tokenizer.decode(gen_ids[i].tolist()))
#                 print(f"TGT {i+1}:", tokenizer.decode(ids_loader[i].tolist()))
#             del pixel_values, ids_loader, paths, raw_labels, gen_ids
#             torch.cuda.empty_cache()
#             break

# if __name__ == "__main__":
#     main()
