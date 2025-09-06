# --------------------------------------------
# DINO → (patch tokens, CLS)  +  LSTM+Bahdanau
# --------------------------------------------
# pip install torch torchvision transformers pillow
# (Make sure the DINO & tokenizer checkpoints are available locally or via HF cache)

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T

try:
    from transformers import AutoModel, AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


# -----------------------
# Tokenizer (two options)
# -----------------------

class SimpleWordTokenizer:
    """
    Minimal whitespace tokenizer as a fallback if HF isn't available.
    Builds a vocab from provided texts.
    """
    def __init__(self, texts: List[str], min_freq: int = 1,
                 bos_token: str = "<bos>", eos_token: str = "<eos>", pad_token: str = "<pad>", unk_token: str = "<unk>"):
        from collections import Counter
        self.bos_token, self.eos_token, self.pad_token, self.unk_token = bos_token, eos_token, pad_token, unk_token
        counter = Counter()
        for t in texts:
            counter.update(t.strip().lower().split())
        self.itos = [pad_token, bos_token, eos_token, unk_token] + [w for w, c in counter.items() if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_token_id = self.stoi[pad_token]
        self.bos_token_id = self.stoi[bos_token]
        self.eos_token_id = self.stoi[eos_token]
        self.unk_token_id = self.stoi[unk_token]

    def encode(self, text: str, add_special_tokens=True) -> List[int]:
        toks = [self.stoi.get(w, self.unk_token_id) for w in text.strip().lower().split()]
        if add_special_tokens:
            return [self.bos_token_id] + toks + [self.eos_token_id]
        return toks

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
                continue
            words.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


# -----------------
# Image pre-process
# -----------------

def dino_image_transform(img_size: int = 1024) -> T.Compose:
    # Standard ViT normalization
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


# ---------------
# DINO Encoder
# ---------------

class DINOEncoder(nn.Module):
    """
    Wraps a DINO ViT (e.g., facebook/dinov3-vits16-pretrain-lvd1689m).
    Returns:
      - cls: (B, D_img)
      - patches: (B, N, D_img)
    """
    def __init__(self, model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m", freeze: bool = True):
        super().__init__()
        if not HF_AVAILABLE:
            raise RuntimeError("Hugging Face transformers not available in this environment.")
        self.model = AutoModel.from_pretrained(model_id)  # expects pixel_values normalized tensors
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pixel_values: (B, 3, H, W), normalized
        """
        out = self.model(pixel_values=pixel_values)
        # ViT-like output: last_hidden_state = (B, 1+N, D)
        tokens = out.last_hidden_state
        cls = tokens[:, 0, :]         # (B, D)
        patches = tokens[:, 5:, :]    # (B, 196, D)
        return cls, patches


# -----------------------
# Bahdanau (Additive) Attn
# -----------------------

class BahdanauAttention(nn.Module):
    """
    Additive attention over image patch keys.
    h_t (decoder state) attends over keys (B, N, d_h).
    Returns context (B, d_h) and weights (B, N)
    """
    def __init__(self, d_h: int):
        super().__init__()
        self.W_h = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.v = nn.Linear(d_h, 1, bias=False)

    def forward(self, h_t: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_t: (B, d_h), keys: (B, N, d_h)
        B, N, d_h = keys.size()
        # (B, 1, d_h) + (B, N, d_h) -> (B, N, d_h)
        energy = torch.tanh(self.W_h(h_t).unsqueeze(1) + self.W_k(keys))
        # (B, N, 1) -> (B, N)
        scores = self.v(energy).squeeze(-1)
        alpha = torch.softmax(scores, dim=-1)     # (B, N)
        # weighted sum -> (B, d_h)
        context = torch.bmm(alpha.unsqueeze(1), keys).squeeze(1)
        return context, alpha


# --------------------------
# LSTM Decoder w/ Attention
# -----------------------

class LSTMAttnDecoder(nn.Module):
    """
    LSTMCell-based autoregressive decoder with Bahdanau attention over patch tokens.
    - keys/values come from projected image patches
    - CLS initializes hidden/cell states
    """
    def __init__(self, vocab_size: int, d_img: int, d_h: int = 512,
                 pad_id: int = 0, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_h = d_h
        self.pad_id = pad_id

        # Project DINO channels -> decoder hidden dim
        self.kv_proj = nn.Linear(d_img, d_h)
        self.init_h = nn.Linear(d_img, d_h)
        self.init_c = nn.Linear(d_img, d_h)

        self.emb = nn.Embedding(vocab_size, d_h, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_h)

        # Input to LSTMCell is [embedding || context]
        self.lstm = nn.LSTMCell(d_h * 2, d_h)
        self.attn = BahdanauAttention(d_h)
        self.out = nn.Linear(d_h, vocab_size)

    def forward(self, patches: torch.Tensor, cls: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Teacher forcing.
        patches: (B, N, D_img)
        cls:     (B, D_img)
        tgt_ids: (B, T)   (BOS ... EOS ... PAD)
        Returns logits: (B, T, V)
        """
        B, T = tgt_ids.size()
        device = tgt_ids.device

        # Prepare keys/values and initial states
        keys = self.kv_proj(patches)          # (B, N, d_h)
        h = self.init_h(cls)                  # (B, d_h)
        c = self.init_c(cls)                  # (B, d_h)

        logits = []
        # Teacher forcing: at step t we feed tgt_ids[:, t] (previous token embedding)
        x = self.emb(tgt_ids[:, 0])           # first token should be BOS
        for t in range(T):
            # Attention using current hidden state
            ctx, _ = self.attn(h, keys)       # (B, d_h), (B, N)
            lstm_in = torch.cat([x, ctx], dim=-1)
            h, c = self.lstm(lstm_in, (h, c))
            h = self.ln(self.dropout(h))
            step_logits = self.out(h)         # (B, V)
            logits.append(step_logits.unsqueeze(1))

            # Next input (teacher forcing)
            if t + 1 < T:
                x = self.emb(tgt_ids[:, t + 1])

        return torch.cat(logits, dim=1)       # (B, T, V)

    @torch.no_grad()
    def generate(self,
                 patches: torch.Tensor,
                 cls: torch.Tensor,
                 bos_id: int,
                 eos_id: int,
                 max_new_tokens: int = 30,
                 top_p: float = 0.9,
                 temperature: float = 1.0,
                 greedy: bool = False) -> torch.Tensor:
        """
        Autoregressive decoding.
        Returns sequences including BOS and (if hit) EOS.
        """
        B = cls.size(0)
        device = cls.device

        keys = self.kv_proj(patches)          # (B, N, d_h)
        h = self.init_h(cls)                  # (B, d_h)
        c = self.init_c(cls)

        # start with BOS
        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        x = self.emb(seq[:, -1])

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            ctx, _ = self.attn(h, keys)
            h, c = self.lstm(torch.cat([x, ctx], dim=-1), (h, c))
            h = self.ln(h)
            logits = self.out(h) / max(1e-6, temperature)  # (B, V)
            # --- add these lines ---
            # never output BOS
            logits[:, bos_id] = -1e9
            # avoid instant termination; allow EOS only after a few tokens, e.g., min_len=2
            min_len = 2
            if seq.size(1) < min_len:
                logits[:, eos_id] = -1e9
            # -----------------------
            probs = torch.softmax(logits, dim=-1)

            if greedy:
                next_tokens = probs.argmax(dim=-1)
            else:
                # nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                # mask tokens outside nucleus
                mask = cumprobs > top_p
                # Ensure at least one token
                mask[:, 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                # sample
                next_pos = torch.multinomial(sorted_probs, num_samples=1).squeeze(1)
                next_tokens = sorted_idx.gather(dim=1, index=next_pos.unsqueeze(1)).squeeze(1)

            # update sequence
            seq = torch.cat([seq, next_tokens.unsqueeze(1)], dim=1)
            x = self.emb(next_tokens)

            # EOS handling
            finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

        return seq  # (B, T_total)


# -----------------------------
# Full Captioner (Enc + Dec)
# -----------------------------

class DinoLSTMAttnCaptioner(nn.Module):
    def __init__(self, vocab_size: int, d_img: int,
                 d_h: int = 512, pad_id: int = 0,
                 dino_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
                 freeze_dino: bool = True):
        super().__init__()
        self.encoder = DINOEncoder(dino_model_id, freeze=freeze_dino)
        self.decoder = LSTMAttnDecoder(vocab_size=vocab_size, d_img=d_img, d_h=d_h, pad_id=pad_id)

    def forward(self, pixel_values: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        cls, patches = self.encoder(pixel_values)   # (B,D_img), (B,N,D_img)
        logits = self.decoder(patches, cls, tgt_ids)
        return logits

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(pixel_values)

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, bos_id: int, eos_id: int,
                 max_new_tokens: int = 30, top_p: float = 0.9,
                 temperature: float = 1.0, greedy: bool = False) -> torch.Tensor:
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


# -------------------
# Dataset + Collation
# -------------------

@dataclass
class Sample:
    image_path: str
    caption: str

class ImageCaptionDataset(Dataset):
    def __init__(self, samples: List[Sample], tokenizer, img_transform=None):
        self.samples = samples
        self.tok = tokenizer
        self.tf = img_transform if img_transform is not None else dino_image_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        pixel_values = self.tf(img)  # (3, H, W)
        ids = self.tok.encode(s.caption, add_special_tokens=True)  # [BOS ... EOS]
        return pixel_values, torch.tensor(ids, dtype=torch.long)

def collate_fn(batch, pad_id: int):
    imgs, ids = zip(*batch)
    pixel_values = torch.stack(imgs, dim=0)  # (B, 3, H, W)

    # Teacher forcing targets:
    #   input:  [BOS, w1, ..., w_{T-1}]
    #   target: [w1,  ..., w_{T-1}, EOS]
    seqs = [
        x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
        for x in ids
    ]  # already contains BOS...EOS
    # Left as-is; loss uses ignore_index on PAD so no need to shift here.
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)  # (B, T)
    return pixel_values, padded


# --------------
# Training Utils
# --------------

def sequence_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    logits:  (B, T, V)  — produced while feeding inputs [BOS, w1, ..., w_{T-1}]
    targets: (B, T)     — contains [BOS, w1, ..., w_{T-1}, EOS, PAD...]
    We want to predict targets[:, 1:] from logits[:, :-1, :].
    """
    B, T, V = logits.size()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    logits_next = logits[:, :-1, :].contiguous().view(-1, V)   # predict next token
    targets_next = targets[:, 1:].contiguous().view(-1)
    return loss_fn(logits_next, targets_next)

@torch.no_grad()
def batch_perplexity(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    loss = sequence_ce_loss(logits, targets, pad_id)
    return float(math.exp(min(loss.item(), 20.0)))


# ------------------
# Example Train Loop
# ------------------

def train_one_epoch(model: DinoLSTMAttnCaptioner, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    pad_id: int, grad_clip: float = 1.0) -> Dict[str, float]:
    model.train()
    total_loss, total_pp, steps = 0.0, 0.0, 0
    for pixel_values, tgt_ids in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values, tgt_ids)                # (B,T,V)
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

        # after you get logits and have targets (B,T)
        # print("inp[0][:10] :", tgt_ids[0, :10].tolist())      # [BOS, w1, w2, ...]
        # print("tgt[0][:10] :", tgt_ids[0, 1:11].tolist())     # [w1, w2, ..., EOS]
        # print("logits shape:", logits.shape)                   # (B, T, V)


    return {"loss": total_loss / steps, "ppl": total_pp / steps}


@torch.no_grad()
def evaluate(model: DinoLSTMAttnCaptioner, loader: DataLoader,
             device: torch.device, pad_id: int) -> Dict[str, float]:
    model.eval()
    total_loss, total_pp, steps = 0.0, 0.0, 0
    for pixel_values, tgt_ids in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)
        logits = model(pixel_values, tgt_ids)
        loss = sequence_ce_loss(logits, tgt_ids, pad_id)
        ppl = batch_perplexity(logits, tgt_ids, pad_id)
        total_loss += loss.item()
        total_pp += ppl
        steps += 1
    return {"val_loss": total_loss / steps, "val_ppl": total_pp / steps}


# -------------
# Usage Example
# -------------

def build_tokenizer(captions: List[str]):
    """
    Prefer a HF tokenizer if available (e.g., DistilBERT).
    We'll treat [CLS] as BOS and [SEP] as EOS for simplicity.
    If HF not available, fallback to SimpleWordTokenizer.
    """
    if HF_AVAILABLE:
        tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")#("distilbert-base-uncased")
        # Ensure pad exists
        if tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": "[PAD]"})
        class _Wrap:
            def __init__(self, tok):
                self.tok = tok
                # map: BOS~[CLS], EOS~[SEP]
                self.bos_token_id = tok.cls_token_id
                self.eos_token_id = tok.sep_token_id
                self.pad_token_id = tok.pad_token_id
            def encode(self, text, add_special_tokens=True):
                ids = self.tok.encode(text, add_special_tokens=False)
                return [self.bos_token_id] + ids + [self.eos_token_id] if add_special_tokens else ids
            def decode(self, ids):
                return self.tok.decode([i for i in ids if i not in (self.bos_token_id, self.eos_token_id, self.pad_token_id)],
                                       skip_special_tokens=True)
            @property
            def vocab_size(self):
                return self.tok.vocab_size
        return _Wrap(tok)
    else:
        return SimpleWordTokenizer(captions)

def main():
    import os

    # ---- Prepare toy data ----
    samples = [
        Sample(r"C:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\demo.jpg", "a dog running on the beach"),
        Sample(r"C:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\demo2.jpg", "a small red car parked by the road"),
        Sample(r"C:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\demo3.jpg", "two people riding bicycles"),
    ]
    if not all(os.path.exists(s.image_path) for s in samples):
        print("[!] Please replace sample image paths with real files before training.")
        return

    # ---- Add test samples ----
    test_samples = [
        Sample(r"C:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\test1.jpg", "a person walking a dog"),
        Sample(r"C:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\test2.jpg", "a yellow bus driving down the street"),
    ]
    if not all(os.path.exists(s.image_path) for s in test_samples):
        print("[!] Please replace test image paths with real files before testing.")
        return

    tokenizer = build_tokenizer([s.caption for s in samples + test_samples])
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else tokenizer.stoi["<pad>"]
    bos_id = tokenizer.bos_token_id if hasattr(tokenizer, "bos_token_id") else tokenizer.stoi["<bos>"]
    eos_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.stoi["<eos>"]

    tf = dino_image_transform(img_size=1024)
    ds = ImageCaptionDataset(samples, tokenizer, img_transform=tf)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))

    test_ds = ImageCaptionDataset(test_samples, tokenizer, img_transform=tf)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    D_IMG = 384
    model = DinoLSTMAttnCaptioner(
        vocab_size=tokenizer.vocab_size,
        d_img=D_IMG,
        d_h=512,
        pad_id=pad_id,
        dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_dino=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2
    )

    for iteration in range(5):
        print(f"\n=== Iteration {iteration + 1} ===")
        # ---- Training (toy) ----
        for epoch in range(10):
            train_stats = train_one_epoch(model, loader, optimizer, device, pad_id, grad_clip=1.0)
            val_stats = evaluate(model, loader, device, pad_id)

        # ---- Inference example ----
        model.eval()
        with torch.no_grad():
            pixel_values, ids_loader = next(iter(loader))
            pixel_values = pixel_values.to(device)
            gen_ids = model.generate(
                pixel_values=pixel_values,
                bos_id=bos_id, eos_id=eos_id,
                max_new_tokens=15, top_p=0.9, temperature=0.9, greedy=False
            )
            print("Predictions train:")
            for i in range(gen_ids.size(0)):
                print("\nGEN:", tokenizer.decode(gen_ids[i].tolist()))
                print("Tokens:", gen_ids[i].tolist())
                print("GT :", tokenizer.decode(ids_loader[i].tolist()))
                print("Tokens:", ids_loader[i].tolist())

        # ---- Test set inference ----
        with torch.no_grad():
            for pixel_values, ids_loader in test_loader:
                pixel_values = pixel_values.to(device)
                gen_ids = model.generate(
                    pixel_values=pixel_values,
                    bos_id=bos_id, eos_id=eos_id,
                    max_new_tokens=15, top_p=0.9, temperature=0.9, greedy=False
                )
                print("Predictions test:")
                for i in range(gen_ids.size(0)):
                    print("\nTEST GEN:", tokenizer.decode(gen_ids[i].tolist()))
                    print("Tokens:", gen_ids[i].tolist())
                    print("TEST GT :", tokenizer.decode(ids_loader[i].tolist()))
                    print("Tokens:", ids_loader[i].tolist())

if __name__ == "__main__":
    main()
