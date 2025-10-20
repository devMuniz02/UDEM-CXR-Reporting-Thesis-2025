from xml.parsers.expat import model
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from transformers import GPT2Tokenizer
from torch.nn import functional as F

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

def build_tokenizer_from_labels(captions=None, gpt2=True):
    try:
        from transformers import AutoTokenizer
        HF_AVAILABLE = True
    except Exception:
        HF_AVAILABLE = False
    if captions is None and HF_AVAILABLE:
        if gpt2:
            tok = GPT2Tokenizer.from_pretrained("gpt2")
            print("Using GPT2 tokenizer.")
            class _Wrap:
                def __init__(self, tok):
                    self.tok = tok
                    self.bos_token_id = tok.eos_token_id  # GPT2 uses  as both bos and eos
                    self.eos_token_id = tok.eos_token_id
                    self.pad_token_id = tok.eos_token_id  # No pad token in GPT2, use eos
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
            tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1-squad")
        if tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": "[PAD]"})
        class _Wrap:
            def __init__(self, tok):
                self.tok = tok
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

class CaptionCollate:
    def __init__(self, tokenizer, pad_id, max_len=None):
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        ids = []
        for b in batch:
            seq = self.tokenizer.encode(b["label"], add_special_tokens=True)
            if self.max_len is not None and len(seq) > self.max_len:
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

def sequence_ce_loss(logits, labels, pad_id):
    B, T, V = logits.size()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.2)
    return loss_fn(logits.reshape(B * T, V), labels.reshape(B * T))

@torch.no_grad()
def batch_perplexity(logits, labels, pad_id):
    import math
    return math.exp(sequence_ce_loss(logits, labels, pad_id).item())


def train_one_epoch(model, loader, optimizer, device, pad_id, num_batches, loss_fn=sequence_ce_loss, grad_clip=1.0):
    model.train()
    total_loss, total_pp, steps = 0.0, 0.0, 0
    from tqdm import tqdm
    for pixel_values, tgt_ids, *_ in tqdm(loader, desc="Training", total=num_batches):
        pixel_values = pixel_values.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)
        input_ids = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(pixel_values, input_ids)
            loss = loss_fn(logits, labels, pad_id)
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
def evaluate(model, loader, device, pad_id, num_batches, loss_fn=sequence_ce_loss):
    model.eval()
    total_loss, total_pp, steps = 0.0, 0.0, 0
    from tqdm import tqdm
    for pixel_values, tgt_ids, *_ in tqdm(loader, desc="Evaluating", total=num_batches):
        pixel_values = pixel_values.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)
        input_ids = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]
        logits = model(pixel_values, input_ids)
        loss = loss_fn(logits, labels, pad_id)
        ppl = batch_perplexity(logits, labels, pad_id)
        total_loss += loss.item()
        total_pp += ppl
        steps += 1
        del pixel_values, tgt_ids, input_ids, labels, logits, loss, ppl
        torch.cuda.empty_cache()
        if steps > num_batches:
            break
    return {"val_loss": total_loss / steps, "val_ppl": total_pp / steps}