import torch
import torch.nn as nn
import math

from transformers import GPT2Tokenizer, GPT2LMHeadModel

class DINOEncoder(nn.Module):
    def __init__(self, model_id="facebook/dinov3-vits16-pretrain-lvd1689m", freeze=True):
        super().__init__()
        from transformers import AutoModel
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
        B, L, D = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def shape(t):
            return t.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        q = shape(q)
        k = shape(k)
        v = shape(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]
            att = att.masked_fill(~attn_mask.to(att.device), float('-inf'))
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :]
            att = att.masked_fill(kpm.to(att.device), float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
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
    def __init__(
        self,
        vocab_size,
        d_img,
        d_model=512,
        n_layer=8,
        n_head=8,
        n_prefix=8,
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
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len + n_prefix + 8, d_model)
        self.proj_img = nn.Linear(d_img, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_head, mlp_ratio=4, attn_dropout=0.0, resid_dropout=dropout)
            for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def _ensure_pos_emb(self, need_len: int, device):
        # need_len is P + T
        cur = self.pos_emb.num_embeddings
        if need_len <= cur:
            return
        new = nn.Embedding(need_len, self.d_model, device=device)
        with torch.no_grad():
            new.weight[:cur].copy_(self.pos_emb.weight)
            nn.init.normal_(new.weight[cur:], mean=0.0, std=0.02)
        self.pos_emb = new


    def _build_visual_prefix(self, patches, cls):
        B, Np, _ = patches.size()
        cls_proj = self.proj_img(cls).unsqueeze(1)
        patches_proj = self.proj_img(patches)
        if self.n_prefix <= 1:
            return cls_proj
        if Np >= (self.n_prefix - 1):
            take = patches_proj[:, :self.n_prefix - 1, :]
        else:
            pad_len = self.n_prefix - 1 - Np
            pad = torch.zeros(B, pad_len, patches_proj.size(-1), device=patches_proj.device, dtype=patches_proj.dtype)
            take = torch.cat([patches_proj, pad], dim=1)
        prefix = torch.cat([cls_proj, take], dim=1)
        return prefix

    def forward(self, patches, cls, input_ids):
        B, T = input_ids.size()
        device = input_ids.device
        prefix = self._build_visual_prefix(patches, cls)
        P = prefix.size(1)
        total = P + T
        self._ensure_pos_emb(total, device)
        tok = self.tok_emb(input_ids)
        pos_idx = torch.arange(P + T, device=device).unsqueeze(0).expand(B, P + T)
        pos = self.pos_emb(pos_idx)
        x = torch.cat([prefix, tok], dim=1) + pos
        causal = torch.tril(torch.ones(P + T, P + T, device=device, dtype=torch.bool))
        pad_mask_text = (input_ids == self.pad_id)
        key_padding_mask = torch.cat([torch.zeros(B, P, device=device, dtype=torch.bool), pad_mask_text], dim=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=causal, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        x_text = x[:, P:, :]
        logits = self.lm_head(x_text)
        return logits

    @torch.no_grad()
    def generate(self, patches, cls, bos_id, eos_id, max_new_tokens=50, beam_size=3, temperature=1.0):
        B = cls.size(0)
        device = cls.device
        prefix = self._build_visual_prefix(patches, cls)
        P = prefix.size(1)
        # Each batch is handled independently for beam search
        final_seqs = []
        for b in range(B):
            beams = [(torch.full((1, 1), bos_id, dtype=torch.long, device=device), 0.0, False)]  # (seq, score, ended)
            for _ in range(max_new_tokens):
                new_beams = []
                for seq, score, ended in beams:
                    if ended:
                        new_beams.append((seq, score, True))
                        continue
                    T = seq.size(1)
                    tok = self.tok_emb(seq)
                    total = P + T
                    self._ensure_pos_emb(total, device)
                    pos_idx = torch.arange(P + T, device=device).unsqueeze(0)
                    pos = self.pos_emb(pos_idx)
                    x = torch.cat([prefix[b:b+1], tok], dim=1) + pos
                    causal = torch.tril(torch.ones(P + T, P + T, device=device, dtype=torch.bool))
                    pad_mask_text = (seq == self.pad_id)
                    key_padding_mask = torch.cat([torch.zeros(1, P, device=device, dtype=torch.bool), pad_mask_text], dim=1)
                    for blk in self.blocks:
                        x = blk(x, attn_mask=causal, key_padding_mask=key_padding_mask)
                    x = self.ln_f(x)
                    logits = self.lm_head(x[:, -1, :]) / max(1e-6, temperature)
                    logits[:, bos_id] = -1e9
                    if seq.size(1) < 2:
                        logits[:, eos_id] = -1e9
                    probs = torch.softmax(logits, dim=-1)
                    topk_probs, topk_idx = torch.topk(probs, beam_size, dim=-1)
                    for k in range(beam_size):
                        next_tok = topk_idx[0, k]
                        next_prob = topk_probs[0, k].item()
                        new_seq = torch.cat([seq, next_tok.view(1, 1)], dim=1)
                        new_score = score + torch.log(topk_probs[0, k] + 1e-12).item()
                        ended_flag = (next_tok == eos_id)
                        new_beams.append((new_seq, new_score, ended_flag))
                # Keep top beam_size beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                # If all beams ended, break
                if all(e for _, _, e in beams):
                    break
            # Select the best beam (highest score)
            best_seq = beams[0][0]
            final_seqs.append(best_seq)
        # Pad sequences to same length
        max_len = max(seq.size(1) for seq in final_seqs)
        out = torch.full((B, max_len), self.pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(final_seqs):
            out[i, :seq.size(1)] = seq
        return out
    
    @torch.no_grad()
    def generate_with_logging(
        self,
        patches: torch.Tensor,                 # [B, Np, D_img]
        cls: torch.Tensor,                     # [B, D_img]
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 256,
        # decoding preset/overrides
        preset: str = "safe_sample",           # "greedy" | "safe_sample" | "creative"
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        # stopping/logging
        stop_sequences: list[str] | None = None,
        tokenizer=None,                        # optional, to decode/encode stop seqs
        log_first_n_steps: int = 6,
        log_topk: int = 5,
        return_text: bool = True,
    ) -> dict:
        """
        Batch-capable logging generator. Iterates over B and returns:
        {
            'per_sample': [ {logging dict for sample 0}, ... ],
            'sequences': LongTensor[B, Tmax]  # padded with self.pad_id
        }
        """
        device = cls.device
        B = cls.size(0)
        results = []
        seqs: list[torch.Tensor] = []

        # ---- preset
        def _preset(name: str):
            name = (name or "safe_sample").lower()
            if name == "greedy":
                return dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                            repetition_penalty=1.0, no_repeat_ngram_size=0)
            if name == "creative":
                return dict(do_sample=True, temperature=0.9, top_p=0.95, top_k=100,
                            repetition_penalty=1.10, no_repeat_ngram_size=4)
            return dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
                        repetition_penalty=1.15, no_repeat_ngram_size=3)

        base_cfg = _preset(preset)
        if do_sample is not None: base_cfg["do_sample"] = do_sample
        if temperature is not None: base_cfg["temperature"] = temperature
        if top_p is not None: base_cfg["top_p"] = top_p
        if top_k is not None: base_cfg["top_k"] = top_k
        if repetition_penalty is not None: base_cfg["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size is not None: base_cfg["no_repeat_ngram_size"] = no_repeat_ngram_size

        # Pre-encode stop sequences (once)
        stop_ids = []
        if tokenizer is not None and stop_sequences:
            for s in stop_sequences:
                stop_ids.append(tokenizer.encode(s, add_special_tokens=False))

        # helpers for repetition stats
        def _max_token_run(ids: list[int]) -> int:
            if not ids: return 0
            best = cur = 1
            for i in range(1, len(ids)):
                if ids[i] == ids[i-1]:
                    cur += 1; best = max(best, cur)
                else:
                    cur = 1
            return best

        def _max_repeated_ngram(ids: list[int], n: int) -> int:
            if len(ids) < n: return 0
            counts = {}
            for i in range(len(ids)-n+1):
                ng = tuple(ids[i:i+n])
                counts[ng] = counts.get(ng, 0) + 1
            return max(counts.values()) if counts else 0

        def _ends_with(seq: list[int], suffix: list[int]) -> bool:
            if len(suffix) == 0 or len(seq) < len(suffix): return False
            return seq[-len(suffix):] == suffix

        def _entropy_from_logits(logits: torch.Tensor) -> float:
            p = torch.softmax(logits, dim=-1)
            return float(-(p * (p.clamp_min(1e-12).log())).sum().item())

        # per-sample loop
        for b in range(B):
            cfg = dict(base_cfg)  # copy
            # build prefix for this sample
            prefix = self._build_visual_prefix(patches[b:b+1], cls[b:b+1])  # [1, P, D]
            P = prefix.size(1)

            # inner helpers use this sample's prefix
            def _prepare_x(seq_ids: torch.Tensor):
                T = seq_ids.size(1)
                tok = self.tok_emb(seq_ids)
                self._ensure_pos_emb(P + T, device)
                pos = self.pos_emb(torch.arange(P + T, device=device).unsqueeze(0))
                x = torch.cat([prefix, tok], dim=1) + pos
                causal = torch.tril(torch.ones(P + T, P + T, device=device, dtype=torch.bool))
                pad_mask_text = (seq_ids == self.pad_id)
                kpm = torch.cat([torch.zeros(1, P, device=device, dtype=torch.bool), pad_mask_text], dim=1)
                for blk in self.blocks:
                    x = blk(x, attn_mask=causal, key_padding_mask=kpm)
                x = self.ln_f(x)
                return x

            def _apply_repetition_penalty_(logits: torch.Tensor, prev_ids: list[int], penalty: float):
                if penalty is None or penalty == 1.0 or len(prev_ids) == 0:
                    return logits
                l = logits.view(-1)
                for t in set(prev_ids):
                    v = l[t]
                    l[t] = v / penalty if v > 0 else v * penalty
                return logits

            def _ban_ngram_repeats_(logits: torch.Tensor, prev_ids: list[int], n: int):
                if n is None or n <= 1 or len(prev_ids) < n - 1:
                    return logits
                n_1 = n - 1
                context = prev_ids[-n_1:] if n_1 > 0 else []
                bans = set()
                for i in range(len(prev_ids) - n_1):
                    if prev_ids[i:i+n_1] == context:
                        bans.add(prev_ids[i+n_1])
                if bans:
                    l = logits.view(-1)
                    l[list(bans)] = -1e9
                return logits

            def _top_k_top_p_filtering_(logits: torch.Tensor, top_k: int, top_p: float):
                # top-k
                if top_k is not None and top_k > 0 and top_k < logits.numel():
                    kth = torch.topk(logits, top_k).values[-1]
                    logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
                # top-p
                if top_p is not None and top_p < 1.0:
                    probs = torch.softmax(logits, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum > top_p
                    mask[..., 0] = False
                    to_remove = torch.zeros_like(probs, dtype=torch.bool)
                    to_remove.scatter_(dim=-1, index=sorted_idx, src=mask)
                    logits = torch.where(to_remove, torch.full_like(logits, -1e9), logits)
                return logits

            # sampling loop for this sample
            seq = torch.tensor([[bos_id]], dtype=torch.long, device=device)   # [1, 1]
            gen_only: list[int] = []
            step_logs = []
            hit_eos_at = None

            for step in range(max_new_tokens):
                x = _prepare_x(seq)
                logits = self.lm_head(x[:, -1, :]).squeeze(0)                 # [V]
                logits[bos_id] = -1e9
                if seq.size(1) < 2:
                    logits[eos_id] = -1e9

                # temperature + anti-repetition
                logits = logits / max(1e-6, float(cfg["temperature"]))
                logits = _apply_repetition_penalty_(logits, gen_only, cfg["repetition_penalty"])
                logits = _ban_ngram_repeats_(logits, seq[0].tolist(), cfg["no_repeat_ngram_size"])

                # filtered dist for selection + logging
                filt = logits.clone()
                filt = _top_k_top_p_filtering_(filt, cfg["top_k"], cfg["top_p"])
                probs = torch.softmax(filt, dim=-1)

                # log first N steps
                if step < log_first_n_steps:
                    kdisp = min(logits.numel(), max(1, min(5, log_topk)))
                    topv, topi = torch.topk(probs, kdisp)
                    step_logs.append({
                        "step": step+1,
                        "entropy": _entropy_from_logits(filt),
                        "topk": [{"token_id": int(topi[j]), "p": float(topv[j])} for j in range(kdisp)]
                    })

                # pick next token
                if cfg["do_sample"]:
                    next_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_id = int(torch.argmax(filt).item())

                seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
                gen_only.append(next_id)

                # stop by EOS
                if next_id == eos_id:
                    hit_eos_at = len(gen_only) - 1
                    break

                # stop by custom stop sequences
                if stop_ids:
                    for sfx in stop_ids:
                        if _ends_with(seq[0].tolist(), sfx):
                            hit_eos_at = len(gen_only) - 1
                            break
                if hit_eos_at is not None:
                    break

            # collect per-sample info
            rep_stats = {
                "max_token_run": _max_token_run(gen_only),
                "max_repeat_trigram": _max_repeated_ngram(gen_only, 3),
                "max_repeat_4gram": _max_repeated_ngram(gen_only, 4),
            }
            text = {}
            if return_text and tokenizer is not None:
                text["generated"] = tokenizer.decode(gen_only)
                text["full"] = tokenizer.decode(seq[0].tolist())

            results.append({
                "preset": preset,
                "params": base_cfg,
                "lengths": {"prompt_tokens": 1, "new_tokens": len(gen_only), "total_tokens": seq.size(1)},
                "stopping": {"hit_eos": hit_eos_at is not None, "eos_pos": hit_eos_at,
                            "stop_sequences": stop_sequences or []},
                "repetition": rep_stats,
                "probes": step_logs,
                "ids": {"gen_only_ids": gen_only, "full_ids": seq[0].tolist()},
                "text": text,
            })
            seqs.append(seq[0])

        # pad sequences to same length
        Tmax = max(s.size(0) for s in seqs)
        out = torch.full((B, Tmax), self.pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s

        return {"per_sample": results, "sequences": out}


import torch
import math
from torch.nn.functional import softmax
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from transformers import StoppingCriteria, StoppingCriteriaList

# --- optional stop sequences (string-based)
class StopOnSequences(StoppingCriteria):
    def __init__(self, tokenizer, stop_seqs: List[str]):
        self.tokenizer = tokenizer
        # encode without special tokens; keep on CPU to avoid device thrash
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_seqs if s]

    def _ends_with(self, a: torch.Tensor, b: List[int]) -> bool:
        if len(b) == 0 or a.size(1) < len(b):
            return False
        tail = a[0, -len(b):].tolist()
        return tail == b

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.size(0) != 1:  # single-example stopping; extend if batching
            return False
        for b in self.stop_ids:
            if self._ends_with(input_ids, b):
                return True
        return False

# --- helpers for repetition diagnostics
def _max_token_run(ids: List[int]) -> int:
    """Longest run of identical token ids in a sequence."""
    if not ids:
        return 0
    best = cur = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i-1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def _max_repeated_ngram(ids: List[int], n: int = 3) -> int:
    """Maximum number of times any n-gram appears (overlapping)."""
    if len(ids) < n:
        return 0
    counts: Dict[tuple, int] = {}
    for i in range(len(ids)-n+1):
        ng = tuple(ids[i:i+n])
        counts[ng] = counts.get(ng, 0) + 1
    return max(counts.values()) if counts else 0

@dataclass
class GenPreset:
    do_sample: bool
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

def _preset(name: str) -> GenPreset:
    name = (name or "safe_sample").lower()
    if name == "greedy":
        return GenPreset(do_sample=False, temperature=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0)
    if name == "safe_sample":
        return GenPreset(do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
                         repetition_penalty=1.15, no_repeat_ngram_size=3)
    if name == "creative":
        return GenPreset(do_sample=True, temperature=0.9, top_p=0.95, top_k=100,
                         repetition_penalty=1.10, no_repeat_ngram_size=4)
    return _preset("safe_sample")

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
    
    @torch.no_grad()
    def generate_with_logging(self, pixel_values, bos_id, eos_id, tokenizer=None, **kwargs):
        """
        Batch-capable wrapper; returns {'per_sample': [...], 'sequences': LongTensor[B, Tmax]}.
        """
        cls, patches = self.encoder(pixel_values)
        return self.decoder.generate_with_logging(
            patches=patches,
            cls=cls,
            bos_id=bos_id,
            eos_id=eos_id,
            tokenizer=tokenizer,
            **kwargs
        )


class VisualPrefixGPT2(nn.Module):
    def __init__(self, gpt2_name="gpt2", vis_dim=384, num_prefix_tokens=8):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_name)
        self.num_prefix_tokens = num_prefix_tokens
        self.vis_to_prefix = nn.Linear(vis_dim, self.model.config.n_embd)

    def forward(self, visual_tokens, input_ids, attention_mask=None, labels=None):
        B, Tvis, Dv = visual_tokens.shape
        prefix = self.vis_to_prefix(visual_tokens)
        inputs_embeds = self.model.transformer.wte(input_ids)
        full_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        vis_mask = torch.ones((B, Tvis), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)
        if labels is not None:
            pad_labels = torch.full((B, Tvis), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([pad_labels, labels], dim=1)
        else:
            full_labels = None
        return self.model(inputs_embeds=full_embeds, attention_mask=full_mask, labels=full_labels)

    @torch.no_grad()
    def generate(self, visual_tokens, input_ids, max_new_tokens=50):
        B, Tvis, Dv = visual_tokens.shape
        prefix = self.vis_to_prefix(visual_tokens)
        inputs_embeds = self.model.transformer.wte(input_ids)
        full_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        full_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=full_embeds.device)
        gen_ids = self.model.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        return gen_ids
    
    @torch.no_grad()
    def generate_with_logging(
        self,
        visual_tokens: torch.Tensor,      # [B, Tvis, Dv]  (already in image-embedding space)
        # input_ids: torch.Tensor,        # [B, Ttxt]      (prompt tokens) <--- REMOVED
        max_new_tokens: int = 256,
        # preset / overrides
        preset: str = "safe_sample",      # "greedy" | "safe_sample" | "creative"
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        # stopping/logging
        tokenizer=None,                   # if None, uses self.tokenizer
        stop_sequences: list[str] | None = None,
        log_first_n_steps: int = 6,
        log_topk: int = 5,
        return_text: bool = True,
        min_gen_before_eos: int = 1,      # ban EOS for first N generated tokens
    ) -> dict:
        """
        Logs top-k candidates, entropy, EOS/stop triggers, and repetition stats.
        Returns:
        {
            "per_sample": [ { …logging for sample i… }, ... ],
            "sequences": LongTensor[B, Tmax]  # text-only (prompt + generated), padded with pad_token_id
        }
        """
        # MODIFICATION 1: Get device from visual_tokens
        device = visual_tokens.device
        tok = tokenizer or self.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        eos_id = tok.eos_token_id
        
        # MODIFICATION 2: Get bos_id to start generation
        bos_id = tok.bos_token_id
        assert bos_id is not None, "Tokenizer must have a bos_token_id to generate from scratch"

        def _preset(name: str):
            name = (name or "safe_sample").lower()
            if name == "greedy":
                return dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                            repetition_penalty=1.0, no_repeat_ngram_size=0)
            if name == "creative":
                return dict(do_sample=True, temperature=0.9, top_p=0.95, top_k=100,
                            repetition_penalty=1.10, no_repeat_ngram_size=4)
            return dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
                        repetition_penalty=1.15, no_repeat_ngram_size=3)

        cfg = _preset(preset)
        if do_sample is not None: cfg["do_sample"] = do_sample
        if temperature is not None: cfg["temperature"] = temperature
        if top_p is not None: cfg["top_p"] = top_p
        if top_k is not None: cfg["top_k"] = top_k
        if repetition_penalty is not None: cfg["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size is not None: cfg["no_repeat_ngram_size"] = no_repeat_ngram_size

        # helpers -------------
        def _entropy_from_logits(l):
            p = torch.softmax(l, dim=-1)
            return float(-(p * p.clamp_min(1e-12).log()).sum().item())

        def _apply_rep_penalty_(logits, prev_ids, penalty: float):
            if penalty is None or penalty == 1.0 or len(prev_ids) == 0:
                return logits
            l = logits.view(-1)
            for t in set(prev_ids):
                v = l[t]
                l[t] = v / penalty if v > 0 else v * penalty
            return logits

        def _ban_ngram_(logits, seq_ids: list[int], n: int):
            if n is None or n <= 1 or len(seq_ids) < n - 1:
                return logits
            n_1 = n - 1
            ctx = seq_ids[-n_1:] if n_1 > 0 else []
            bans = set()
            for i in range(len(seq_ids) - n_1):
                if seq_ids[i:i+n_1] == ctx:
                    bans.add(seq_ids[i+n_1])
            if bans:
                l = logits.view(-1)
                l[list(bans)] = -1e9
            return logits

        def _topk_topp_filter_(logits, top_k: int | None, top_p: float | None):
            # top-k
            if top_k is not None and top_k > 0 and top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
            # top-p
            if top_p is not None and top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sprob, sidx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sprob, dim=-1)
                mask = cum > top_p
                mask[..., 0] = False
                to_remove = torch.zeros_like(probs, dtype=torch.bool)
                to_remove.scatter_(dim=-1, index=sidx, src=mask)
                logits = torch.where(to_remove, torch.full_like(logits, -1e9), logits)
            return logits

        def _max_token_run(ids):
            if not ids: return 0
            best = cur = 1
            for i in range(1, len(ids)):
                if ids[i] == ids[i-1]:
                    cur += 1; best = max(best, cur)
                else: cur = 1
            return best

        def _max_rep_ngram(ids, n):
            if len(ids) < n: return 0
            cnt = {}
            for i in range(len(ids)-n+1):
                ng = tuple(ids[i:i+n]); cnt[ng] = cnt.get(ng, 0) + 1
            return max(cnt.values()) if cnt else 0

        # encode stop sequences once
        stop_ids = []
        if tok is not None and stop_sequences:
            for s in stop_sequences:
                stop_ids.append(tok.encode(s, add_special_tokens=False))

        def _ends_with(seq: list[int], suffix: list[int]) -> bool:
            return len(suffix) > 0 and len(seq) >= len(suffix) and seq[-len(suffix):] == suffix

        # ---------- per-sample loop ----------
        B, Tvis, Dv = visual_tokens.shape
        prefix_all = self.vis_to_prefix(visual_tokens)  # [B, Tvis, n_embd]
        results = []
        seqs = []

        for b in range(B):
            prefix = prefix_all[b:b+1]                  # [1, Tvis, n_embd]
            
            # MODIFICATION 3: Initialize seq with bos_id instead of input_ids
            prompt_seq = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            seq = prompt_seq.clone()                    # [1, 1] (starts with BOS)
            
            gen_only: list[int] = []
            step_logs = []
            hit_eos_at = None

            # build constant visual mask
            vis_mask = torch.ones((1, Tvis), dtype=torch.long, device=device)

            for step in range(max_new_tokens):
                # build embeds for current text sequence
                text_embeds = self.model.transformer.wte(seq)        # [1, Ttxt, n_embd]
                full_embeds = torch.cat([prefix, text_embeds], dim=1)  # [1, Tvis+Ttxt, n_embd]

                # attention mask (1s everywhere)
                text_mask = torch.ones_like(seq, dtype=torch.long, device=device)  # [1, Ttxt]
                full_mask = torch.cat([vis_mask, text_mask], dim=1)                # [1, Tvis+Ttxt]

                out = self.model(inputs_embeds=full_embeds, attention_mask=full_mask)
                logits = out.logits[:, -1, :].squeeze(0)             # [V]

                # safety
                if (len(gen_only) < min_gen_before_eos) and eos_id is not None:
                    logits[eos_id] = -1e9

                # temperature & anti-repetition
                logits = logits / max(1e-6, float(cfg["temperature"]))
                logits = _apply_rep_penalty_(logits, gen_only, cfg["repetition_penalty"])
                logits = _ban_ngram_(logits, seq[0].tolist(), cfg["no_repeat_ngram_size"])

                # filtered distribution (for sampling + logging)
                filt = _topk_topp_filter_(logits.clone(), cfg["top_k"], cfg["top_p"])
                probs = torch.softmax(filt, dim=-1)

                # log first N steps
                if step < log_first_n_steps:
                    kdisp = min(int(logits.numel()), max(1, min(5, log_topk)))
                    topv, topi = torch.topk(probs, kdisp)
                    step_logs.append({
                        "step": step + 1,
                        "entropy": _entropy_from_logits(filt),
                        "topk": [{"token_id": int(topi[j]), "p": float(topv[j])} for j in range(kdisp)]
                    })

                # pick next token
                if cfg["do_sample"]:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_id = int(torch.argmax(filt).item())

                seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
                gen_only.append(next_id)

                # stop by EOS
                if eos_id is not None and next_id == eos_id:
                    hit_eos_at = len(gen_only) - 1
                    break

                # stop by custom stop sequences
                if stop_ids:
                    for sfx in stop_ids:
                        if _ends_with(seq[0].tolist(), sfx):
                            hit_eos_at = len(gen_only) - 1
                            break
                    if hit_eos_at is not None:
                        break

            # repetition stats & text
            rep_stats = {
                "max_token_run": _max_token_run(gen_only),
                "max_repeat_trigram": _max_rep_ngram(gen_only, 3),
                "max_repeat_4gram": _max_rep_ngram(gen_only, 4),
            }
            text = {}
            if return_text and tok is not None:
                text["generated"] = tok.decode(gen_only)
                text["full"] = tok.decode(seq[0].tolist())

            results.append({
                "preset": preset,
                "params": cfg,
                "lengths": {
                    # MODIFICATION 4: Log length of new prompt_seq
                    "prompt_tokens": int(prompt_seq.size(1)),
                    "new_tokens": len(gen_only),
                    "total_tokens": int(seq.size(1)),
                },
                "stopping": {"hit_eos": (hit_eos_at is not None), "eos_pos": hit_eos_at,
                            "stop_sequences": stop_sequences or []},
                "repetition": rep_stats,
                "probes": step_logs,
                "ids": {"gen_only_ids": gen_only, "full_ids": seq[0].tolist()},
                "text": text,
            })
            seqs.append(seq[0])

        # pad to rectangular tensor of text ids
        Tmax = max(s.size(0) for s in seqs)
        out = torch.full((B, Tmax), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return {"per_sample": results, "sequences": out}



class DinoGPT2Captioner(nn.Module):
    def __init__(
        self,
        d_img=384,
        num_prefix_tokens=8,
        gpt2_name="gpt2",
        dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_dino=True
    ):
        super().__init__()
        self.encoder = DINOEncoder(model_id=dino_model_id, freeze=freeze_dino)
        self.decoder = VisualPrefixGPT2(gpt2_name=gpt2_name, vis_dim=d_img, num_prefix_tokens=num_prefix_tokens)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        cls, patches = self.encoder(pixel_values)
        prefix_tokens = torch.cat([cls.unsqueeze(1), patches[:, :self.decoder.num_prefix_tokens-1, :]], dim=1)
        output = self.decoder(prefix_tokens, input_ids, attention_mask=attention_mask, labels=labels)
        # Remove prefix logits so output matches input_ids length
        logits = output.logits
        if logits.size(1) > input_ids.size(1):
            logits = logits[:, -input_ids.size(1):, :]
        return logits

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, max_new_tokens=50, **generation_kwargs):
        cls, patches = self.encoder(pixel_values)
        prefix_tokens = torch.cat([cls.unsqueeze(1), patches[:, :self.decoder.num_prefix_tokens-1, :]], dim=1)
        gen_ids = self.decoder.generate(prefix_tokens, input_ids, max_new_tokens=max_new_tokens)
        return gen_ids
    
    @torch.no_grad()
    def generate_with_logging(
        self,
        pixel_values: torch.Tensor,            # [B, C, H, W]
        input_ids: torch.Tensor,               # [B, Ttxt] prompt
        max_new_tokens: int = 256,
        tokenizer=None,
        **kwargs,                               # preset/do_sample/temperature/top_p/top_k/etc.
    ):
        cls, patches = self.encoder(pixel_values)
        # build visual-prefix tokens: [CLS] + first K-1 patches
        prefix_tokens = torch.cat(
            [cls.unsqueeze(1), patches[:, :self.decoder.num_prefix_tokens-1, :]],
            dim=1
        )  # [B, Tvis, Dimg]
        return self.decoder.generate_with_logging(
            visual_tokens=prefix_tokens,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer or self.decoder.tokenizer,
            **kwargs
        )
