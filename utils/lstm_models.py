import torch
import torch.nn as nn

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
    
    @torch.no_grad()
    def generate_with_logging(
        self,
        patches: torch.Tensor,  # [B, Np, Dimg]
        cls: torch.Tensor,      # [B, Dimg]
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 128,
        # presets / overrides
        preset: str = "safe_sample",     # "greedy" | "safe_sample" | "creative"
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        # logging / stopping
        tokenizer=None,
        stop_sequences: list[str] | None = None,
        log_first_n_steps: int = 6,
        log_topk: int = 5,
        return_text: bool = True,
    ) -> dict:
        device = cls.device
        B = cls.size(0)

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

        # helpers
        def _entropy_from_logits(l): 
            p = torch.softmax(l, dim=-1); 
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
            if top_k is not None and top_k > 0 and top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
            if top_p is not None and top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sprob, sidx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sprob, dim=-1)
                mask = cum > top_p; mask[..., 0] = False
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

        stop_ids = []
        if tokenizer is not None and stop_sequences:
            for s in stop_sequences:
                stop_ids.append(tokenizer.encode(s, add_special_tokens=False))

        def _ends_with(seq: list[int], suffix: list[int]) -> bool:
            return len(suffix) > 0 and len(seq) >= len(suffix) and seq[-len(suffix):] == suffix

        # prepare visual memory and init states
        keys = self.kv_proj(patches)        # [B, N, d_h]
        h = self.init_h(cls)                # [B, d_h]
        c = self.init_c(cls)                # [B, d_h]

        seqs, results = [], []
        for b in range(B):
            seq = [bos_id]
            finished = False
            x = self.emb(torch.tensor([bos_id], device=device)).squeeze(0)   # [d_h]
            hb, cb = h[b:b+1], c[b:b+1]                                      # keep batch dims for attn

            step_logs = []
            for step in range(max_new_tokens):
                ctx, _ = self.attn(hb, keys[b:b+1])          # [1, d_h]
                lstm_in = torch.cat([x.unsqueeze(0), ctx], dim=-1)  # [1, 2*d_h]
                hb, cb = self.lstm(lstm_in, (hb, cb))
                hb = self.ln(self.dropout(hb))               # [1, d_h]
                logits = self.out(hb).squeeze(0)             # [V]

                # safety & temp
                logits[bos_id] = -1e9
                if len(seq) < 2: logits[eos_id] = -1e9
                logits = logits / max(1e-6, float(cfg["temperature"]))

                # repetition controls
                logits = _apply_rep_penalty_(logits, seq[1:], cfg["repetition_penalty"])  # exclude BOS
                logits = _ban_ngram_(logits, seq, cfg["no_repeat_ngram_size"])

                # filtering
                filt = _topk_topp_filter_(logits.clone(), cfg["top_k"], cfg["top_p"])
                probs = torch.softmax(filt, dim=-1)

                # logging
                if step < log_first_n_steps:
                    kdisp = min(int(logits.numel()), max(1, min(5, log_topk)))
                    topv, topi = torch.topk(probs, kdisp)
                    step_logs.append({
                        "step": step+1,
                        "entropy": _entropy_from_logits(filt),
                        "topk": [{"token_id": int(topi[j]), "p": float(topv[j])} for j in range(kdisp)]
                    })

                # pick next
                if cfg["do_sample"]:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_id = int(torch.argmax(filt).item())

                seq.append(next_id)
                x = self.emb(torch.tensor(next_id, device=device))

                if next_id == eos_id:
                    finished = True
                    break
                if stop_ids:
                    for sfx in stop_ids:
                        if _ends_with(seq, sfx):
                            finished = True
                            break
                if finished: break

            # repetition stats
            gen_only = seq[1:]  # drop BOS
            rep = {
                "max_token_run": _max_token_run(gen_only),
                "max_repeat_trigram": _max_rep_ngram(gen_only, 3),
                "max_repeat_4gram": _max_rep_ngram(gen_only, 4),
            }
            text = {}
            if return_text and tokenizer is not None:
                text["generated"] = tokenizer.decode(gen_only)
                text["full"] = tokenizer.decode(seq)

            results.append({
                "preset": preset,
                "params": cfg,
                "lengths": {"prompt_tokens": 1, "new_tokens": len(gen_only), "total_tokens": len(seq)},
                "stopping": {"hit_eos": any(t==eos_id for t in gen_only),
                            "eos_pos": (gen_only.index(eos_id) if eos_id in gen_only else None),
                            "stop_sequences": stop_sequences or []},
                "repetition": rep,
                "probes": step_logs,
                "ids": {"gen_only_ids": gen_only, "full_ids": seq},
                "text": text,
            })
            seqs.append(torch.tensor(seq, device=device, dtype=torch.long))

        # pad to tensor
        Tmax = max(s.numel() for s in seqs)
        out = torch.full((B, Tmax), self.pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            out[i, :s.numel()] = s
        return {"per_sample": results, "sequences": out}


class BiLSTMAttnDecoder(nn.Module):
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
        self.ln = nn.LayerNorm(d_h * 2)
        self.bilstm = nn.LSTM(d_h * 2, d_h, batch_first=True, bidirectional=True)
        self.attn = BahdanauAttention(d_h)
        self.out = nn.Linear(d_h * 2, vocab_size)

    def forward(self, patches, cls, tgt_ids):
        B, T = tgt_ids.size()
        device = tgt_ids.device
        keys = self.kv_proj(patches)
        h = self.init_h(cls)
        c = self.init_c(cls)
        x = self.emb(tgt_ids)  # (B, T, d_h)
        ctx_seq = []
        for t in range(T):
            ctx, _ = self.attn(h, keys)
            ctx_seq.append(ctx)
        ctx_seq = torch.stack(ctx_seq, dim=1)  # (B, T, d_h)
        lstm_in = torch.cat([x, ctx_seq], dim=-1)  # (B, T, d_h*2)
        output, _ = self.bilstm(lstm_in)
        output = self.ln(self.dropout(output))
        logits = self.out(output)
        return logits

    @torch.no_grad()
    def generate(self, patches, cls, bos_id, eos_id, max_new_tokens=30, top_p=0.9, temperature=1.0, greedy=False):
        B = cls.size(0)
        device = cls.device
        keys = self.kv_proj(patches)
        h = self.init_h(cls)
        c = self.init_c(cls)
        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            x = self.emb(seq)  # (B, T, d_h)
            ctx_seq = []
            for t in range(seq.size(1)):
                ctx, _ = self.attn(h, keys)
                ctx_seq.append(ctx)
            ctx_seq = torch.stack(ctx_seq, dim=1)  # (B, T, d_h)
            lstm_in = torch.cat([x, ctx_seq], dim=-1)  # (B, T, d_h*2)
            output, _ = self.bilstm(lstm_in)
            output = self.ln(output[:, -1, :])
            logits = self.out(output) / max(1e-6, temperature)
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
            finished = finished | (next_tokens == eos_id)
            if finished.all():
                break
        return seq
    
    @torch.no_grad()
    def generate_with_logging(
        self,
        patches: torch.Tensor, cls: torch.Tensor,
        bos_id: int, eos_id: int,
        max_new_tokens: int = 128,
        preset: str = "safe_sample",
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        tokenizer=None,
        stop_sequences: list[str] | None = None,
        log_first_n_steps: int = 6,
        log_topk: int = 5,
        return_text: bool = True,
    ) -> dict:
        device = cls.device
        B = cls.size(0)

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

        def _entropy_from_logits(l): 
            p = torch.softmax(l, dim=-1); 
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
            if top_k is not None and top_k > 0 and top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
            if top_p is not None and top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sprob, sidx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sprob, dim=-1)
                mask = cum > top_p; mask[..., 0] = False
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

        stop_ids = []
        if tokenizer is not None and stop_sequences:
            for s in stop_sequences:
                stop_ids.append(tokenizer.encode(s, add_special_tokens=False))

        def _ends_with(seq: list[int], suffix: list[int]) -> bool:
            return len(suffix) > 0 and len(seq) >= len(suffix) and seq[-len(suffix):] == suffix

        keys = self.kv_proj(patches)
        h0 = self.init_h(cls)  # used only for attention context creation (like your forward)
        # c0 not needed here because BiLSTM forward ignores it (we re-run full seq each step)

        seqs, results = [], []
        for b in range(B):
            seq = [bos_id]
            step_logs = []
            for step in range(max_new_tokens):
                x = self.emb(torch.tensor(seq, device=device)).unsqueeze(0)  # [1, T, d_h]
                # build a context per position (your forward repeats attn over fixed h0)
                ctx_seq = []
                for _t in range(len(seq)):
                    ctx, _ = self.attn(h0[b:b+1], keys[b:b+1])
                    ctx_seq.append(ctx.squeeze(0))
                ctx_seq = torch.stack(ctx_seq, dim=0).unsqueeze(0)           # [1, T, d_h]
                lstm_in = torch.cat([x, ctx_seq], dim=-1)                    # [1, T, 2*d_h]
                output, _ = self.bilstm(lstm_in)                             # [1, T, 2*d_h]
                logits = self.out(self.ln(self.dropout(output)))[:, -1, :].squeeze(0)  # [V]

                logits[bos_id] = -1e9
                if len(seq) < 2: logits[eos_id] = -1e9
                logits = logits / max(1e-6, float(cfg["temperature"]))
                logits = _apply_rep_penalty_(logits, seq[1:], cfg["repetition_penalty"])
                logits = _ban_ngram_(logits, seq, cfg["no_repeat_ngram_size"])

                filt = _topk_topp_filter_(logits.clone(), cfg["top_k"], cfg["top_p"])
                probs = torch.softmax(filt, dim=-1)

                if step < log_first_n_steps:
                    kdisp = min(int(logits.numel()), max(1, min(5, log_topk)))
                    topv, topi = torch.topk(probs, kdisp)
                    step_logs.append({
                        "step": step+1,
                        "entropy": _entropy_from_logits(filt),
                        "topk": [{"token_id": int(topi[j]), "p": float(topv[j])} for j in range(kdisp)]
                    })

                if cfg["do_sample"]:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_id = int(torch.argmax(filt).item())
                seq.append(next_id)

                if next_id == eos_id:
                    break
                if stop_ids:
                    stop_hit = any(_ends_with(seq, sfx) for sfx in stop_ids)
                    if stop_hit: break

            gen_only = seq[1:]
            rep = {
                "max_token_run": _max_token_run(gen_only),
                "max_repeat_trigram": _max_rep_ngram(gen_only, 3),
                "max_repeat_4gram": _max_rep_ngram(gen_only, 4),
            }
            text = {}
            if return_text and tokenizer is not None:
                text["generated"] = tokenizer.decode(gen_only)
                text["full"] = tokenizer.decode(seq)

            results.append({
                "preset": preset,
                "params": cfg,
                "lengths": {"prompt_tokens": 1, "new_tokens": len(gen_only), "total_tokens": len(seq)},
                "stopping": {"hit_eos": any(t==eos_id for t in gen_only),
                            "eos_pos": (gen_only.index(eos_id) if eos_id in gen_only else None),
                            "stop_sequences": stop_sequences or []},
                "repetition": rep,
                "probes": step_logs,
                "ids": {"gen_only_ids": gen_only, "full_ids": seq},
                "text": text,
            })
            seqs.append(torch.tensor(seq, device=device, dtype=torch.long))

        Tmax = max(s.numel() for s in seqs)
        out = torch.full((B, Tmax), self.pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            out[i, :s.numel()] = s
        return {"per_sample": results, "sequences": out}


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
    
    @torch.no_grad()
    def generate_with_logging(self, pixel_values, bos_id, eos_id, tokenizer=None, **kwargs):
        cls, patches = self.encoder(pixel_values)
        return self.decoder.generate_with_logging(
            patches=patches, cls=cls,
            bos_id=bos_id, eos_id=eos_id,
            tokenizer=tokenizer,
            **kwargs
        )

class DinoBiLSTMAttnCaptioner(nn.Module):
    def __init__(self, vocab_size, d_img, d_h=512, pad_id=0, dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m", freeze_dino=True):
        super().__init__()
        self.encoder = DINOEncoder(dino_model_id, freeze=freeze_dino)
        self.decoder = BiLSTMAttnDecoder(vocab_size=vocab_size, d_img=d_img, d_h=d_h, pad_id=pad_id)

    def forward(self, pixel_values, tgt_ids):
        cls, patches = self.encoder(pixel_values)
        logits = self.decoder(patches, cls, tgt_ids)
        return logits

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
    
    @torch.no_grad()
    def generate_with_logging(self, pixel_values, bos_id, eos_id, tokenizer=None, **kwargs):
        cls, patches = self.encoder(pixel_values)
        return self.decoder.generate_with_logging(
            patches=patches, cls=cls,
            bos_id=bos_id, eos_id=eos_id,
            tokenizer=tokenizer,
            **kwargs
        )
