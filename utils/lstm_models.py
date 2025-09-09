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