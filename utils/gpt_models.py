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
    def generate(self, patches, cls, bos_id, eos_id, max_new_tokens=50, top_p=0.9, temperature=1.0, greedy=False):
        B = cls.size(0)
        device = cls.device
        prefix = self._build_visual_prefix(patches, cls)
        P = prefix.size(1)
        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            T = seq.size(1)
            tok = self.tok_emb(seq)
            total = P + T
            self._ensure_pos_emb(total, device)
            pos_idx = torch.arange(P + T, device=device).unsqueeze(0).expand(B, P + T)
            pos = self.pos_emb(pos_idx)
            x = torch.cat([prefix, tok], dim=1) + pos
            causal = torch.tril(torch.ones(P + T, P + T, device=device, dtype=torch.bool))
            pad_mask_text = (seq == self.pad_id)
            key_padding_mask = torch.cat([torch.zeros(B, P, device=device, dtype=torch.bool), pad_mask_text], dim=1)
            for blk in self.blocks:
                x = blk(x, attn_mask=causal, key_padding_mask=key_padding_mask)
            x = self.ln_f(x)
            logits = self.lm_head(x[:, -1, :]) / max(1e-6, temperature)
            logits[:, bos_id] = -1e9
            if seq.size(1) < 2:
                logits[:, eos_id] = -1e9
            probs = torch.softmax(logits, dim=-1)
            if greedy:
                next_tok = probs.argmax(dim=-1)
            else:
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