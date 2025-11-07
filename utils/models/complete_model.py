import os
import torch
import torch.nn as nn
from transformers import AutoModel, GPT2Tokenizer

from utils.models.modifiedGPT2 import create_decoder

from utils.layer_mask import gaussian_layer_stack_pipeline


class DINOEncoder(nn.Module):
    def __init__(self, model_id="facebook/dinov3-vits16-pretrain-lvd1689m", freeze=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, C, H, W]
        returns patches: [B, Np, Cenc]
        """
        out = self.model(pixel_values=pixel_values)
        tokens = out.last_hidden_state  # [B, 1+Np, Cenc] (CLS + patches) for ViT-like
        # Skip a few special tokens if your backbone adds them; adjust as needed.
        patches = tokens[:, 5:, :]  # [B, Np, Cenc]
        return patches

class DinoUNet(nn.Module):
    def __init__(self, model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", freeze=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # NOTE: confirm channels of the chosen hidden state; 768 is common for small convnext/dinov3
        self.channel_adapter = nn.Conv2d(768, 512, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        if freeze:
            for m in (self.encoder, self.channel_adapter, self.decoder):
                for p in m.parameters():
                    p.requires_grad = False
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor, num_layers: int) -> torch.Tensor:
        """
        x: [B, C, H, W]; returns mask: [B, 1, H', W'] (your upsampling stack defines H',W')
        """
        enc_feats = self.encoder(x, output_hidden_states=True, return_dict=True)
        # take the last 4D feature map from hidden_states
        feats = next(h for h in reversed(enc_feats.hidden_states) if isinstance(h, torch.Tensor) and h.ndim == 4)
        feats = self.channel_adapter(feats)
        pred = self.decoder(feats)                    # (B,1,h,w)
        _, _, segmentation_mask = gaussian_layer_stack_pipeline(pred, n_layers = num_layers)
        return segmentation_mask    # [B, num_layers, h, w]


class LinearProjection(nn.Module):
    def __init__(self, input_dim=384, output_dim=768, freeze=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        if freeze:
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Np, input_dim] -> [B, Np, output_dim]
        return self.proj(x)


class CustomModel(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        ENCODER_MODEL_PATH: str | None = "dino_encoder.pth",
        SEGMENTER_MODEL_PATH: str | None = "dino_segmenter.pth",
        DECODER_MODEL_PATH: str | None = "dino_decoder.pth",
        LINEAR_PROJECTION_PATH: str | None = "linear_projection.pth",
        freeze_encoder: bool = True,
        freeze_segmenter: bool = True,
        freeze_linear_projection: bool = False,
        freeze_decoder: bool = False,
        attention_implementation: str = "sdpa",
    ):
        super().__init__()
        self.device = torch.device(device)

        # Encoder
        self.encoder = DINOEncoder()
        if ENCODER_MODEL_PATH and os.path.exists(ENCODER_MODEL_PATH):
            self.encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location="cpu"), strict=False)
            print("Loaded encoder weights from", ENCODER_MODEL_PATH)
        if freeze_encoder:
            self.encoder.eval()

        # Segmenter
        self.segmenter = DinoUNet()
        if SEGMENTER_MODEL_PATH and os.path.exists(SEGMENTER_MODEL_PATH):
            self.segmenter.load_state_dict(torch.load(SEGMENTER_MODEL_PATH, map_location="cpu"), strict=False)
            print("Loaded segmenter weights from", SEGMENTER_MODEL_PATH)
        if freeze_segmenter:
            self.segmenter.eval()

        # Decoder (modified GPT-2)
        self.decoder = create_decoder(attention=attention_implementation)  # must expose .config.hidden_size & .config.num_hidden_layers
        if DECODER_MODEL_PATH and os.path.exists(DECODER_MODEL_PATH):
            self.decoder.load_state_dict(torch.load(DECODER_MODEL_PATH, map_location="cpu"), strict=False)
            print("Loaded decoder weights from", DECODER_MODEL_PATH)
        if freeze_decoder:
            self.decoder.eval()

        # Linear projection: DINO hidden -> GPT2 hidden
        enc_h = self.encoder.model.config.hidden_size
        dec_h = self.decoder.config.hidden_size
        self.linear_projection = LinearProjection(input_dim=enc_h, output_dim=dec_h)
        if LINEAR_PROJECTION_PATH and os.path.exists(LINEAR_PROJECTION_PATH):
            self.linear_projection.load_state_dict(torch.load(LINEAR_PROJECTION_PATH, map_location="cpu"), strict=False)
            print("Loaded linear projection weights from", LINEAR_PROJECTION_PATH)
        if freeze_linear_projection:
            self.linear_projection.eval()

        # Tokenizer (pad token for GPT-2)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id  # ✅ use ID, not string

        self.num_layers = self.decoder.config.num_hidden_layers

        # move everything once
        self.to(self.device)

    def forward(self, pixel_values: torch.Tensor, tgt_ids: torch.Tensor | None = None, **kwargs) -> dict:
        """
        pixel_values: [B,C,H,W], float
        tgt_ids: [B,T], long (token IDs), padded with pad_token_id if any padding is present
        """
        pixel_values = pixel_values.to(self.device, non_blocking=True)

        # Visual path
        patches = self.encoder(pixel_values)                           # [B,Np,Cenc]
        projected_patches = self.linear_projection(patches)            # [B,Np,n_embd]

        # Segmentation path per layer
        segmented_layers = self.segmenter(pixel_values, self.num_layers) # [B,n_layers,H,W] (per current decoder)

        # Text path (optional teacher-forced training)
        labels = None
        if tgt_ids is not None:
            if tgt_ids.dtype != torch.long:
                tgt_ids = tgt_ids.long()
            tgt_ids = tgt_ids.to(self.device, non_blocking=True)       # [B,T]
            text_embeds = self.decoder.transformer.wte(tgt_ids)        # [B,T,n_embd]
            inputs_embeds = torch.cat([projected_patches, text_embeds], dim=1)  # [B,Np+T,n_embd]

            # Labels: ignore prefix tokens (vision) and PADs in text
            B, Np, _ = projected_patches.shape
            labels_prefix = torch.full((B, Np), -100, device=self.device, dtype=torch.long)
            text_labels = tgt_ids.clone()
            text_labels[text_labels == self.pad_token_id] = -100       # ✅ compare to ID
            labels = torch.cat([labels_prefix, text_labels], dim=1)    # [B,Np+T]
        else:
            inputs_embeds = projected_patches

        # Decoder forward
        out = self.decoder(inputs_embeds=inputs_embeds, segmentation_mask=segmented_layers, labels=labels, **kwargs)
        return out
    
    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 100,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        pixel_values: [B,C,H,W], float
        returns generated_ids: [B, T]
        """
        pixel_values = pixel_values.to(self.device, non_blocking=True)

        # Visual path
        patches = self.encoder(pixel_values)                           # [B,Np,Cenc]
        projected_patches = self.linear_projection(patches)            # [B,Np,n_embd]

        # Segmentation path per layer
        segmented_layers = self.segmenter(pixel_values, self.num_layers) # [B,n_layers,H,W] (per current decoder)

        # Generate
        output = self.decoder.generate(
            inputs_embeds=projected_patches,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
            segmentation_mask=segmented_layers,
            prefix_allowed_length=0,
            plot_attention_mask=False,
            plot_attention_mask_layer=[],
            plot_attention_map=False,
            plot_attention_map_layer=[],
            plot_attention_map_generation=0,
            output_attentions=output_attentions,
            return_dict_in_generate=True,
        )
        # Remove prefix tokens (vision)
        generated_ids = output.sequences#[:, projected_patches.shape[1]:]   # [B,T]
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_ids, generated_text, output.attentions if output_attentions else None

def create_complete_model(device: str = "cuda", **kwargs) -> CustomModel:
    model = CustomModel(device=device, **kwargs)
    return model

def save_complete_model(model: CustomModel, save_path: str, device: str = "cuda") -> None:
    # Ensure folder exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Save on CPU to keep checkpoint portable
    orig_device = next(model.parameters()).device
    model.to("cpu")
    torch.save(model.state_dict(), save_path)
    print(f"Saved complete model weights to {save_path}")

    # Restore model device
    model.to(device if isinstance(device, str) else orig_device)

def save_checkpoint(model: CustomModel, optimizer: torch.optim.Optimizer, save_path: str) -> None:
    # Ensure folder exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")

def load_complete_model(model: CustomModel, load_path: str, device: str = "cpu", strict: bool = True) -> CustomModel:
    if not os.path.exists(load_path):
        print(f"No weights found at {load_path}")
        model.to(device)
        return model

    # Load to CPU first, then move to target device
    state = torch.load(load_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict:
        if missing:
            print(f"[load warning] Missing keys: {missing}")
        if unexpected:
            print(f"[load warning] Unexpected keys: {unexpected}")

    model.to(device)
    print(f"Loaded complete model weights from {load_path}")
    return model

def load_checkpoint(model: CustomModel, optimizer: torch.optim.Optimizer, load_path: str, device: str = "cpu") -> tuple[CustomModel, torch.optim.Optimizer]:
    if not os.path.exists(load_path):
        print(f"No checkpoint found at {load_path}")
        model.to(device)
        return model, optimizer

    # Load to CPU first, then move to target device
    checkpoint = torch.load(load_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.to(device)
    print(f"Loaded checkpoint from {load_path}")
    return model, optimizer