import os
import torch
import torch.nn as nn
# from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModel, GPT2Tokenizer

from utils.models.modifiedGPT2 import create_decoder

from utils.layer_mask import gaussian_layer_stack_pipeline


class DINOEncoder(nn.Module):
    def __init__(self, model_id="facebook/dinov3-vits16-pretrain-lvd1689m", freeze=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        self.model.gradient_checkpointing_enable()
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

class DinoUNetLung(nn.Module):
    def __init__(self, model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", freeze=True, mask_implementation="default"):
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
        self.mask_implementation = mask_implementation
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]; returns mask: [B, 1, H', W'] (your upsampling stack defines H',W')
        """
        enc_feats = self.encoder(x, output_hidden_states=True, return_dict=True)
        # take the last 4D feature map from hidden_states
        feats = next(h for h in reversed(enc_feats.hidden_states) if isinstance(h, torch.Tensor) and h.ndim == 4)
        feats = self.channel_adapter(feats)
        pred = self.decoder(feats)                    # (B,1,h,w)
        lung = torch.sigmoid(pred) > 0.5
        return lung

class DinoUNetHeart(nn.Module):
    def __init__(self, model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", num_classes=3, freeze=True):
        super().__init__()
        print("🧠 Cargando encoder DINOv3...")
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.adapter = nn.Conv2d(768, 512, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 2, 2), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  2, 2), nn.ReLU(True),
            nn.Conv2d(64, num_classes, 1)
        )
        if freeze:
            for m in (self.encoder, self.adapter, self.decoder):
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        enc = self.encoder(x, output_hidden_states=True, return_dict=True)
        feat = next(h for h in reversed(enc.hidden_states) if isinstance(h, torch.Tensor) and h.ndim == 4)
        feat = self.adapter(feat)
        logits = self.decoder(feat)
        pred = torch.argmax(logits, 1)
        heart = (pred == 2).unsqueeze(1)
        return heart

class DinoUNet(nn.Module):
    def __init__(self, freeze=True, mask_implementation="default"):
        super().__init__()
        self.heart_model = DinoUNetHeart(freeze=freeze)
        self.lung_model = DinoUNetLung(freeze=freeze, mask_implementation=mask_implementation)
        self.mask_implementation = mask_implementation

    @torch.no_grad()
    def forward(self, x: torch.Tensor, n_layers: int) -> torch.Tensor:
        """
        x: [B, C, H, W]; returns stacked layers: [B, n_layers, 32, 32]
        """
        heart_mask = self.heart_model(x)  # (B,1,H,W)
        lung_mask = self.lung_model(x)    # (B,1,H,W)
        combined_mask = heart_mask | lung_mask  # (B,1,H,W)
        stacked_layers, _, _ = gaussian_layer_stack_pipeline(
            combined_mask.float(),  # (B,1,H,W)
            n_layers=n_layers,
            mask_implementation=self.mask_implementation
        )  # (B,n_layers,32,32)
        return stacked_layers

class LinearProjection(nn.Module):
    def __init__(self, input_dim=384, output_dim=768, freeze=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        # hidden_dim = (input_dim + output_dim) // 2
        # # create a MPL-4 adapter
        # self.proj = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, output_dim),
        # )
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
        SEGMENTER_MODEL_PATH_LUNG: str | None = "dino_segmenter.pth",
        SEGMENTER_MODEL_PATH_HEART: str | None = "dino_segmenter_heart.pth",
        DECODER_MODEL_PATH: str | None = "dino_decoder.pth",
        LINEAR_PROJECTION_PATH: str | None = "linear_projection.pth",
        freeze_encoder: bool = True,
        freeze_segmenter: bool = True,
        freeze_linear_projection: bool = False,
        freeze_decoder: bool = False,
        attention_implementation: str = "sdpa",
        use_segmentation_mask: bool = True,
        mask_implementation: str = "default",
    ):
        super().__init__()
        self.use_segmentation_mask = use_segmentation_mask
        self.device = torch.device(device)

        # Encoder
        self.encoder = DINOEncoder(freeze=freeze_encoder)
        if ENCODER_MODEL_PATH and os.path.exists(ENCODER_MODEL_PATH):
            self.encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location="cpu"), strict=False)
            print("Loaded encoder weights from", ENCODER_MODEL_PATH)
        if freeze_encoder:
            self.encoder.eval()

        # Segmenter
        self.segmenter = DinoUNet(freeze=freeze_segmenter, mask_implementation=mask_implementation)
        if SEGMENTER_MODEL_PATH_HEART and os.path.exists(SEGMENTER_MODEL_PATH_HEART):
            self.segmenter.heart_model.load_state_dict(torch.load(SEGMENTER_MODEL_PATH_HEART, map_location="cpu"), strict=False)
            print("Loaded segmenter weights from", SEGMENTER_MODEL_PATH_HEART)
        if SEGMENTER_MODEL_PATH_LUNG and os.path.exists(SEGMENTER_MODEL_PATH_LUNG):
            self.segmenter.lung_model.load_state_dict(torch.load(SEGMENTER_MODEL_PATH_LUNG, map_location="cpu"), strict=False)
            print("Loaded segmenter weights from", SEGMENTER_MODEL_PATH_LUNG)
        if freeze_segmenter:
            self.segmenter.eval()

        # Decoder (modified GPT-2)
        self.decoder = create_decoder(attention=attention_implementation)  # must expose .config.hidden_size & .config.num_hidden_layers
        if DECODER_MODEL_PATH and os.path.exists(DECODER_MODEL_PATH):
            self.decoder.load_state_dict(torch.load(DECODER_MODEL_PATH, map_location="cpu"), strict=False)
            print("Loaded decoder weights from", DECODER_MODEL_PATH)
        if hasattr(self.decoder.config, 'use_cache'):
            print("Set use_cache=False for training.")
            self.decoder.config.use_cache = False
        if freeze_decoder:
            self.decoder.eval()

                # # ----------------------------------------------------
                # # 2. Decoder (GPT-2)
                # # ----------------------------------------------------
                # self.decoder = create_decoder(attention=attention_implementation)
                # if DECODER_MODEL_PATH and os.path.exists(DECODER_MODEL_PATH):
                #     self.decoder.load_state_dict(torch.load(DECODER_MODEL_PATH, map_location="cpu"), strict=False)
                #     print("Loaded decoder weights from", DECODER_MODEL_PATH)

                # # ====================================================
                # # ⚡️ APPLICATION OF LoRA ⚡️
                # # ====================================================
                # use_lora = True
                # lora_rank = 8
                # lora_alpha = 16
                # lora_dropout = 0.1
                # lora_target_modules = ["c_attn"]#, "q_attn", "k_attn", "v_attn", "o_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
                # if use_lora and not freeze_decoder:
                #     print("✨ Applying Low-Rank Adaptation (LoRA).")
                #     # 1. Create LoRA configuration
                #     lora_config = LoraConfig(
                #         r=lora_rank,
                #         lora_alpha=lora_alpha,
                #         lora_dropout=lora_dropout,
                #         bias="none",
                #         task_type="CAUSAL_LM",
                #         target_modules=lora_target_modules,
                #     )
                #     # 2. Patch the model
                #     # This freezes the original decoder weights
                #     self.decoder = get_peft_model(self.decoder, lora_config)
                    
                #     # Optional: Print trainable parameters of the PeftModel (LoRA only)
                #     def print_trainable_parameters(model):
                #         trainable_params = 0
                #         all_param = 0
                #         for _, param in model.named_parameters():
                #             all_param += param.numel()
                #             if param.requires_grad:
                #                 trainable_params += param.numel()
                #         print(
                #             f"   Trainable parameters (LoRA): {trainable_params} || Total: {all_param} || Percentage: {100 * trainable_params / all_param:.4f}%"
                #         )
                #     print_trainable_parameters(self.decoder)
                    
                #     # Disable use_cache if it's a GPT-2 model (required for training)
                #     if hasattr(self.decoder.config, 'use_cache'):
                #         print("Set use_cache=False for training.")
                #         self.decoder.config.use_cache = False
                
                # # If LoRA is used, freezing is handled implicitly by PEFT.
                # # If LoRA is NOT used, but freezing was requested, we handle it like this:
                # elif freeze_decoder:
                #     self.decoder.eval()
                #     for param in self.decoder.parameters():
                #         param.requires_grad = False

        # Linear projection: DINO hidden -> GPT2 hidden
        enc_h = self.encoder.model.config.hidden_size
        dec_h = self.decoder.config.hidden_size
        self.linear_projection = LinearProjection(input_dim=enc_h, output_dim=dec_h, freeze=freeze_linear_projection)
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
        out = self.decoder(inputs_embeds=inputs_embeds, segmentation_mask=segmented_layers if self.use_segmentation_mask else None, labels=labels, **kwargs)
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
        if hasattr(self.decoder.config, 'use_cache'):
            self.decoder.config.use_cache = True
            print("Set use_cache=True for generation.")
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
            segmentation_mask=segmented_layers if self.use_segmentation_mask else None,
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