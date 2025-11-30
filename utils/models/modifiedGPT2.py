from typing import Optional, Union
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Config
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.utils import (
    logging,
)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, eager_attention_forward
from torch import nn
from typing import Callable
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)

class GPT2AttentionModified(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.config = config
        max_positions = 2048
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_value.layers[self.layer_idx].keys
                value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            if getattr(self.config, "prefix_allowed_length", None) is not None:
                temp = self
                temp.is_cross_attention = True
            attn_output, attn_weights = attention_interface(
                self if getattr(self.config, "prefix_allowed_length", None) is None else temp,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal if getattr(self.config, "is_prefix", None) is None else False,
                **kwargs,
            )
            if getattr(self.config, "plot_attention_map", False) and self.layer_idx in getattr(self.config, "plot_attention_map_layer", []):
                # pick batch=0, head=0
                attn_bh = attn_weights[0, 0]                 # [L,S]
                L, S = attn_bh.shape
                if L > 1:
                    if getattr(self.config, "plot_attention_map_generation", 0) == 0:
                        print(f"Plotting attention map for inputs on layer {self.layer_idx}")
                        # full 2D heatmap
                        data = attn_bh.detach().float().cpu().numpy()  # [L,S]
                        plt.figure(figsize=(6,5))
                        plt.imshow(data, aspect="auto", cmap="hot", vmin=0, vmax=0.01)
                        plt.colorbar()
                        plt.xlabel("Keys (S)")
                        plt.ylabel("Queries (L)")
                        plt.title(f"Attention map (B0,H0)  L={L}, S={S}")
                        plt.show()
                else:
                    if getattr(self.config, "plot_attention_map_generation", 0) == S:
                        print(f"Plotting attention row map for token {S} generation on layer {self.layer_idx}")
                        # attn_bh expected shape: [..., S] for the selected (B0, H0) row
                        row = attn_bh[0].detach().float().cpu().numpy()  # -> np.ndarray shape [S]
                        n = row.shape[0]

                        # ----- First 1024 as 32x32 -----
                        head_1024 = row[:min(1024, n)]
                        grid = head_1024.reshape(32, 32)

                        plt.figure(figsize=(6, 5))
                        plt.imshow(grid, aspect="auto", cmap="hot", vmin=0, vmax=0.01)
                        plt.yticks([])
                        plt.colorbar()
                        plt.xlabel("Keys (S) [indices 0..1023]")
                        plt.title(f"Attention row (B0,H0)  L={self.layer_idx}, S={S} — first 1024")
                        plt.tight_layout()
                        plt.show()

                        # ----- Tail (>=1024) as a single-row heatmap -----
                        tail = row[1024:]
                        if tail.size > 0:
                            plt.figure(figsize=(10, 1.2))
                            # one-row heatmap
                            plt.imshow(tail[None, :], aspect="auto", cmap="hot", vmin=0, vmax=0.01)
                            plt.yticks([])
                            plt.colorbar()
                            plt.xlabel(f"Keys (S) [indices 1024..{n-1}]")
                            plt.title(f"Attention row tail (B0,H0)  L={self.layer_idx}, S={S}")
                            plt.tight_layout()
                            plt.show()

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights

class GPT2BlockModified(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config=config)
        self.attn = GPT2AttentionModified(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], Optional[tuple[torch.Tensor, tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, self_attn_weights = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output, cross_attn_weights = self.crossattention(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # residual connection
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)

        return outputs


class GPT2ModelModified(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config_causal = config
        self.config_causal._attn_implementation = "eager"  # Ensure causal mask creation uses eager implementation
        # TEMPORARY: override the transformer blocks to pass segmentation masks
        self.h = nn.ModuleList([GPT2BlockModified(config, layer_idx=i) for i in range(config.num_hidden_layers)])
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[tuple[tuple[torch.Tensor]], Cache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        segmentation_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            elif isinstance(past_key_values, tuple):
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                    "You should pass an instance of `Cache` instead, e.g. "
                    "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config_causal,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if segmentation_mask is not None and causal_mask is not None:
                # Make a safe copy of the causal mask and ensure its spatial
                # dimensions match the sequence length that the attention
                # functions expect. This prevents off-by-one shape errors
                # when using eager attention (torch.where requires same sizes).
                causal_mask_modified = causal_mask.clone()
                if causal_mask_modified.dtype == torch.bool:
                    min_dtype = torch.finfo(segmentation_mask.dtype).min
                    # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
                    causal_mask_modified = torch.where(causal_mask_modified, torch.tensor(0.0, device=causal_mask_modified.device, dtype=causal_mask_modified.dtype), min_dtype)
                if getattr(self.config, "prefix_allowed_length", None) is not None:
                    causal_mask_modified[:, :, :, :self.config.prefix_allowed_length].zero_()

                # Use the input sequence length to crop the causal mask if needed
                seq_len = input_shape[-1]
                if causal_mask_modified.shape[2] != seq_len or causal_mask_modified.shape[3] != seq_len:
                    causal_mask_modified = causal_mask_modified[:, :, :seq_len, :seq_len]

                # Clip segmentation mask to fit into causal_mask_modified before adding.
                _, _, M, N = segmentation_mask.shape
                M = min(M, causal_mask_modified.shape[2])
                N = min(N, causal_mask_modified.shape[3])
                try:
                    causal_mask_modified[:, :, :M, :N] += segmentation_mask[:, i, :M, :N].unsqueeze(1)
                except Exception as e:
                    print(f"Error adding segmentation mask at block {i} with shapes causal_mask_modified {causal_mask_modified[:, :, :M, :N].shape} and segmentation_mask {segmentation_mask[:, i, :M, :N].unsqueeze(1).shape}: {e}")
                    # print the datatypes and devices
                    print(f"causal_mask_modified dtype: {causal_mask_modified.dtype}, device: {causal_mask_modified.device}")
                    print(f"segmentation_mask dtype: {segmentation_mask.dtype}, device: {segmentation_mask.device}")
            if getattr(self.config, "plot_attention_mask", False) and i in getattr(self.config, "plot_attention_mask_layer", [0]):
                if segmentation_mask is not None and causal_mask is not None:
                    print(f"Block {i}: segmentation mask added to causal mask.")
                    plt.imshow(causal_mask_modified[0,0].detach().cpu(), aspect='auto', cmap='hot', vmin=-1, vmax=1)
                    plt.colorbar()
                    plt.title(f"Causal Mask with Segmentation (Block {i})")
                    plt.show()
                else:
                    print(f"Block {i}: no segmentation mask applied.")
                    plt.imshow(causal_mask[0,0].detach().cpu(), aspect='auto', cmap='hot', vmin=-1, vmax=1)
                    plt.colorbar()
                    plt.title(f"Causal Mask (Block {i})")
                    plt.show()


            outputs = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask_modified if segmentation_mask is not None and causal_mask is not None else causal_mask,
                head_mask[i],
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class GPT2LMHeadModelModified(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # replace the base transformer with our modified transformer implementation
        self.transformer = GPT2ModelModified(config)
        self.post_init() 
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        segmentation_mask: Optional[torch.FloatTensor] = None,
        prefix_allowed_length: Optional[int] = None,
        plot_attention_mask: Optional[bool] = False,
        plot_attention_mask_layer: Optional[list[int]] = [0],
        plot_attention_map: Optional[bool] = False,
        plot_attention_map_layer: Optional[list[int]] = [0],
        plot_attention_map_generation: Optional[int] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if prefix_allowed_length is not None:
            self.config.prefix_allowed_length = prefix_allowed_length

        if plot_attention_mask is not None:
            self.config.plot_attention_mask = plot_attention_mask
            if plot_attention_mask_layer is not None:
                self.config.plot_attention_mask_layer = plot_attention_mask_layer

        if plot_attention_map is not None:
            if plot_attention_map_layer is not None:
                self.config.plot_attention_map_layer = plot_attention_map_layer
            if plot_attention_map_generation is not None:
                self.config.plot_attention_map_generation = plot_attention_map_generation
            self.config.plot_attention_map = plot_attention_map

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            segmentation_mask=segmentation_mask, #Added this parameter
            **kwargs,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

@torch.no_grad()
def expand_gpt2_positional_embeddings(
    model: torch.nn.Module,
    new_max_positions: int,
    mode: str = "linear",        # "linear" | "copy_last" | "zeros"
    align_corners: bool = True,  # for linear interpolation
):
    """
    Expand GPT-2's learned positional embeddings (wpe) to `new_max_positions`.

    Works with GPT2LMHeadModel or GPT2Model (HF). Updates model.config.n_positions (and n_ctx if present).
    Does NOT mutate token embeddings; only position table + config.

    Args:
        model: HF GPT2LMHeadModel or GPT2Model (already loaded).
        new_max_positions: int, desired max sequence length (e.g., 1536 or 2048).
        mode: how to initialize new rows if expanding:
              - "linear": 1D linear interpolation along position dim (recommended)
              - "copy_last": copy the last learned vector into all new rows
              - "zeros": initialize new rows to zero
        align_corners: passed to F.interpolate for "linear" mode.

    Returns:
        model (same instance) with expanded wpe and updated config.
    """
    # Locate the position embedding table.
    # Support both:
    # - GPT2LMHeadModel (has .transformer which is a GPT2Model with .wpe)
    # - GPT2Model (exposes .wpe directly)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
        model_for_wpe = model.transformer
    elif hasattr(model, "wpe"):
        model_for_wpe = model
    else:
        raise ValueError("Model does not look like a GPT-2 family model with a position embedding 'wpe')")

    wpe = model_for_wpe.wpe

    old_n, d = wpe.weight.shape
    if new_max_positions <= 0:
        raise ValueError("new_max_positions must be positive")
    if new_max_positions == old_n:
        # Still update config for consistency
        if hasattr(model.config, "n_positions"):
            model.config.n_positions = new_max_positions
        if hasattr(model.config, "n_ctx"):
            model.config.n_ctx = new_max_positions
        return model

    device = wpe.weight.device
    dtype  = wpe.weight.dtype

    if new_max_positions < old_n:
        # Shrink (rare): just slice
        new_weight = wpe.weight[:new_max_positions].clone()
    else:
        # Expand
        if mode == "linear":
            # Interpolate along position dimension.
            # Treat embedding dim as channels: (1, d, old_n) -> (1, d, new_n) -> (new_n, d)
            w = wpe.weight.transpose(0, 1).unsqueeze(0)  # (1, d, old_n)
            w_new = F.interpolate(w, size=new_max_positions, mode="linear", align_corners=align_corners)
            new_weight = w_new.squeeze(0).transpose(0, 1).contiguous()  # (new_n, d)
        elif mode == "copy_last":
            new_weight = torch.empty((new_max_positions, d), device=device, dtype=dtype)
            new_weight[:old_n].copy_(wpe.weight)
            new_weight[old_n:].copy_(wpe.weight[old_n - 1].expand(new_max_positions - old_n, d))
        elif mode == "zeros":
            new_weight = torch.zeros((new_max_positions, d), device=device, dtype=dtype)
            new_weight[:old_n].copy_(wpe.weight)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    # Replace embedding module on whichever object held the original table
    new_wpe = torch.nn.Embedding(new_max_positions, d, device=device, dtype=dtype)
    new_wpe.weight.copy_(new_weight)

    # Keep requires_grad True (default). If you want to freeze, set .requires_grad_(False).
    if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
        model.transformer.wpe = new_wpe
    else:
        model.wpe = new_wpe

    # Update config fields used by HF
    if hasattr(model.config, "n_positions"):
        model.config.n_positions = new_max_positions
    if hasattr(model.config, "n_ctx"):
        model.config.n_ctx = new_max_positions

    return model

def create_decoder(attention = "sdpa"):
    config = GPT2Config.from_pretrained("gpt2")
    config._attn_implementation = attention
    new_max_positions = 2048
    decoder = GPT2LMHeadModelModified.from_pretrained("gpt2", config=config)
    decoder.config._attn_implementation = attention
    decoder = expand_gpt2_positional_embeddings(decoder, new_max_positions=new_max_positions, mode="linear")
    return decoder