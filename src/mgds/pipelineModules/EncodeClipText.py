from contextlib import nullcontext

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeClipText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            pooled_out_name: str | None,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
            add_layer_norm: bool,
            hidden_state_output_index: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeClipText, self).__init__()
        self.in_name = in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.text_encoder = text_encoder
        self.add_layer_norm = add_layer_norm
        self.hidden_state_output_index = hidden_state_output_index

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        if self.pooled_out_name:
            return [self.hidden_state_out_name, self.pooled_out_name]
        else:
            return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.in_name, index)
        tokens = torch.atleast_2d(tokens)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
            tokens_attention_mask = torch.atleast_2d(tokens_attention_mask)
        else:
            tokens_attention_mask = None

        with self._all_contexts(self.autocast_contexts):
            if tokens_attention_mask is not None and self.dtype:
                tokens_attention_mask = tokens_attention_mask.to(dtype=self.dtype)

            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=tokens_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Pooled State shape: [n, 768 or 1280]
        pooled_state: torch.Tensor | None = None
        if self.pooled_out_name:
            if hasattr(text_encoder_output, "text_embeds"):
                pooled_state = text_encoder_output.text_embeds
            if hasattr(text_encoder_output, "pooler_output"):
                pooled_state = text_encoder_output.pooler_output

            if pooled_state is not None:
                # Use pooler output of first chunk only
                pooled_state = pooled_state[0].clone()

        hidden_states = text_encoder_output.hidden_states
        hidden_state = hidden_states[self.hidden_state_output_index]  # shape: [n, 77, 768 or 1280]

        # Apply layer norm before reshaping
        if self.add_layer_norm:
            with self._all_contexts(self.autocast_contexts):
                final_layer_norm = self.text_encoder.text_model.final_layer_norm
                hidden_state = final_layer_norm(hidden_state)

        # Resulting shape: [77*n, 768 or 1280]
        num_chunks, num_tokens, state_length = hidden_state.shape
        hidden_state = hidden_state.reshape(num_chunks * num_tokens, state_length)

        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state,
        }
