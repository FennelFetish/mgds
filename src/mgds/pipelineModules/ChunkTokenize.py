from transformers import CLIPTokenizer, T5Tokenizer, T5TokenizerFast, GemmaTokenizer, LlamaTokenizer
import torch

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


# Chunks: [       Prompt      ][        Continued       ][       Empty      ]
# Tokens: <BOS>CHUNK1<EOS><PAD><BOS>CHUNK2<EOS><PAD><PAD><BOS><EOS><PAD><PAD>
# Padding:                -----                ----------          ----------
# Mask:   1111111111111111000001111111111111111000000000000000000000000000000


# TODO: Variations / Augmentation for padding and attention mask:
# - Different max_pad_length
# - Shuffle text-chunks around chunks
# - Different masks, some with padding (or patches of padding) enabled
# It should learn that padding and chunk number is irrelevant.


def log(text: str):
    #print(text)
    pass


class ChunkTokenize(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            text_chunks_in_name: str,
            token_chunks_out_name: str,
            mask_chunks_out_name: str,
            tokenizer: CLIPTokenizer | T5Tokenizer | T5TokenizerFast | GemmaTokenizer | LlamaTokenizer,
            max_num_chunks: int = 3,
            max_pad_length: int = 7,  # This should be accounted for when choosing max_num_chunks
            max_last_pad_length: int = 2,
    ):
        super(ChunkTokenize, self).__init__()
        self.text_chunks_in_name = text_chunks_in_name
        self.token_chunks_out_name = token_chunks_out_name
        self.mask_chunks_out_name = mask_chunks_out_name
        self.tokenizer = tokenizer

        self.max_num_chunks = max_num_chunks
        self.max_pad_length = max_pad_length
        self.max_last_pad_length = min(max_last_pad_length, max_pad_length)

        self.chunk_length = tokenizer.model_max_length

    def length(self) -> int:
        return self._get_previous_length(self.text_chunks_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_chunks_in_name]

    def get_outputs(self) -> list[str]:
        return [self.token_chunks_out_name, self.mask_chunks_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text_chunks = self._get_previous_item(variation, self.text_chunks_in_name, index)

        log(f"Tokenizer BOS: {self.tokenizer.bos_token_id}, EOS: {self.tokenizer.eos_token_id}, PAD: {self.tokenizer.pad_token_id}")
        bos = torch.full((1,), self.tokenizer.bos_token_id)
        eos = torch.full((1,), self.tokenizer.eos_token_id)

        token_chunks, mask_chunks = self.__tokenize_chunks(text_chunks, bos, eos)

        # Finalize last chunk and fill to max chunks to allow for batching.
        # The padding chunks are all completely masked (=0).
        # TODO: This is not required if batch size is 1.
        num_pad_chunks = self.max_num_chunks - len(token_chunks)
        if num_pad_chunks > 0:
            pad_tokens, pad_mask = [bos], [0]
            self.__finalize_chunk(pad_tokens, pad_mask, eos, 0)
            token_chunks.extend([pad_tokens] * num_pad_chunks)
            mask_chunks.extend([pad_mask] * num_pad_chunks)

        token_chunks = [torch.cat(tokens) for tokens in token_chunks]
        mask_chunks  = [torch.LongTensor(mask) for mask in mask_chunks] # Clip Tokenizer outputs mask as int64

        log("===> Processed Chunks:")
        for i, (chunk, mask) in enumerate(zip(token_chunks, mask_chunks)):
            log(f"=> Tokens {i} ({chunk.shape})\n{chunk}")
            log(f"=> Mask {i} ({mask.shape})\n{mask}")

        log(f"Processed shape: tokens={token_chunks[0].shape}, masks={mask_chunks[0].shape}")

        stacked_tokens = torch.stack(token_chunks).to(self.pipeline.device)
        stacked_masks  = torch.stack(mask_chunks).to(self.pipeline.device)

        log(f"Stacked shape: tokens={stacked_tokens.shape}, masks={stacked_masks.shape}")

        return {
            self.token_chunks_out_name: stacked_tokens,
            self.mask_chunks_out_name: stacked_masks,
        }

    def __tokenize_chunks(self, text_chunks: list[str], bos: torch.Tensor, eos: torch.Tensor):
        current_tokens = [bos]
        current_mask   = [1]

        token_chunks = [current_tokens]
        mask_chunks  = [current_mask]

        max_token_length = self.chunk_length * self.max_num_chunks

        for text in text_chunks:
            # Output includes BOS and EOS, no padding
            tokenizer_output = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=max_token_length,
                return_tensors="pt",
            )

            #tokens: torch.Tensor = tokenizer_output.input_ids     # Shape: (1, n)
            #mask: torch.Tensor = tokenizer_output.attention_mask  # Shape: (1, n), Content: All ones

            # Remove BOS and EOS
            tokens = tokenizer_output.input_ids.squeeze(dim=0)[1:-1]

            log(f"Tokenized: '{text}' => ({tokens.numel()}) {tokens}")

            while (space := self.chunk_length - len(current_mask) - 1) < tokens.numel():
                is_last_chunk = len(token_chunks) >= self.max_num_chunks
                max_pad_length = self.max_last_pad_length if is_last_chunk else self.max_pad_length

                if space > max_pad_length:
                    log("Split long text")
                    # This text-chunk is long, so split it across chunks
                    current_tokens.append(tokens[:space])
                    current_mask += [1] * space
                    tokens = tokens[space:]

                self.__finalize_chunk(current_tokens, current_mask, eos)

                if is_last_chunk:
                    return token_chunks, mask_chunks

                # Begin next chunk
                log("--- New Chunk ---")
                current_tokens = [bos]
                current_mask   = [1]
                token_chunks.append(current_tokens)
                mask_chunks.append(current_mask)

            # Should num_tokens be 0, which should be rare, this does nothing
            current_tokens.append(tokens)
            current_mask += [1] * tokens.numel()

        self.__finalize_chunk(current_tokens, current_mask, eos)
        return token_chunks, mask_chunks

    def __finalize_chunk(self, current_tokens: list, current_mask: list, eos, eos_mask: int = 1):
        # Append EOS
        current_tokens.append(eos)
        current_mask.append(eos_mask)

        # Append padding
        pad_length = self.chunk_length - len(current_mask)
        current_tokens.append(torch.full((pad_length,), self.tokenizer.pad_token_id))
        current_mask += [0] * pad_length
