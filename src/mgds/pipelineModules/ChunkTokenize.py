from transformers import CLIPTokenizer
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
    print(text)


class ChunkTokenize(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            text_chunks_in_name: str,
            token_chunks_out_name: str,
            mask_chunks_out_name: str,
            tokenizer: CLIPTokenizer,
            max_num_chunks: int = 1,
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

        self.chunk_length: int = tokenizer.model_max_length

        self._bos_id: int = self.tokenizer.bos_token_id
        self._eos_id: int = self.tokenizer.eos_token_id
        self._pad_id: int = self.tokenizer.pad_token_id
        #log(f"Tokenizer BOS: {self._bos_id}, EOS: {self._eos_id}, PAD: {self._pad_id}")

    def length(self) -> int:
        return self._get_previous_length(self.text_chunks_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_chunks_in_name]

    def get_outputs(self) -> list[str]:
        return [self.token_chunks_out_name, self.mask_chunks_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text_chunks: list[str] = self._get_previous_item(variation, self.text_chunks_in_name, index)

        token_chunks, mask_chunks = self.__tokenize_chunks(text_chunks)
        self.__add_empty_chunks(token_chunks, mask_chunks)

        token_chunks = torch.tensor(token_chunks, dtype=torch.long, device=self.pipeline.device)
        mask_chunks  = torch.tensor(mask_chunks, dtype=torch.long, device=self.pipeline.device)

        # log(f"Shapes: tokens={token_chunks.shape}, masks={mask_chunks.shape}")
        # log(f"Final Tokens: {token_chunks}")
        # log(f"Final Mask: {mask_chunks}")

        return {
            self.token_chunks_out_name: token_chunks,
            self.mask_chunks_out_name: mask_chunks,
        }

    def __tokenize_chunks(self, text_chunks: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        current_tokens: list[int] = [self._bos_id]
        current_mask: list[int]   = [1]

        token_chunks: list[list[int]] = [current_tokens]
        mask_chunks: list[list[int]]  = [current_mask]

        max_total_tokens = self.chunk_length * self.max_num_chunks

        #log("Text:")
        #log("".join(text_chunks))
        for text in text_chunks:
            tokenizer_output = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=max_total_tokens,
                add_special_tokens=False,
                return_attention_mask=False
            )

            tokens: list[int] = tokenizer_output.input_ids
            #log(f"Tokenized: '{text}' => ({len(tokens)}) {tokens}")

            # While the tokens don't fit in current chunk
            while (space := self.chunk_length - len(current_tokens) - 1) < len(tokens):
                is_last_chunk = len(token_chunks) >= self.max_num_chunks
                max_pad_length = self.max_last_pad_length if is_last_chunk else self.max_pad_length

                if space > max_pad_length:
                    #log("Split long text")
                    # This text-chunk is long, so split it across chunks
                    current_tokens += tokens[:space]
                    current_mask += [1] * space
                    tokens = tokens[space:]

                self.__finalize_chunk(current_tokens, current_mask)

                if is_last_chunk:
                    return token_chunks, mask_chunks

                # Begin next chunk
                #log("--- New Chunk ---")
                current_tokens = [self._bos_id]
                current_mask   = [1]
                token_chunks.append(current_tokens)
                mask_chunks.append(current_mask)

            # Should token count be 0, which should be rare, this does nothing
            current_tokens += tokens
            current_mask += [1] * len(tokens)

        self.__finalize_chunk(current_tokens, current_mask)
        return token_chunks, mask_chunks

    def __finalize_chunk(self, current_tokens: list[int], current_mask: list[int], eos_mask: int = 1):
        # Append EOS
        current_tokens.append(self._eos_id)
        current_mask.append(eos_mask)

        # Append padding
        pad_length = self.chunk_length - len(current_tokens)
        current_tokens += [self._pad_id] * pad_length
        current_mask += [0] * pad_length

    def __add_empty_chunks(self, token_chunks: list[list[int]], mask_chunks: list[list[int]]):
        # Fill to max chunks to allow for batching. The padding chunks are all completely masked (=0).
        # TODO: This is not strictly required if batch size is 1, but without padding chunks,
        #       the cache would need rebuilding when the batch size is changed.
        #       Maybe the padding could be done later when selecting samples for batches,
        #       dynamically and directly on the embeddings.
        num_pad_chunks = self.max_num_chunks - len(token_chunks)
        if num_pad_chunks > 0:
            pad_tokens, pad_mask = [self._bos_id], [0]
            self.__finalize_chunk(pad_tokens, pad_mask, 0)
            token_chunks += [pad_tokens] * num_pad_chunks
            mask_chunks += [pad_mask] * num_pad_chunks
