from transformers import CLIPTokenizer
import torch
import math

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


# Chunks: [       Prompt      ][        Continued       ][       Empty      ]
# Tokens: <BOS>CHUNK1<EOS><PAD><BOS>CHUNK2<EOS><PAD><PAD><BOS><EOS><PAD><PAD>
# Padding:                -----                ----------          ----------
# Mask:   1111111111111111000001111111111111111000000000011111111110000000000


class ChunkTokenizerData:
    def __init__(
            self,
            tokenizer: CLIPTokenizer,
            max_num_chunks: int = 1,
            max_pad_length: int = 7,  # This should be accounted for when choosing max_num_chunks
            max_last_pad_length: int = 2,
            unmasked_padding: int = 0,
            dynamic_length: bool = False,
    ):
        self.tokenizer = tokenizer

        self.max_num_chunks = 1000000 if dynamic_length else max_num_chunks
        self.max_pad_length = max_pad_length
        self.max_last_pad_length = min(max_last_pad_length, max_pad_length)
        self.unmasked_padding = unmasked_padding if unmasked_padding >= 0 else 1000000
        self.dynamic_length = dynamic_length

        self.chunk_length: int = tokenizer.model_max_length

        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id
        self.pad_id: int = self.tokenizer.pad_token_id

    @staticmethod
    def with_max_length(tokenizer: CLIPTokenizer, max_length: int):
        max_num_chunks = math.ceil(max_length / tokenizer.model_max_length)
        dynamic_length = (max_length <= 0)
        return ChunkTokenizerData(tokenizer, max_num_chunks=max_num_chunks, dynamic_length=dynamic_length)



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
            max_pad_length: int = 7,
            max_last_pad_length: int = 2,
            unmasked_padding: int = 0,
            dynamic_length: bool = False,
    ):
        super(ChunkTokenize, self).__init__()
        self.text_chunks_in_name = text_chunks_in_name
        self.token_chunks_out_name = token_chunks_out_name
        self.mask_chunks_out_name = mask_chunks_out_name

        self.data = ChunkTokenizerData(
            tokenizer,
            max_num_chunks,
            max_pad_length,
            max_last_pad_length,
            unmasked_padding,
            dynamic_length,
        )

    def length(self) -> int:
        return self._get_previous_length(self.text_chunks_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_chunks_in_name]

    def get_outputs(self) -> list[str]:
        return [self.token_chunks_out_name, self.mask_chunks_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text_chunks: list[str] = self._get_previous_item(variation, self.text_chunks_in_name, index)

        token_chunks, mask_chunks = self.tokenize(self.data, text_chunks, self.pipeline.device)

        return {
            self.token_chunks_out_name: token_chunks,
            self.mask_chunks_out_name: mask_chunks,
        }

    @classmethod
    def tokenize(cls, tokenizer_data: ChunkTokenizerData, text_chunks: list[str], device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        token_chunks, mask_chunks = cls.__tokenize_chunks(tokenizer_data, text_chunks)

        if not tokenizer_data.dynamic_length:
            cls.__add_empty_chunks(tokenizer_data, token_chunks, mask_chunks)

        #cls.__print_debug(token_chunks, mask_chunks)

        token_chunks = torch.tensor(token_chunks, dtype=torch.long, device=device)
        mask_chunks  = torch.tensor(mask_chunks, dtype=torch.long, device=device)
        return token_chunks, mask_chunks

    @classmethod
    def __tokenize_chunks(cls, data: ChunkTokenizerData, text_chunks: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        current_tokens: list[int] = [data.bos_id]
        current_mask: list[int]   = [1]

        token_chunks: list[list[int]] = [current_tokens]
        mask_chunks: list[list[int]]  = [current_mask]

        max_total_tokens = data.chunk_length * data.max_num_chunks

        for text in text_chunks:
            tokenizer_output = data.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=max_total_tokens,
                add_special_tokens=False,
                return_attention_mask=False
            )

            tokens: list[int] = tokenizer_output.input_ids

            # While the tokens don't fit in current chunk. (-1: Leave space for EOS)
            while (space := data.chunk_length - len(current_tokens) - 1) < len(tokens):
                is_last_chunk = len(token_chunks) >= data.max_num_chunks
                max_pad_length = data.max_last_pad_length if is_last_chunk else data.max_pad_length

                if space > max_pad_length:
                    # This text-chunk is long, so split it across chunks
                    current_tokens += tokens[:space]
                    current_mask += [1] * space
                    tokens = tokens[space:]

                cls.__finalize_chunk(data, current_tokens, current_mask)

                if is_last_chunk:
                    return token_chunks, mask_chunks

                # Begin next chunk
                current_tokens = [data.bos_id]
                current_mask   = [1]
                token_chunks.append(current_tokens)
                mask_chunks.append(current_mask)

            # Should token count be 0, which should be rare, this does nothing
            current_tokens += tokens
            current_mask += [1] * len(tokens)

        cls.__finalize_chunk(data, current_tokens, current_mask)
        return token_chunks, mask_chunks

    @staticmethod
    def __finalize_chunk(data: ChunkTokenizerData, current_tokens: list[int], current_mask: list[int]):
        # Append EOS
        current_tokens.append(data.eos_id)
        current_mask.append(1)

        # Append padding
        pad_length = data.chunk_length - len(current_tokens)
        if pad_length > 0:
            current_tokens += [data.pad_id] * pad_length

            current_mask += [1] * min(pad_length, data.unmasked_padding)
            current_mask += [0] * (data.chunk_length - len(current_mask))

    @classmethod
    def __add_empty_chunks(cls, data: ChunkTokenizerData, token_chunks: list[list[int]], mask_chunks: list[list[int]]):
        # Fill to max chunks to allow for batching.
        # TODO: This is not strictly required if batch size is 1, but without padding chunks,
        #       the cache would need rebuilding when the batch size is changed.
        #       Maybe the padding could be done later when selecting samples for batches,
        #       dynamically and directly on the embeddings.
        #       This would reduce cache size, but introduce overhead during training.
        num_pad_chunks = data.max_num_chunks - len(token_chunks)
        if num_pad_chunks > 0:
            pad_tokens, pad_mask = [data.bos_id], [1]
            cls.__finalize_chunk(data, pad_tokens, pad_mask)
            token_chunks += [pad_tokens] * num_pad_chunks
            mask_chunks += [pad_mask] * num_pad_chunks


    @staticmethod
    def __print_debug(token_chunks: list[list[int]], mask_chunks: list[list[int]]):
        assert len(token_chunks) == len(mask_chunks)
        for chunk_nr, (tokens, mask) in enumerate(zip(token_chunks, mask_chunks), 1):
            title = f"[{chunk_nr}] "
            print(title, end="")

            for i, (t, m) in enumerate(zip(tokens, mask), 1):
                print(f"{t:5}:{m}  ", end="")
                if i % 10 == 0 and i < len(tokens):
                    print()
                    print(" " * len(title), end="")

            print()
        print()
