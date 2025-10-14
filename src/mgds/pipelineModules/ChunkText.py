from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ChunkText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            text_in_name: str,
            text_chunks_out_name: str,
            delimeter_chars: str = ",.:;"
    ):
        super(ChunkText, self).__init__()
        self.text_in_name = text_in_name
        self.text_chunks_out_name = text_chunks_out_name
        self.delimeter_chars = delimeter_chars

    def length(self) -> int:
        return self._get_previous_length(self.text_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_in_name]

    def get_outputs(self) -> list[str]:
        return [self.text_chunks_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text: str = self._get_previous_item(variation, self.text_in_name, index)

        return {
            self.text_chunks_out_name: self.chunk_text(text)
        }


    @staticmethod
    def chunk_text(text: str, delimeter_chars: str = ",.:;") -> list[str]:
        text_chunks = list[str]()

        while text:
            deli_index = next((i for i, char in enumerate(text) if char in delimeter_chars), -1)
            if deli_index < 0:
                text_chunks.append(text)
                break

            # Put delimeter into separate chunk so tags fit better when token-chunks are getting full.
            text_chunks.append(text[:deli_index])
            text_chunks.append(text[deli_index])
            text = text[deli_index+1:]

        return text_chunks
