import math
from torchvision.transforms import functional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class PadImage(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            name: str,
            target_resolution_in_name: str,
            padding_mode: str = "edge",
            quantization: int = 0,
    ):
        super(PadImage, self).__init__()
        self.name = name
        self.target_resolution_in_name = target_resolution_in_name
        self.padding_mode = padding_mode
        self.quantization = quantization

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.name]

    def get_outputs(self) -> list[str]:
        return [self.name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.name, index)
        h, w = image.shape[-2:]

        target_resolution: tuple[int, int] = self._get_previous_item(variation, self.target_resolution_in_name, index)
        target_h, target_w = target_resolution

        pad_bottom = max(target_h - h, 0)
        pad_right  = max(target_w - w, 0)

        if pad_bottom or pad_right:
            padding = (0, 0, pad_right, pad_bottom)
            image = functional.pad(image, padding, padding_mode=self.padding_mode)

            if self.quantization > 0:
                fill_bottom = math.ceil(pad_bottom / self.quantization) * self.quantization
                if fill_bottom > 0:
                    image[..., :, target_h - fill_bottom:, :] = 0

                fill_right = math.ceil(pad_right / self.quantization) * self.quantization
                if fill_right > 0:
                    image[..., :, :, target_w - fill_right:] = 0

        return {
            self.name: image
        }
