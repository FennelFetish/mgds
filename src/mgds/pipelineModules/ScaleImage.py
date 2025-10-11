from torchvision.transforms import functional, InterpolationMode

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ScaleImage(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, in_name: str, out_name: str, factor: float, interpolation: str = "bilinear"):
        super(ScaleImage, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.factor = factor
        self.interpolation = InterpolationMode(interpolation)

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)

        h, w = image.shape[-2:]
        size = (round(h * self.factor), round(w * self.factor))

        image = functional.resize(image, size, interpolation=self.interpolation, antialias=True)

        return {
            self.out_name: image
        }
