import math
from typing import Any, Callable

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class QuantizeResolution(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
        self,
        quantization: int,
        resolution_in_name: str,
        scale_resolution_out_name: str,
        crop_resolution_out_name: str,
        possible_resolutions_out_name: str
    ):
        super(QuantizeResolution, self).__init__()
        self.quantization = quantization
        self.resolution_in_name = resolution_in_name
        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name

        # TODO: Possible resolutions is used only in RandomLatentMaskRemove.
        # It uses the sizes to pre-allocate empty conditioning images.
        # -> Not needed right now.
        self.possible_resolutions_out_name = possible_resolutions_out_name
        self._possible_resolutions = [
            (832, 1216), (1216, 832),
            (896, 1152), (1152, 896),
            (1024, 1024),
            (1024, 1536), (1536, 1024),
            (1280, 1280), (1536, 1536),
        ]

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.scale_resolution_out_name, self.crop_resolution_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        res: tuple[int, int] = self._get_previous_item(variation, self.resolution_in_name, index)
        h, w = res

        if h % self.quantization or w % self.quantization:
            res_candidates = [
                self.__quantize_res(h, w, math.floor, math.floor),
                self.__quantize_res(h, w, math.ceil,  math.floor),
                self.__quantize_res(h, w, math.floor, math.ceil),
                self.__quantize_res(h, w, math.ceil,  math.ceil)
            ]

            aspect_ratio = w / h
            res_closest = min(res_candidates, key=lambda t: abs(aspect_ratio - t[2]))
            res = res_closest[:2]

        return {
            self.scale_resolution_out_name: res,
            self.crop_resolution_out_name: res
        }

    def __quantize_res(self, h: int, w: int, func_h: Callable[[float], int], func_w: Callable[[float], int]) -> tuple[int, int, float]:
        h = func_h(h / self.quantization) * self.quantization
        w = func_w(w / self.quantization) * self.quantization
        ar = w / h
        return (h, w, ar)

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return self._possible_resolutions
        else:
            return None
