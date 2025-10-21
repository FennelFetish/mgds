from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SelectInput(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            setting_name: str,
            out_name: str,
            setting_to_in_name_map: dict[str, str | None],
            default_in_name: str | None = None
    ):
        super(SelectInput, self).__init__()
        self.setting_name = setting_name
        self.out_name = out_name
        self.setting_to_in_name_map = setting_to_in_name_map
        self.default_in_name = default_in_name

        self.in_names = [name for name in setting_to_in_name_map.values() if name is not None]

    def length(self) -> int:
        return self._get_previous_length(self.in_names[0])

    def get_inputs(self) -> list[str]:
        return self.in_names

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        setting = self._get_previous_item(variation, self.setting_name, index)

        in_name = self.setting_to_in_name_map[setting]
        out = None

        if in_name is not None:
            out = self._get_previous_item(variation, in_name, index)

            if out is None and self.default_in_name is not None:
                out = self._get_previous_item(variation, self.default_in_name, index)

        return {
            self.out_name: out
        }
