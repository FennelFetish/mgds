import json

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class LoadJson(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, path_in_name: str, json_out_name: str):
        super(LoadJson, self).__init__()
        self.path_in_name = path_in_name
        self.json_out_name = json_out_name

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.json_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)

        try:
            with open(path, encoding='utf-8') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            json_data = None
        except:
            print("could not load json, it might be corrupted: " + path)
            raise

        return {
            self.json_out_name: json_data
        }
