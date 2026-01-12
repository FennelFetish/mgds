import json
from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class LoadJson(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, path_in_name: str, data_out_name: str, key_in_name: str | None = None):
        super(LoadJson, self).__init__()
        self.path_in_name = path_in_name
        self.key_in_name = key_in_name

        self.data_out_name = data_out_name

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.data_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)

        try:
            with open(path, encoding='utf-8') as f:
                json_data = json.load(f)

            if self.key_in_name:
                json_key = self._get_previous_item(variation, self.key_in_name, index)
                json_data = self.__get_data(json_data, json_key)
        except FileNotFoundError:
            json_data = None
        except:
            print("could not load json, it might be corrupted: " + path)
            raise

        return {
            self.data_out_name: json_data
        }

    @staticmethod
    def __get_data(json_data: Any, json_key: str) -> Any:
        json_key = json_key.strip()
        if not json_key:
            return json_data

        json_path = (key.strip() for key in json_key.split("."))
        for key in json_path:
            if isinstance(json_data, dict):
                json_data = json_data.get(key)
            else:
                return None

        return json_data
