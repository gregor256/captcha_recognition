from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class ModelParams:
    input_filepath: str
    model_parameters_filepath: str
    num_epochs: int
    letters_amount: int
    image_rotation_degree: int


ModelParamsSchema = class_schema(ModelParams)


def read_model_params(path: str) -> ModelParams:
    with open(path, "r") as input_stream:
        schema = ModelParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
