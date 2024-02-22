from pydantic import BaseModel


class Ai_model(BaseModel):
    model_name: str
    model_path: str


class result_input(BaseModel):
    username: str
    project_name: str
    modelname: str


class MetadataModel(BaseModel):
    average_precision: float = 0
    precision: float = 0
    recall: float = 0
    total_image: int = 0
    train_image: int = 0
    validate_image: int = 0
    test_image: int = 0
    train_ratio: float = 0
    validate_ratio: float = 0
    test_ratio: float = 0
    training_status: bool = False
