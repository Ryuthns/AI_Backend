from pydantic import BaseModel


class Ai_model(BaseModel):
    model_name: str
    model_path: str


class result_input(BaseModel):
    username: str
    project_name: str
    modelname: str
