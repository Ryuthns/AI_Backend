from pydantic import BaseModel


class Ai_model(BaseModel):
    model_name: str
    model_path: str
