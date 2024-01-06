from pydantic import BaseModel


class Project(BaseModel):
    project_type: str
    project_name: str
    project_path: str
