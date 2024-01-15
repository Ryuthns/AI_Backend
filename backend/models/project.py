from pydantic import BaseModel


class Project(BaseModel):
    project_type: str
    project_name: str
    project_path: str
    username: str
    
    @classmethod
    def from_mongo(cls, data: dict):
        id = data.pop("_id", None)
        return cls(**dict(data, id=id))