from typing import List
from pydantic import BaseModel
from models.project import Project


class UserLogin(BaseModel):
    username: str
    password: str


class User(BaseModel):
    username: str
    password: str
    projects: List[Project]
