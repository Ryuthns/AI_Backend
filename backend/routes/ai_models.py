from fastapi import APIRouter, Body, Response, Header
from fastapi.encoders import jsonable_encoder
from torchvision import os
from database.database import MongoModel
import shutil

router = APIRouter()


@router.get("/")
async def get_models(project_name: str = "", username: str = ""):
    if project_name == "" or username == "":
        return Response("invalid input", status_code=401)
    path = f"user_project/{username}/{ project_name }/models"
    return os.listdir(path)


@router.post("/")
async def create_models(
    project_name: str = "", username: str = "", model_name: str = ""
):
    if project_name == "" or username == "" or model_name == "":
        return Response("invalid input", status_code=401)
    path = f"user_project/{username}/{ project_name }/models/{model_name}"
    os.makedirs(path)
    return Response("models was created", 201)


@router.delete("/")
async def delete_models(
    project_name: str = "", username: str = "", model_name: str = ""
):
    if project_name == "" or username == "" or model_name == "":
        return Response("invalid input", status_code=401)
    path = f"user_project/{username}/{ project_name }/models/{model_name}"
    try:
        shutil.rmtree(path)
        print(f"Folder '{path}' removed successfully.")
        return Response(f"Folder '{path}' removed successfully.", 200)
    except Exception as e:
        return Response(f"Error: {e}", 500)
