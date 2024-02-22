from fastapi import APIRouter, Body, Response, Header
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from torchvision import os
from database.database import MongoModel
import shutil

from helpers.model import load_metadata

router = APIRouter()


@router.get("/")
async def get_models(project_name: str = "", username: str = ""):
    if project_name == "" or username == "":
        return Response("invalid input", status_code=401)
    path = f"user_project/{username}/{ project_name }/models"
    model_list = os.listdir(path)
    result = []
    for i, model_name in enumerate(model_list):
        data = {"_id": i, "model_name": model_name}
        result.append(data)
    return result


@router.post("/")
async def create_models(
    project_name: str = "", username: str = "", model_name: str = ""
):
    if project_name == "" or username == "" or model_name == "":
        return Response("invalid input", status_code=401)
    path = f"user_project/{username}/{ project_name }/models/{model_name}"
    if not os.path.exists(path):
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


@router.get("/metadata")
async def get_metadata(
    project_name: str = "", username: str = "", model_name: str = ""
):
    if project_name == "" or username == "":
        return Response("invalid input", status_code=401)
    path = os.path.join("user_project", username, project_name, "models", model_name)
    data = load_metadata(path)
    return data


@router.get("/confusion_matrix")
async def get_confusion_matrix(
    project_name: str = "", username: str = "", model_name: str = ""
):
    if project_name == "" or username == "":
        return Response("invalid input", status_code=401)
    path = os.path.join("user_project", username, project_name, "models", model_name)
    confusion_matrix_path = os.path.join(path, "confusion_matrix.png")
    if os.path.exists(confusion_matrix_path):
        return FileResponse(confusion_matrix_path)
    return Response("confusion matrix not found", 404)
