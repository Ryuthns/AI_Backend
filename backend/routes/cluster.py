from fastapi import APIRouter, Body, Response
from fastapi.responses import FileResponse
import os
import json

from cluster import cluster_result

router = APIRouter()


@router.get("")
async def cluster(project_name: str = "", username: str = ""):
    result_path = os.path.join("user_project", username, project_name, "cluster.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            data = json.load(f)
            return data

    images_path = f"user_project/{username}/{project_name}/images"
    labels_path = f"user_project/{username}/{project_name}/labels/classification.json"

    save_path = f"user_project/{username}/{project_name}"
    data = cluster_result(images_path, labels_path, save_path)
    return data


@router.get("/unique")
async def cluster_unique(project_name: str = "", username: str = ""):
    result_path = os.path.join("user_project", username, project_name, "unique.png")
    if os.path.exists(result_path):
        return FileResponse(result_path)
    return Response("confusion matrix not found", 404)
