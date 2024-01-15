from fastapi import APIRouter, Body, Response, Header
from fastapi.encoders import jsonable_encoder
from database.database import MongoModel
from database.project import create_project, delete_project, find_project_by_name, find_projects
from models.project import Project

router = APIRouter()

@router.get("/")
async def greeting():
    return "hi project"

@router.get("/all") 
async def get_all_projects(username):
    projects = find_projects(username)
    projects_res = []
    for p in projects:
        projects_res.append(Project.from_mongo(p))
    if len(projects_res) == 0:
        return Response(status_code=404, content="There is no project in this user")
    return projects_res

@router.get("/{project_name}")
async def get_project_by_name(username, project_name):
    project = find_project_by_name(username, project_name)
    if project is None:
        return Response(status_code=404, content="project not found")
    return Project.from_mongo(project)

@router.post("/create")
async def crea_project(body: Project):
    data = body.dict()
    project = find_project_by_name(data["username"], data["project_name"])
    if project is not None:
        return Response(status_code=401, content="Duplicate project name")
    create_project(data["project_type"], data["project_name"], data["username"])
    return Response(status_code=201, content="Project created successfully")  

@router.delete("/{project_name}")
async def del_project(username, project_name):
    result = delete_project(username, project_name)
    if result.deleted_count == 0:
        return Response(status_code=404, content="No existing project to delete")
    return Response(status_code=201, content="Project deleted successfully")