from fastapi import APIRouter, Body, Response
from fastapi.encoders import jsonable_encoder

from database.user import create_user, find_user_by_name, login
from models.user import UserLogin

router = APIRouter()


@router.post("/login")
async def user_login(body: UserLogin):
    data = jsonable_encoder(body)
    is_login = login(data)
    if is_login:
        user = find_user_by_name(data["username"])
        return user
    return Response(status_code=401, content="faild to login")


@router.post("/signup")
async def user_signup(body: UserLogin):
    data = jsonable_encoder(body)
    create_user(data)


@router.get("/{username}")
async def get_user_by_id(username):
    user = find_user_by_name(username)
    return user
