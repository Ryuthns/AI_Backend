from fastapi import APIRouter, Body, Response
from fastapi.encoders import jsonable_encoder
from database.database import MongoModel

from database.user import create_user, delete_user, find_user, find_user_by_name, login
from models.user import User, UserLogin

router = APIRouter()


@router.get("/")
async def greeting():
    return "hi user"


@router.get("/all")
async def get_all_user():
    user = find_user()
    users_res = []
    for u in user:
        users_res.append(User.from_mongo(u))
    return users_res


@router.get("/{username}")
async def get_user_by_id(username):
    user = find_user_by_name(username)
    if user is None:
        return Response(status_code=404, content="user not found")
    return User.from_mongo(user)


@router.post("/login")
async def user_login(body: UserLogin):
    data = jsonable_encoder(body)
    is_login = login(data)
    if is_login:
        user = find_user_by_name(data["username"])
        if user is None:
            return Response(status_code=500, content="failed to query user")
        return User.from_mongo(user)
    return Response(status_code=401, content="faild to login")


@router.post("/signup")
async def user_signup(body: UserLogin):
    data = jsonable_encoder(body)
    user = find_user_by_name(data["username"])
    if user is not None:
        return Response(status_code=401, content="duplicate username")
    data["projects"] = []
    create_user(data)
    return Response(status_code=201, content="create user success")


@router.delete("/{username}")
async def delete_user_handler(username):
    result = delete_user(username)
    if result.deleted_count == 0:
        return {"error": "User not found"}
    else:
        return {"message": f"{result.deleted_count} user deleted"}
