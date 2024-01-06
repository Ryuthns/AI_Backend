from database.database import MongoModel, client
from helper.helper import verify_password, hash_password

data = client["project"]
user_collection = data.get_collection("user")


def create_user(data):
    data["password"] = hash_password(data["password"])
    user_collection.insert_one(data)


def find_user_by_name(username):
    # Query the collection for a user with the specified username
    # user = user_collection.find()
    user = user_collection.find_one({"username": username})
    return user


def find_user():
    # Query the collection for a user with the specified username
    # user = user_collection.find()
    user = user_collection.find()
    return user


def login(data):
    user = find_user_by_name(data["username"])
    if user is None:
        return False

    if verify_password(data["password"], user["password"]):
        return True
    return False


def delete_user(username):
    result = user_collection.delete_one({"username": username})
    return result
