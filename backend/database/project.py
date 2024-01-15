from database.database import MongoModel, client

data = client["project"]
project_collection = data.get_collection("projects")

# Create a new project in the database
def create_project(project_type, project_name, username):
    project = {"project_type": project_type, "project_name": project_name, "project_path": f"user_project/{username}/{project_name}", "username": username}   
    project_collection.insert_one(project)

# Find a project belonging to a user
def find_project_by_name(username, project_name):
    project = project_collection.find_one({"username": username, "project_name": project_name})
    return project

# Find all projects belonging to a user
def find_projects(username):
    projects = project_collection.find({"username": username})
    return projects

# Delete a project belonging to a user
def delete_project(username, project_name):
    result = project_collection.delete_one({"username": username, "project_name": project_name})
    return result