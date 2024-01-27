import json
import os
from typing import BinaryIO, List, Union

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from classification import TrainClassification
from helpers.helper import get_key_by_value
from models.ai_models import result_input
from objectdetection import ObjectDetection
from routes.ai_models import router as ModelsRouter
from routes.project import router as ProjectRouter
from routes.user import router as UserRouter

app = FastAPI()

origins = ["*"]
mapping = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(UserRouter, prefix="/user")
app.include_router(ProjectRouter, prefix="/project")
app.include_router(ModelsRouter, prefix="/model")
result_data = {}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/result/")
async def get_result(result_data: result_input):
    data = result_data.model_dump()
    c = TrainClassification(
        2, data["username"], data["project_name"], data["modelname"]
    )
    result = c._load_train_result()
    if not result:
        return Response("failed to get result", status_code=404)
    return result


@app.post("/uploadfile/")
async def create_upload_file(
    bytefiles: List[UploadFile] = File(...),
    num_class: int = 2,
    username: str = Form(...),
    project_name: str = Form(...),
    modelname: str = Form(...),
):
    bytefile_data = [bf.file for bf in bytefiles]
    c = TrainClassification(num_class, username, project_name, modelname)
    result = c.predict(bytefile_data)
    predicted_labels = []
    for p in result["predicted"]:
        print(get_key_by_value(mapping, p))
        predicted_labels.append(get_key_by_value(mapping, p))
    result["predicted_labels"] = predicted_labels
    return result


@app.post("/saveimage/")
async def save_image(
    bytefiles: List[UploadFile] = File(...),
    username: str = Form(...),
    project_name: str = Form(...),
    labels: List[str] = Form(...),
):
    # Save user's labels and images
    try:
        # save labels
        filename_list = [file.filename for file in bytefiles]
        label_list = [labels[i] for i in range(len(labels))]
        data = [
            {"image": filename, "annotations": [label]}
            for filename, label in zip(filename_list, label_list)
        ]
        json_data = json.dumps(data, indent=2)

        directory_path = f"user_project/{username}/{project_name}/labels"
        os.makedirs(directory_path, exist_ok=True)
        file_path = os.path.join(directory_path, "classification.json")

        with open(file_path, "w") as json_file:
            json_file.write(json_data)

        # save images
        for file in bytefiles:
            directory_path = f"user_project/{username}/{project_name}/images"

            os.makedirs(directory_path, exist_ok=True)

            file_path = os.path.join(directory_path, file.filename)

            with open(file_path, "wb") as f:
                f.write(file.file.read())

        return Response("Files saved successfully", status_code=200)
    except Exception as e:
        return Response(
            f"failed to save image(s) and label(s) {e.args}", status_code=404
        )


@app.get("/getimage/")
async def get_images(username: str = Form(...), project_name: str = Form(...)):
    try:
        folder_path = f"user_project/{username}/{project_name}/images"

        image_urls = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Assuming all files in the folder are images
                image_urls.append(
                    f"http://localhost:8000/image/?username={username}&project_name={project_name}&file_name={file}"
                )  # Replace with your actual API server URL

        file_path = f"user_project/{username}/{project_name}/labels/classification.json"
        with open(file_path, "r") as f:
            label = json.load(f)

        response_data = {
            "username": username,
            "project_name": project_name,
            "image_urls": image_urls,
            "labels": label,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to get image URLs: {e}"}, status_code=500
        )


@app.get("/image/")
async def get_image(
    username: str = Query(...),
    project_name: str = Query(...),
    file_name: str = Query(...),
):
    try:
        folder_path = f"user_project/{username}/{project_name}/images"
        file_path = os.path.join(folder_path, file_name)
        return FileResponse(file_path)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to get image URLs: {e}"}, status_code=500
        )


@app.post("/train/")
async def train_model(
    bytefiles: List[UploadFile] = File(...),
    labels: List[str] = Form(...),
    epochs: int = Form(...),
    lr: float = Form(...),
    username: str = Form(...),
    project_name: str = Form(...),
    modelname: str = Form(...),
):
    # Convert UploadFile objects to BinaryIO
    bytefile_data = [bf.file for bf in bytefiles]
    print(bytefile_data)

    # Convert the labels to integers
    label_list = []
    for value in labels:
        if value in mapping:
            integer_index = mapping[value]
        else:
            # If the value is not in the dictionary, add it with a new integer index
            new_index = len(mapping)
            mapping[value] = new_index
            integer_index = new_index
        label_list.append(integer_index)

    # Call the train method with the received data
    c = TrainClassification(len(set(label_list)), username, project_name, modelname)
    result = c.train(bytefile_data, label_list, epochs, lr)
    global result_data
    result_data = result

    # Return the result as JSON
    return result


@app.post("/object_train/")
def object_detection_train(
    username: str = Form(...),
    project_name: str = Form(...),
    modelname: str = Form(...),
    epochs: int = Form(...),
):
    obj = ObjectDetection(username, project_name, modelname)
    result = obj.train(epochs)
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
