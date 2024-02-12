import json
import os
import queue
import shutil
import threading
from typing import Dict, List, Union
import asyncio

import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    Query,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from classification import TrainClassification, get_labels_dict
from helpers.helper import get_key_by_value
from models.ai_models import result_input
from objectdetection import ObjectDetection, prepare_yaml, save_predict_img
from routes.ai_models import router as ModelsRouter
from routes.project import router as ProjectRouter
from routes.user import router as UserRouter

app = FastAPI()

origins = ["*"]
mapping = {}

main_queue = queue.Queue(maxsize=0)

# Dictionary to store WebSocket connections for each channel
channel_connections: Dict[str, WebSocket] = {}

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
    folder_path = os.path.join("user_project", username, project_name)
    mapping = get_labels_dict(folder_path)
    print("result", result)
    for p in result["predicted"]:
        print(get_key_by_value(mapping, p))
        predicted_labels.append(get_key_by_value(mapping, p))
    result["predicted_labels"] = predicted_labels
    return result


@app.post("/saveimage/")
async def save_image(
    file_name: List[str] = Form(...),
    image_file: List[UploadFile] = File(None),
    username: str = Form(...),
    project_name: str = Form(...),
    labels: List[str] = Form(...),
):
    # Save user's labels and images
    try:
        # save labels
        label_list = [labels[i] for i in range(len(labels))]
        data = [
            {"image": filename, "annotations": [label]}
            for filename, label in zip(file_name, label_list)
        ]
        sorted_data = sorted(data, key=lambda x: x["image"])
        json_data = json.dumps(sorted_data, indent=2)

        directory_path = f"user_project/{username}/{project_name}/labels"
        os.makedirs(directory_path, exist_ok=True)
        file_path = os.path.join(directory_path, "classification.json")

        with open(file_path, "w") as json_file:
            json_file.write(json_data)

        # save images
        if image_file is not None:
            for file in image_file:
                directory_path = f"user_project/{username}/{project_name}/images"

                os.makedirs(directory_path, exist_ok=True)

                file_path = os.path.join(directory_path, file.filename)

                with open(file_path, "wb") as f:
                    f.write(file.file.read())
            return Response("File(s) saved successfully", status_code=200)

        return Response("Label(s) updated successfully", status_code=200)
    except Exception as e:
        return Response(
            f"Failed to save image(s) and label(s) {e.args}", status_code=404
        )


@app.post("/saveobject/")
async def save_object(
    image_file: List[UploadFile] = File(None),
    username: str = Form(...),
    project_name: str = Form(...),
    labels: str = Form(...),
):
    try:
        # Save labels
        labels_dict: Dict[str, str] = json.loads(labels)
        labels_directory = f"user_project/{username}/{project_name}/labels"
        os.makedirs(labels_directory, exist_ok=True)
        for key, value in labels_dict.items():
            file_name, _ = os.path.splitext(key)
            labels_path = os.path.join(labels_directory, f"{file_name}.txt")

            with open(labels_path, "w") as json_file:
                # Write each bounding box annotation as a separate line
                for annotation in value:
                    json_file.write(annotation + "\n")
        prepare_yaml(username, project_name, ["mask", "no_mask"])

        # Save images
        if image_file is not None:
            images_directory = f"user_project/{username}/{project_name}/images"
            os.makedirs(images_directory, exist_ok=True)

            for file in image_file:
                # Use a different variable for the images directory path
                image_directory_path = f"user_project/{username}/{project_name}/images"

                image_path = os.path.join(image_directory_path, file.filename)
                with open(image_path, "wb") as f:
                    f.write(file.file.read())

            return Response("Image(s) saved successfully", status_code=200)

        return Response("Label(s) updated successfully", status_code=200)

    except Exception as e:
        return Response(
            f"Failed to save image(s) and label(s): {str(e)}", status_code=500
        )


@app.post("/getimage/")
async def get_images(username: str = Form(...), project_name: str = Form(...)):
    try:
        folder_path = f"user_project/{username}/{project_name}/images"

        image_urls = []
        for root, _, files in os.walk(folder_path):
            files.sort()
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


@app.post("/getobject/")
async def get_object(username: str = Form(...), project_name: str = Form(...)):
    try:
        folder_path = f"user_project/{username}/{project_name}/images"

        response_data = {
            "username": username,
            "project_name": project_name,
            "images": [],
        }

        for root, _, files in os.walk(folder_path):
            files.sort()
            for file in files:
                image_url = f"http://localhost:8000/image/?username={username}&project_name={project_name}&file_name={file}"
                file_name, _ = os.path.splitext(file)
                label_file_path = (
                    f"user_project/{username}/{project_name}/labels/{file_name}.txt"
                )

                # Read the plain text file for labels
                try:
                    with open(label_file_path, "r") as f:
                        lines = f.readlines()
                        labels = [
                            line.strip() for line in lines if line.strip()
                        ]  # Filter out empty lines
                except FileNotFoundError:
                    labels = []  # Handle the case where the label file is not found

                image_data = {"filename": file, "url": image_url, "labels": labels}
                response_data["images"].append(image_data)

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get image URLs and labels: {str(e)}"
        )


@app.delete("/deletefolder/")
async def delete_folder(username: str, project_name: str):
    try:
        folder_path = f"user_project/{username}/{project_name}"
        shutil.rmtree(folder_path)
        return Response("Folder deleted successfully", status_code=200)
    except Exception as e:
        return Response(f"Failed to delete folder {e.args}", status_code=500)


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
    # bytefiles: List[UploadFile] = File(...),
    # labels: List[str] = Form(...),
    epochs: int = Form(...),
    lr: float = Form(...),
    username: str = Form(...),
    project_name: str = Form(...),
    modelname: str = Form(...),
):
    # Call the train method with the received data
    c = TrainClassification(len(set([])), username, project_name, modelname)
    on_successs = lambda: send_message_to_channel(
        f"{username}_{project_name}", {"training_status": "success"}
    )
    print("channel", f"{username}_{project_name}")
    # threading.Thread(target=c.train, args=(epochs, lr, on_successs)).start()
    threading.Thread(
        target=wrap_async_func, args=(c.train, epochs, lr, on_successs)
    ).start()
    # result = c.train(epochs, lr)

    # Return the result as JSON
    return "running"


@app.post("/object/train")
def object_detection_train(
    username: str = Form(...),
    project_name: str = Form(...),
    modelname: str = Form(...),
    epochs: int = Form(...),
):
    prepare_yaml(username, project_name, ["mask", "no_mask"])
    obj = ObjectDetection(username, project_name, modelname)
    # threading.Thread(target=obj.train, args=(epochs,)).start()
    on_successs = lambda: send_message_to_channel(
        f"{username}_{project_name}", {"training_status": "success"}
    )
    threading.Thread(
        target=wrap_async_func, args=(obj.train, epochs, on_successs)
    ).start()
    result = obj.train(epochs)
    return "running"


@app.post("/object/predict")
def object_detection_predict(
    bytefiles: List[UploadFile] = File(...),
    username: str = Form(...),
    project_name: str = Form(...),
    modelname: str = Form(...),
):
    # bytefile_data = [bf.file for bf in bytefiles]
    c = ObjectDetection(username, project_name, modelname)
    save_predict_img(bytefiles, username, project_name)
    result = c.predict()
    return result


@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    await websocket.accept()

    # Store the WebSocket connection in the dictionary
    channel_connections[channel] = websocket

    try:
        while True:
            # Receive message from the client
            data = await websocket.receive_text()

            # # Broadcast the message to all clients in the same channel
            # await send_message_to_channel(channel, f"Channel {channel}: {data}")
    except WebSocketDisconnect:
        # Remove the WebSocket connection when the client disconnects
        del channel_connections[channel]


async def broadcast(channel: str, message: str):
    # Send the message to all clients in the specified channel
    for connection in channel_connections.values():
        if connection:
            await connection.send_text(message)


async def send_message_to_channel(channel: str, message: Union[str, Dict[str, str]]):
    # Check if the channel exists in the dictionary
    print("message: ", message)
    if channel in channel_connections:
        connection = channel_connections[channel]

        # Check if the connection is open
        if connection:
            # Send the message to the specific channel
            if type(message) is str:
                await connection.send_text(message)
            else:
                await connection.send_json(message)
        else:
            print(f"Connection for channel {channel} is not open.")
    else:
        print(f"Channel {channel} not found.")


def wrap_async_func(func, *args):
    asyncio.run(func(*args))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
