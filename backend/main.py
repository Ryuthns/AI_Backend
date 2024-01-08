import uvicorn
import os
from typing import Union, List, BinaryIO
from fastapi import FastAPI, File, Response, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from classification import TrainClassification
from helpers.helper import get_key_by_value
from models.ai_models import result_input

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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
