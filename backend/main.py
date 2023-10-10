from typing import Union

from fastapi import FastAPI, File, UploadFile
from classification import TestClassification

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    c = TestClassification()
    result = c.predict(bytefile=file.file)
    return result
