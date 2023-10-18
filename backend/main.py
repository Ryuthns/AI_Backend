import uvicorn
from typing import Union, List

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from classification import TestClassification

app = FastAPI()

origins = [
   "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    c = TestClassification(10)
    result = c.predict(bytefile=file.file)
    print(result)
    return result

@app.post("/train/")
async def train_model(
    bytefiles: List[UploadFile] = File(...),
    labels: List[int] = Form(...),
    epochs: int = Form(...),
    lr: float = Form(...)
):
    # Convert UploadFile objects to BinaryIO
    bytefile_data = [bf.file for bf in bytefiles]
    
    # Call the train method with the received data
    c = TestClassification(10)
    result = c.train(bytefile_data, labels, epochs, lr)

    # Return the result as JSON
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)