import os
import pathlib
import shutil
from typing import List
import json

import pandas as pd
from ultralytics import YOLO


def _on_train_epoch_end(pred, username, project_name, model_name):
    loss = pred.loss.item()
    print("on_train_epoch_end loss", pred.loss)
    fitness = pred.fitness
    path = get_model_folder(username, project_name, model_name)
    result_path = os.path.join(path, "result.json")
    if fitness is None:
        fitness = 0
    if os.path.exists(result_path):
        with open(result_path, "r+") as file:
            data = json.load(file)  # Load existing data if the file already exists

            data["accuracy"].append(fitness)  # Append the new fitness value
            data["loss"].append(loss)  # Append the new loss value

            file.seek(0)  # Move to the beginning of the file
            json.dump(data, file)
    else:
        data = {"loss": [], "accuracy": []}
        data["accuracy"].append(fitness)
        data["loss"].append(loss)
        _save_train_result(data, result_path)
    return


def _save_train_result(result, path):
    # Save to JSON file
    with open(path, "w") as json_file:
        json.dump(result, json_file)


def get_model_folder(username, project_name, model_name):
    current_path = pathlib.Path().resolve()
    folder_path = f"user_project/{username}/{project_name}/models/{model_name}"
    folder_path = os.path.join(current_path, folder_path)
    if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
        os.makedirs(folder_path)
    return folder_path


def save_image(file, folder_path):
    if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
        os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


def save_predict_img(images, username, project_name):
    current_path = pathlib.Path().resolve()
    w_path = os.path.join(current_path, "user_project", username, project_name)
    predict_path = os.path.join(w_path, "predict")
    if os.path.exists(predict_path):
        shutil.rmtree(predict_path)
    for im in images:
        print(im.__dir__())
        save_image(im, predict_path)


def prepare_yaml(username, project_name, class_list: List[str]):
    """
    Prepare yaml file for training

    @param username: str
    @param project_name: str
    @param class_list: unique classes in dataset
    """
    current_path = pathlib.Path().resolve()
    w_path = os.path.join(current_path, "user_project", username, project_name)
    data_path = os.path.join(w_path, "data.yaml")
    with open(data_path, "w") as file:
        file.write(f"train: {os.path.join(w_path)}\n")
        file.write(f"val: {os.path.join(w_path)}\n")
        file.write("\n")
        file.write(f"nc: {len(class_list)}\n")
        file.write(f"names: [{','.join(class_list)}]\n")


class ObjectDetection:
    def __init__(self, username: str, project_name: str, model_name: str) -> None:
        self.username = username
        self.project_name = project_name
        self.model_name = model_name

    def working_path(self):
        current_path = pathlib.Path().resolve()
        path = os.path.join(
            current_path, "user_project", self.username, self.project_name
        )
        return path

    def remove_old_model(self):
        path = os.path.join(self.working_path(), "models", self.model_name)
        if not (os.path.exists(path) and os.path.isdir(path)):
            return
        shutil.rmtree(path)
        print(f"Folder '{path}' removed successfully.")

    def read_result(self):
        result_path = os.path.join(
            self.working_path(), "models", self.model_name, "results.csv"
        )
        df = pd.read_csv(result_path)
        df.columns = df.columns.str.strip()
        return df

    async def train(self, epoch: int = 20, on_success=None):
        model = YOLO("yolov8n.pt")

        model.add_callback(
            "on_train_epoch_end",
            lambda pred: _on_train_epoch_end(
                pred, self.username, self.project_name, self.model_name
            ),
        )
        self.remove_old_model()
        w_path = self.working_path()
        model_name = "models/" + self.model_name
        yaml_path = os.path.join(w_path, "data.yaml")
        print(yaml_path)
        train_result = model.train(
            data=yaml_path,
            epochs=epoch,
            imgsz=224,
            project=w_path,
            name=model_name,
        )

        model.export(format="onnx")

        df = self.read_result()
        result = {}
        result["loss"] = df["train/box_loss"].to_list()
        if on_success is not None:
            print("on success process")
            await on_success()
        print("-" * 20)
        print("training success")
        print("-" * 20)
        return result

    def predict(self):
        model_path = os.path.join(
            self.working_path(), "models", self.model_name, "weights", "best.pt"
        )
        model = YOLO(model_path)

        predict_path = os.path.join(self.working_path(), "predict")
        results = model.predict(source=predict_path)
        print("-" * 20)
        print("model path:", model_path)
        print("model info", model.info())
        print("-" * 20)
        output = []
        for r in results:
            o = {}
            o["boxes"] = r.boxes.xyxy.tolist()
            o["classes"] = r.boxes.cls.tolist()
            o["path"] = r.path.split("/")[-1]
            output.append(o)
            print("-" * 20)
            print("output:", o)
            print("box:", r.boxes)
            print("-" * 20)
        return output


if __name__ == "__main__":
    obj = ObjectDetection("user2", "project2", "model1")
    # prepare_yaml("user2", "project2", ["no_mask", "mask"])
    # result = obj.train(epoch=50)
    # print(result)
    results = obj.predict()
    print(results)
