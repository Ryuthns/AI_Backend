import os
import pathlib
import shutil
from typing import List
import pandas as pd

from ultralytics import YOLO


def prepare_yaml(username, project_name, class_list: List[str]):
    """
    Prepare yaml file for training

    @param username: str
    @param project_name: str
    @param class_list: list
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

    def train(self, epoch: int = 20):
        model = YOLO("yolov8n.pt")

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
        return result


if __name__ == "__main__":
    obj = ObjectDetection("user2", "project1", "model2")
    prepare_yaml("user2", "project1", ["box", "person", "table"])
    result = obj.train(epoch=10)
    print(result)
