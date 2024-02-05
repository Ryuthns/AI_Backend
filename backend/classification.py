import io
import json
import os
from os import listdir, path
from typing import BinaryIO, Callable, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


def get_labels_dict(folder_path):
    labels_path = os.path.join(folder_path, "labels/classification.json")
    with open(labels_path, "r") as path:
        data = json.load(path)
    # labels_list = data.map(lambda x: x["annotations"][0])
    labels_list = [x["annotations"][0] for x in data]
    # Get unique labels
    unique_labels = list(set(labels_list))

    # Create a mapping from labels to numeric indices
    result = {label: index for index, label in enumerate(unique_labels)}
    return result


def read_json_and_get_image_bytes(json_file_path, images_folder_path):
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    image_bytes_list = []
    labels_list = []

    for item in data:
        image_filename = item["image"]
        image_path = path.join(images_folder_path, image_filename)

        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                image_bytes_list.append(image_bytes)
                labels_list.append(item["annotations"][0])
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")

    return image_bytes_list, labels_list


class TrainClassification:
    def __init__(
        self, num_classes: int, username: str, project_name: str, model_name: str
    ):
        self.username = username
        self.project_name = project_name
        self.model_name = model_name
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_model(self):
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
        )
        self.model.to(self.device)
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the last layer to have the number of output classes as specified
        in_features = self.model.classifier[
            1
        ].in_features  # Number of input features to the final layer
        self.model.classifier[1] = nn.Linear(
            in_features, self.num_classes
        )  # Replace the final layer

        for param in self.model.classifier[1].parameters():
            param.requires_grad = True

    def get_model_folder(self):
        folder_path = (
            f"user_project/{self.username}/{self.project_name}/models/{self.model_name}"
        )
        if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
            os.makedirs(folder_path)
        return folder_path

    def get_model_path(self):
        model_folder = self.get_model_folder()
        return f"{model_folder}/{self.model_name}.pth"

    def change_to_num_labels(self, labels):
        # Get unique labels
        unique_labels = list(set(labels))

        # Create a mapping from labels to numeric indices
        label_to_index = {label: index for index, label in enumerate(unique_labels)}

        # Convert original labels to numeric labels using the mapping
        numeric_labels = torch.tensor([label_to_index[label] for label in labels])
        return numeric_labels

    def get_label_path(self):
        path = f"{self.username}/{self.project_name}/labels/lebel.txt"
        return path

    async def train(self, epochs: int, lr: float, on_success=None):
        # if bytefiles is None or labels is None:
        #     return

        labels_path = f"user_project/{self.username}/{self.project_name}/labels/classification.json"
        images_path = f"user_project/{self.username}/{self.project_name}/images"
        bytefiles, labels = read_json_and_get_image_bytes(labels_path, images_path)
        self.num_classes = len(set(labels))
        self.prepare_model()

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        images = []
        for bf in bytefiles:
            image = Image.open(io.BytesIO(bf))
            input_tensor = preprocess(image)
            images.append(input_tensor)

        print(images[0].shape, labels)
        print("labels:", labels)
        print("n labels", self.change_to_num_labels(labels))
        labels = self.change_to_num_labels(labels)
        train_dataset = TensorDataset(torch.stack(images), torch.Tensor(labels).long())
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(train_loader, 0):
                inputs, label = data
                print("input: ", inputs.shape, label)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                print("output: ", outputs, label)
                loss = self.criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_losses.append(epoch_loss)

            epoch_accuracy = 100 * correct / total
            epoch_accuracies.append(epoch_accuracy)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")
            batch_result = {"loss": epoch_losses, "accuracy": epoch_accuracies}
            self._save_train_result(batch_result)

        filename = self.get_model_path()

        torch.save(self.model, filename)
        print("Finished Training and saved the model")

        result = {"loss": epoch_losses, "accuracy": epoch_accuracies}
        self._save_train_result(result)
        if on_success is not None:
            print("on success process")
            await on_success()
        return result

    def _save_train_result(self, result):
        model_folder = self.get_model_folder()
        result_path = f"{model_folder}/result.json"
        # Save to JSON file
        with open(result_path, "w") as json_file:
            json.dump(result, json_file)

    def _load_train_result(self):
        model_folder = self.get_model_folder()
        result_path = f"{model_folder}/result.json"
        if not os.path.exists(result_path):
            return {}
        # Load from JSON file
        with open(result_path, "r") as json_file:
            loaded_result = json.load(json_file)
        return loaded_result

    def _load_model(self):
        model_path = self.get_model_path()
        print("load models", model_path)
        if os.path.exists(model_path):
            self.model = torch.load(model_path)
            print("load models success!!")
            return
        if os.path.exists("mobilenet_v2_trained.pth"):
            self.model.load_state_dict(torch.load("mobilenet_v2_trained.pth"))

    def _preprocess_image(self, input_image):
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return preprocess(input_image).unsqueeze(0)

    def predict(self, bytefile: Union[List[BinaryIO], None] = None, filename=""):
        self._load_model()
        self.model.eval()

        if bytefile is None:
            return {}

        images = []
        for bf in bytefile:
            input_image = Image.open(io.BytesIO(bf.read()))
            input_tensor = self._preprocess_image(input_image)
            images.append(input_tensor)

        self.model.eval()
        print(images)

        if torch.cuda.is_available():
            self.model.to("cuda")

        torch.cat(images, dim=0)
        with torch.no_grad():
            output = self.model(torch.cat(images, dim=0))
            print("output", output)

        probabilities = torch.nn.functional.softmax(output, dim=0)
        if len(images) == 1:
            probabilities = torch.nn.functional.softmax(output)
        print(probabilities)
        predicted = torch.argmax(probabilities, dim=1).tolist()
        print("topk:", predicted)

        return {"probabilities": probabilities.tolist(), "predicted": predicted}

    def _get_top_classes(self, probabilities):
        categories = []
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        top5_prob, top5_catid = torch.topk(probabilities, 5)
        print(top5_catid, top5_prob)
        c = {}
        for i in range(top5_prob.size(0)):
            c[categories[top5_catid[i]]] = top5_prob[i].item()
        return c
