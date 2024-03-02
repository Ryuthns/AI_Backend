import io
import json
import os
import numpy as np
from os import listdir, path
from typing import BinaryIO, Callable, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from helpers.model import load_metadata, save_metadata


def preprocess_image(input_image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(input_image).unsqueeze(0)


def plot_confusion_matrix(confusion_mat, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_mat.T,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Predicted")
    plt.xlabel("True")
    plt.savefig(save_path)


def change_to_class_names(numeric_labels, unique_labels):
    # Create a reverse mapping from numeric indices to labels
    index_to_label = {index: label for index, label in enumerate(unique_labels)}

    # Convert numeric labels to class names using the reverse mapping
    class_names = [index_to_label[index.item()] for index in numeric_labels]
    return class_names


def change_to_num_labels(labels):
    # Get unique labels
    unique_labels = sorted(list(set(labels)))

    # Create a mapping from labels to numeric indices
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    # Convert original labels to numeric labels using the mapping
    numeric_labels = torch.tensor([label_to_index[label] for label in labels])
    return unique_labels, numeric_labels


def get_labels_dict(folder_path):
    labels_path = os.path.join(folder_path, "labels/classification.json")
    with open(labels_path, "r") as path:
        data = json.load(path)
    # labels_list = data.map(lambda x: x["annotations"][0])
    labels_list = [x["annotations"][0] for x in data]
    # Get unique labels
    unique_labels = sorted(list(set(labels_list)))

    # Create a mapping from labels to numeric indices
    result = {label: index for index, label in enumerate(unique_labels)}
    return result


def read_json_and_get_image_bytes(
    json_file_path, images_folder_path, get_image_name=False
):
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    image_bytes_list = []
    labels_list = []
    image_name = []

    for item in data:
        image_filename = item["image"]
        if get_image_name:
            image_name.append(image_filename)
        image_path = path.join(images_folder_path, image_filename)

        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                image_bytes_list.append(image_bytes)
                labels_list.append(item["annotations"][0])
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")

    if get_image_name:
        return [image_bytes_list, image_name], labels_list

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
        unique_labels = sorted(list(set(labels)))

        # Create a mapping from labels to numeric indices
        label_to_index = {label: index for index, label in enumerate(unique_labels)}

        # Convert original labels to numeric labels using the mapping
        numeric_labels = torch.tensor([label_to_index[label] for label in labels])
        return unique_labels, numeric_labels

    def get_label_path(self):
        path = f"{self.username}/{self.project_name}/labels/lebel.txt"
        return path

    async def train(
        self,
        epochs: int,
        lr: float,
        on_success=None,
        train_ratio=0.8,
        validate_ratio=0.1,
        test_ratio=0.1,
        on_epoch_end=None,
    ):
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
            if image.mode != "RGB":
                image = image.convert("RGB")
            input_tensor = preprocess(image)
            images.append(input_tensor)

        classname, labels = self.change_to_num_labels(labels)

        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=validate_ratio + test_ratio, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio, random_state=42
        )

        # Create DataLoader for training set
        train_dataset = TensorDataset(
            torch.stack(X_train), torch.Tensor(y_train).long()
        )
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # Create DataLoader for validation set
        val_dataset = TensorDataset(torch.stack(X_val), torch.Tensor(y_val).long())
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Create DataLoader for test set
        test_dataset = TensorDataset(torch.stack(X_test), torch.Tensor(y_test).long())
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        epoch_losses = []
        epoch_accuracies = []
        epoch_precision = []
        epoch_recall = []
        epoch_val_losses = []
        epoch_val_accuracies = []
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            true_labels = []
            predicted_labels = []

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

                # Collect true and predicted labels for precision and recall calculation
                true_labels.extend(label.numpy())
                predicted_labels.extend(predicted.numpy())

            # Calculate precision and recall
            precision = precision_score(
                true_labels, predicted_labels, average="weighted"
            )
            recall = recall_score(true_labels, predicted_labels, average="weighted")
            epoch_precision.append(precision)
            epoch_recall.append(recall)

            epoch_loss = running_loss / len(train_loader)
            epoch_losses.append(epoch_loss)

            epoch_accuracy = 100 * correct / total
            epoch_accuracies.append(epoch_accuracy)

            # Validation loop
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_label = val_data
                    val_outputs = self.model(val_inputs)
                    val_loss = self.criterion(val_outputs, val_label)

                    val_running_loss += val_loss.item()

                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_label.size(0)
                    val_correct += (val_predicted == val_label).sum().item()

            # Calculate validation loss
            val_epoch_loss = val_running_loss / len(val_loader)
            val_epoch_accuracy = 100 * val_correct / val_total
            epoch_val_losses.append(val_epoch_loss)
            epoch_val_accuracies.append(val_epoch_accuracy)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")
            batch_result = {
                "loss": epoch_losses,
                "accuracy": epoch_accuracies,
                "precision": epoch_precision,
                "recall": epoch_recall,
                "val_loss": epoch_val_losses,
                "val_accuracy": epoch_val_accuracies,
            }
            self._save_train_result(batch_result)

            # call function on_epoch_end
            if on_epoch_end is not None:
                await on_epoch_end(epochs, epoch)

        filename = self.get_model_path()

        torch.save(self.model, filename)
        print("Finished Training and saved the model")

        result = {
            "loss": epoch_losses,
            "accuracy": epoch_accuracies,
            "precision": epoch_precision,
            "recall": epoch_recall,
            "val_loss": epoch_val_losses,
            "val_accuracy": epoch_val_accuracies,
        }
        self._save_train_result(result)
        self.calculate_summalize(val_loader, classname)
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
            self.model.eval()
            print("load models success!!")
            return
        if os.path.exists("mobilenet_v2_trained.pth"):
            self.model.load_state_dict(torch.load("mobilenet_v2_trained.pth"))

    def calculate_summalize(self, validation_loader, classname):
        # Evaluate the model
        if not self.model:
            self._load_model()
        self.model.eval()

        all_labels = []
        all_predictions = []
        all_predictions2d = []

        with torch.no_grad():
            for data in validation_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                all_predictions2d.extend(outputs.numpy())

                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.numpy())

        # Ensure the lengths match
        print("Length of labels:", all_labels)
        print("Length of predictions:", all_predictions)

        # Convert to NumPy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_predictions2d = np.array(all_predictions2d)

        # Ensure labels are numeric
        all_labels = all_labels.astype(int)
        all_predictions = all_predictions.astype(int)
        all_predictions2d = all_predictions2d.astype(float)

        # Calculate precision, recall, and average precision
        precision = precision_score(all_labels, all_predictions, average="weighted")
        recall = recall_score(all_labels, all_predictions, average="weighted")
        print("Precision:", precision, "Recall:", recall)

        print("wtf" * 20)
        # Calculate average precision
        if len(all_labels) == 1:
            all_predictions2d = all_predictions
        print(all_labels.shape, all_predictions2d.shape, all_predictions.shape)
        average_precision = average_precision_score(
            all_labels, all_predictions, average="macro"
        )
        print("wtf" * 20)
        #
        # Compute confusion matrix
        confusion_mat = confusion_matrix(all_labels, all_predictions)

        # Plot and save confusion matrix
        model_path = self.get_model_folder()
        save_path = os.path.join(model_path, "confusion_matrix.png")
        print("confusion path:", save_path)
        print("*" * 30)
        plot_confusion_matrix(confusion_mat, classname, save_path)

        # # Calculate average precision
        # average_precision = average_precision_score(all_labels, all_predictions)

        path = self.get_model_folder()
        data = load_metadata(path)
        data["average_precision"] = average_precision
        data["precision"] = precision
        data["recall"] = recall
        save_metadata(path, data)

        print("-" * 20)
        print("Summarize success")
        print(data)
        print("-" * 20)

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
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")
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

        prob_with_classname = []
        path = os.path.join("user_project", self.username, self.project_name)
        classname = get_labels_dict(path).keys()
        probabilities = probabilities.tolist()

        for p in probabilities:
            output = {c: p[i] for i, c in enumerate(classname)}
            prob_with_classname.append(output)
        return {
            "probabilities": probabilities,
            "predicted": predicted,
            "prob_with_classname": prob_with_classname,
        }

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
