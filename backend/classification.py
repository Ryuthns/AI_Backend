import os
from PIL import Image
from pydantic_core.core_schema import model_field
from torchvision import transforms
import torch
import io
from typing import BinaryIO, Tuple, Union, List
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


class TrainClassification:
    def __init__(
        self, num_classes: int, username: str, project_name: str, model_name: str
    ):
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
        )
        self.username = username
        self.project_name = project_name
        self.model_name = model_name

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the last layer to have the number of output classes as specified
        in_features = self.model.classifier[
            1
        ].in_features  # Number of input features to the final layer
        self.model.classifier[1] = nn.Linear(
            in_features, num_classes
        )  # Replace the final layer

        for param in self.model.classifier[1].parameters():
            param.requires_grad = True

    def check_model_folder(self):
        folder_path = f"user_project/{self.username}/{self.project_name}/models"
        if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
            os.makedirs(folder_path)
        return folder_path

    def get_model_path(self):
        model_folder = self.check_model_folder()
        return f"{model_folder}/{self.model_name}.pth"

    def train(
        self,
        bytefiles: Union[List[BinaryIO], None],
        labels: List[int],
        epochs: int,
        lr: float,
    ):
        if bytefiles is None or labels is None:
            return

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
            image = Image.open(io.BytesIO(bf.read()))
            input_tensor = preprocess(image)
            images.append(input_tensor)

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
                optimizer.zero_grad()
                outputs = self.model(inputs)
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

        filename = self.get_model_path()

        torch.save(self.model.state_dict(), filename)
        print("Finished Training and saved the model")

        return {"loss": epoch_losses, "accuracy": epoch_accuracies}

    def _load_model(self):
        model_path = self.get_model_path()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
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
