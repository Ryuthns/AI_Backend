import os
from PIL import Image
from torchvision import transforms
import torch
import io
from typing import BinaryIO, Tuple, Union, List
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


class TrainClassification:
    def __init__(self, num_classes: int):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the last layer to have the number of output classes as specified
        in_features = self.model.classifier[1].in_features  # Number of input features to the final layer
        self.model.classifier[1] = nn.Linear(in_features, num_classes)  # Replace the final layer

        for param in self.model.classifier[1].parameters():
            param.requires_grad = True


    def train(self, bytefiles:Union[List[BinaryIO], None], labels: List[int], epochs:int, lr:float):
        if bytefiles is None or labels is None:
            return

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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

        torch.save(self.model.state_dict(), 'mobilenet_v2_trained.pth')
        print("Finished Training and saved the model")

        return {'loss': epoch_losses, 'accuracy': epoch_accuracies}


    def _load_model(self):
        if os.path.exists('mobilenet_v2_trained.pth'):
            self.model.load_state_dict(torch.load('mobilenet_v2_trained.pth'))

    def _preprocess_image(self, input_image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(input_image).unsqueeze(0)

    def predict(self, bytefile: Union[BinaryIO, None]=None, filename=""):
        self._load_model()
        
        input_image = None
        if bytefile is not None:
            input_image = Image.open(io.BytesIO(bytefile.read()))
        elif filename:
            input_image = Image.open(filename)

        if input_image is None:
            return {}

        self.model.eval()

        input_batch = self._preprocess_image(input_image)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities[:5])

        if not os.path.exists('mobilenet_v2_trained.pth'):
            return self._get_top_classes(probabilities)
        
        return {idx: prob.item() for idx, prob in enumerate(probabilities)}

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

class TestClassification:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # Freeze all layers

    def _load_model(self):
        if os.path.exists('mobilenet_v2_trained.pth'):
            self.model.load_state_dict(torch.load('mobilenet_v2_trained.pth'))

    def _preprocess_image(self, input_image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(input_image).unsqueeze(0)

    def predict(self, bytefile: Union[BinaryIO, None]=None, filename=""):
        self._load_model()
        
        input_image = None
        if bytefile is not None:
            input_image = Image.open(io.BytesIO(bytefile.read()))
        elif filename:
            input_image = Image.open(filename)

        if input_image is None:
            return {}

        self.model.eval()

        input_batch = self._preprocess_image(input_image)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities[:5])

        if not os.path.exists('mobilenet_v2_trained.pth'):
            return self._get_top_classes(probabilities)
        
        return {idx: prob.item() for idx, prob in enumerate(probabilities)}

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

def load_images_from_folder(folder_path: str) -> (List[BinaryIO], List[int]):
    bytefiles = []
    labels = []
    label_mapping = {'Cat': 0, 'Dog': 1}

    for label_name, label_idx in label_mapping.items():
        dir_path = os.path.join(folder_path, label_name)
        for filename in os.listdir(dir_path):
            if filename.endswith('.png'):
                img_path = os.path.join(dir_path, filename)
                with open(img_path, 'rb') as f:
                    byte_content = f.read()
                    bytefiles.append(io.BytesIO(byte_content))
                    labels.append(label_idx)
    
    return bytefiles, labels

if __name__ == "__main__":
    folder_path = "Dog and Cat .png"
    bytefiles, labels = load_images_from_folder(folder_path)
    #
    classifier = TestClassification(2)
    result = classifier.train(bytefiles, labels, epochs=3, lr=0.001)
    predicted = classifier.predict(filename='./images/images.jpg')
    # print(result)
    print(predicted)