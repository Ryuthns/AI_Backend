import os

# from models.ai_models import MetadataModel
import json


def save_metadata(path: str, metadata):
    # metadata = metadata.dict()
    path = os.path.join(path, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f)
    return True


def load_metadata(path: str):
    path = os.path.join(path, "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    metadata = {"train_image": 100, "test_image": 200, "validate_image": 20}
    save_metadata("test.json", metadata)
    data = load_metadata("test.json")
    print(data)
    print(data["train_image"])
