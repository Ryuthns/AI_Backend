# Load library
import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clustimage import Clustimage
from PIL import Image

from classification import (
    change_to_class_names,
    change_to_num_labels,
    read_json_and_get_image_bytes,
)

from torchvision import transforms

#
# Turn interactive plotting off
plt.ioff()


def preprocess_image(input_image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
        ]
    )
    # Apply the transformations
    preprocessed_image = preprocess(input_image)

    # Convert the preprocessed image to a NumPy array and flatten it
    preprocessed_array = np.ravel(np.array(preprocessed_image))

    return preprocessed_array


def cluster_result(image_path, label_path, save_path=None):
    byte_files, labels_name = read_json_and_get_image_bytes(
        label_path, image_path, get_image_name=True
    )
    byte_files, images_name = byte_files

    # Convert the list of byte files to a NumPy array
    images = [np.frombuffer(byte_file, dtype=np.uint8) for byte_file in byte_files]
    images = []
    for bf in byte_files:
        input_image = Image.open(io.BytesIO(bf))
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        input_tensor = preprocess_image(input_image)
        images.append(input_tensor)
    print(len(images))
    images = np.array(images)
    print(images.shape)
    # images = np.array(images)
    unique_labels, labels = change_to_num_labels(labels_name)
    X = pd.DataFrame(images, index=images_name)
    print(X.head())
    #
    cl = Clustimage()
    results = cl.fit_transform(images, min_clust=2, max_clust=len(unique_labels))
    actual_labels = change_to_class_names(results["labels"], unique_labels)
    results["true_labels"] = labels_name
    results["labels"] = actual_labels
    cl.plot_unique()
    if save_path is not None:
        unique_path = os.path.join(save_path, "unique.png")
        plt.savefig(unique_path)
    save_result = {}
    save_result["filename"] = results["filenames"].tolist()
    save_result["actual_labels"] = results["labels"]
    save_result["true_labels"] = results["true_labels"]
    save_result["xycoord"] = results["xycoord"].tolist()
    print(save_result["actual_labels"])
    result_path = os.path.join(save_path, "cluster.json")
    with open(result_path, "w") as f:
        json.dump(save_result, f, indent=4)
    # plt.show()
    return save_result


if __name__ == "__main__":
    result = cluster_result(
        "/home/dsy/coding/project-year-4/AI_Backend/backend/user_project/ong2/ball/images",
        "/home/dsy/coding/project-year-4/AI_Backend/backend/user_project/ong2/ball/labels/classification.json",
        "",
    )
