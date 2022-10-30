"""Code that helps us test our neural network before deploying to the cloud."""

import logging
import json

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

from neural_network import NeuralNetwork
from utils_score_nn import predict

IMAGES_JSON = "aml_online_endpoint_no_mlflow/test_data/images.json"
MODEL_DIR = "aml_online_endpoint_no_mlflow/model/weights.pth"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODEL_DIR
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    with open(IMAGES_JSON, "rt", encoding="utf-8") as f:
        data = json.load(f)["data"]

    images_dataset = TensorDataset(torch.Tensor(data))

    dataloader = DataLoader(images_dataset)
    predicted_indices = predict(dataloader, model, device)
    predictions = [
        FashionMNIST.classes[predicted_index]
        for predicted_index in predicted_indices
    ]

    logging.info("Predictions: %s", predictions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()