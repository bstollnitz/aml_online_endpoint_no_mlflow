"""Prediction."""

import json
import logging
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

from .neural_network import NeuralNetwork
from .utils_score_nn import predict

model = None
device = None
BATCH_SIZE = 64


def init() -> None:
    logging.info("Init started")

    global model
    global device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Device: %s", device)

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", default=""),
                              "model/weights.pth")

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    logging.info("Init completed")


def run(raw_data: str) -> list[str]:
    logging.info("Run started")

    json_list = json.loads(raw_data)["input_data"]["data"]
    x = DataLoader(TensorDataset(torch.Tensor(json_list)),
                   batch_size=BATCH_SIZE)

    predicted_indices = predict(x, model, device)
    predictions = [
        FashionMNIST.classes[predicted_index]
        for predicted_index in predicted_indices
    ]

    logging.info("Predictions: %s", predictions)

    logging.info("Run completed")
    return predictions
