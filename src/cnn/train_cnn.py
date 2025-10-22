import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.mlflow_utils import init_mlflow


def run():
    mlflow = init_mlflow()
    X = np.random.rand(200, 1, 64, 64).astype("float32")
    y = (X.mean(axis=(2, 3)) > 0.5).astype("int64").reshape(-1)
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    mlflow.start_run(run_name="cnn")
    for epoch in range(5):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    mlflow.log_metric("epochs", 5)
    mlflow.end_run()


if __name__ == "__main__":
    run()
