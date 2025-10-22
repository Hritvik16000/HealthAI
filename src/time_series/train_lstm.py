import numpy as np
import torch
from torch import nn

from src.common.mlflow_utils import init_mlflow


def make_series(n=500):
    t = np.arange(n)
    s = np.sin(0.02 * t) + 0.1 * np.random.randn(n)
    return s.astype("float32")


def run():
    mlflow = init_mlflow()
    s = make_series()
    X = []
    y = []
    w = 20
    for i in range(len(s) - w - 1):
        X.append(s[i : i + w])
        y.append(s[i + w])
    X = torch.tensor(np.array(X)).unsqueeze(-1)
    y = torch.tensor(np.array(y))
    model = nn.Sequential(
        nn.LSTM(input_size=1, hidden_size=16, batch_first=True), nn.Flatten(), nn.Linear(16, 1)
    )
    lstm = model[0]
    head = model[2]
    opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=1e-3)
    lossf = nn.MSELoss()
    mlflow.start_run(run_name="lstm")
    for _ in range(10):
        opt.zero_grad()
        out, _ = lstm(X)
        pred = head(out[:, -1, :]).squeeze()
        loss = lossf(pred, y)
        loss.backward()
        opt.step()
    mlflow.log_metric("epochs", 10)
    mlflow.end_run()


if __name__ == "__main__":
    run()
