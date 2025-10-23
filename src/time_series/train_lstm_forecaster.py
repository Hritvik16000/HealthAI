import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.common.mlflow_utils import init_mlflow

def build_dataset(df, window=24):
    g = df.groupby("patient_id")
    Xs, ys = [], []
    for _, sub in g:
        v = sub.sort_values("t")["heart_rate"].to_numpy(dtype=np.float32)
        for i in range(len(v)-window-1):
            Xs.append(v[i:i+window])
            ys.append(v[i+window])
    X = torch.tensor(np.array(Xs))[:, :, None]
    y = torch.tensor(np.array(ys))
    return X, y

def main():
    mlflow = init_mlflow()
    df = pd.read_csv("data/raw/vitals_timeseries.csv")
    X, y = build_dataset(df, window=24)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
    head = nn.Linear(32,1)
    opt = torch.optim.Adam(list(lstm.parameters())+list(head.parameters()), lr=1e-3)
    lossf = nn.MSELoss()

    mlflow.start_run(run_name="lstm_forecaster")
    for _ in range(8):
        for xb,yb in dl:
            opt.zero_grad()
            out,_ = lstm(xb)
            pred = head(out[:,-1,:]).squeeze()
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()
    out_dir = Path("artifacts/timeseries"); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"lstm": lstm.state_dict(), "head": head.state_dict()}, out_dir/"lstm_hr.pt")
    with open(out_dir/"meta.json","w") as f: json.dump({"window":24}, f, indent=2)
    mlflow.log_artifact(str(out_dir/"lstm_hr.pt"))
    mlflow.log_artifact(str(out_dir/"meta.json"))
    mlflow.end_run()

if __name__=="__main__":
    main()
