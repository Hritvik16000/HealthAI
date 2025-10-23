import json, pickle, csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.common.mlflow_utils import init_mlflow

class ImgDS(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.labels = {"normal":0,"pneumonia":1}
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, lab = self.rows[i]
        img = Image.open(p).convert("L").resize((128,128))
        x = np.array(img, dtype=np.float32)/255.0
        x = x[None, ...]
        y = self.labels[lab]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

def split_rows(rows, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n = len(rows)
    tr = rows[idx[:int(0.7*n)]]
    va = rows[idx[int(0.7*n):int(0.85*n)]]
    te = rows[idx[int(0.85*n):]]
    return tr, va, te

def main():
    mlflow = init_mlflow()
    with open("data/raw/image_labels.csv") as f:
        r = list(csv.DictReader(f))
    rows = np.array([(ri["path"], ri["label"]) for ri in r])
    tr, va, te = split_rows(rows)
    dl_tr = DataLoader(ImgDS(tr), batch_size=32, shuffle=True)
    dl_va = DataLoader(ImgDS(va), batch_size=64, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64,2)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    mlflow.start_run(run_name="cnn_pneumonia")
    for epoch in range(5):
        model.train()
        for xb,yb in dl_tr:
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    acc = 0; n = 0
    with torch.no_grad():
        for xb,yb in dl_va:
            yp = model(xb).argmax(1)
            acc += (yp==yb).sum().item()
            n += yb.numel()
    val_acc = acc/n
    out = Path("artifacts/cnn"); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out/"pneumonia_cnn.pt")
    with open(out/"metrics_cnn.json","w") as f: json.dump({"val_accuracy": float(val_acc)}, f, indent=2)
    mlflow.log_metric("cnn_val_accuracy", float(val_acc))
    mlflow.log_artifact(str(out/"pneumonia_cnn.pt"))
    mlflow.log_artifact(str(out/"metrics_cnn.json"))
    mlflow.end_run()

if __name__=="__main__":
    main()
