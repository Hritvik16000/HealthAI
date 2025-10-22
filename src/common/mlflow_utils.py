import os

import mlflow
from dotenv import load_dotenv


def init_mlflow():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("healthai")
    return mlflow
