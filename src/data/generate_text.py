from pathlib import Path
import numpy as np
import pandas as pd
from src.common.mlflow_utils import init_mlflow

def synth_feedback(n=2000, seed=21):
    rng = np.random.default_rng(seed)
    pos = [
        "Nurse was attentive and kind.",
        "Medication explained clearly by the doctor.",
        "Waiting time was acceptable.",
        "The ward was clean and quiet.",
        "Follow-up instructions were helpful."
    ]
    neg = [
        "Had to wait too long for assistance.",
        "Medication side effects were not explained.",
        "Room was noisy at night.",
        "Discharge process felt rushed.",
        "Staff seemed overwhelmed and unresponsive."
    ]
    rows = []
    for i in range(n):
        if rng.random() < 0.5:
            txt = rng.choice(pos)
            label = "positive"
        else:
            txt = rng.choice(neg)
            label = "negative"
        rows.append((i+1, txt, label))
    return pd.DataFrame(rows, columns=["id","text","label"])

def synth_notes(n=2000, seed=22):
    rng = np.random.default_rng(seed)
    templates = [
        "Patient reports {symptom} for {days} days. Vitals stable. Recommend {plan}.",
        "Complains of {symptom}. No fever. Start {plan} and monitor.",
        "{symptom} noted post-op day {days}. Continue {plan}."
    ]
    symptoms = ["chest pain","shortness of breath","dizziness","nausea","cough","fatigue"]
    plans = ["analgesics","antibiotics","physiotherapy","hydration","rest","inhaler"]
    rows = []
    for i in range(n):
        t = rng.choice(templates)
        s = rng.choice(symptoms)
        p = rng.choice(plans)
        d = int(rng.integers(1,7))
        txt = t.format(symptom=s, days=d, plan=p)
        rows.append((i+1, txt))
    return pd.DataFrame(rows, columns=["id","note"])

def main():
    mlflow = init_mlflow()
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    fb = synth_feedback()
    notes = synth_notes()
    fb.to_csv(raw / "patient_feedback.csv", index=False)
    notes.to_csv(raw / "doctor_notes.csv", index=False)
    mlflow.start_run(run_name="data_generate_text")
    mlflow.log_artifact(str(raw / "patient_feedback.csv"))
    mlflow.log_artifact(str(raw / "doctor_notes.csv"))
    mlflow.end_run()

if __name__ == "__main__":
    main()
