from pathlib import Path

import pandas as pd

p = Path("artifacts/association_rules.csv")
if p.exists():
    df = pd.read_csv(p)
    df.to_html("artifacts/association_rules.html", index=False)
