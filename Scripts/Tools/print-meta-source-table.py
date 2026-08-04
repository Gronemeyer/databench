
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from databench.config import resolve_dataset
from databench.project import Project


dataset = resolve_dataset("etoh-hfsa")

proj = Project(dataset=dataset).filter(exclude={"Session": ["ses-00", "ses-11"], "Task": "task-movies"})

pt_s = proj.df[("session_config", "PT_s")].explode().reset_index(name="PT_s")
pt_s["PT_s"] = pd.to_numeric(pt_s["PT_s"], errors="coerce")

# Group by Subject and calculate mean PT_s across sessions
pt_s_avg = pt_s.groupby("Subject")["PT_s"].mean().sort_values(ascending=False)
print(pt_s_avg)

pt_s_avg.plot(kind="bar", title="Average PT_s Time by Subject", figsize=(10, 5))
plt.ylabel("PT_s (seconds)")
plt.tight_layout()
plt.show()

weight = proj.df[("session_config", "weight")].explode().reset_index(name="weight")
weight["weight"] = pd.to_numeric(weight["weight"], errors="coerce")
weight["session_n"] = pd.to_numeric(
    weight["Session"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
)

weight_by_session = (
    weight.dropna(subset=["weight", "session_n"])
    .groupby(["Subject", "session_n"], as_index=False)["weight"]
    .mean()
)

weight_pivot = (
    weight_by_session
    .pivot(index="session_n", columns="Subject", values="weight")
    .sort_index()
)
print(weight_pivot)

weight_pivot.plot(kind="line", marker="o", figsize=(10, 6), title="session_config.weight by Subject Across Sessions")
plt.xlabel("Session")
plt.ylabel("Weight")
plt.tight_layout()
plt.show()
