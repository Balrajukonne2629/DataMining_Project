from pathlib import Path

from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
STATIC_DIR = BASE_DIR / "static"

SUBJECTS = ["DBMS", "DM", "EAD", "EEA", "ES", "PQT"]
RISK_THRESHOLD = 75

data = pd.read_csv(DATASET_PATH)

feature_columns = [f"{subject}_{kind}" for subject in SUBJECTS for kind in ("Present", "Absent")]
missing_columns = [column for column in feature_columns + ["Attended_Classes", "Absentee_Count", "Risk_Label"] if column not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required dataset columns: {', '.join(missing_columns)}")

def generate_graphs():
    attendance = pd.Series(
        {
            subject: ((data[f"{subject}_Present"] / (data[f"{subject}_Present"] + data[f"{subject}_Absent"])) * 100).mean().round(2)
            for subject in SUBJECTS
        }
    )
    absent = pd.Series(
        {
            subject: ((data[f"{subject}_Absent"] / (data[f"{subject}_Present"] + data[f"{subject}_Absent"])) * 100).mean().round(2)
            for subject in SUBJECTS
        }
    )

    STATIC_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = attendance.plot(kind="bar", color="#2e8b57", ax=ax)
    ax.set_title("Subject-wise Attendance", fontsize=14, fontweight="bold", color="#111827")
    ax.set_ylabel("Average Attendance (%)", fontsize=11, fontweight="bold", color="#111827")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=10, colors="#111827", rotation=0)
    ax.tick_params(axis="y", labelsize=10, colors="#111827")
    ax.grid(axis="y", alpha=0.25)
    for container in bars.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9, color="#111827", fontweight="bold")
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "attendance.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = absent.plot(kind="bar", color="#c0392b", ax=ax)
    ax.set_title("Subject-wise Absentees", fontsize=14, fontweight="bold", color="#111827")
    ax.set_ylabel("Average Absence (%)", fontsize=11, fontweight="bold", color="#111827")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=10, colors="#111827", rotation=0)
    ax.tick_params(axis="y", labelsize=10, colors="#111827")
    ax.grid(axis="y", alpha=0.25)
    for container in bars.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9, color="#111827", fontweight="bold")
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "absent.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        attendance,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2563eb", "#0ea5e9", "#14b8a6", "#22c55e", "#84cc16", "#f59e0b"],
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 10, "fontweight": "bold", "color": "#111827"},
    )
    for text in autotexts:
        text.set_color("#111827")
        text.set_fontsize(10)
        text.set_fontweight("bold")
    ax.legend(wedges, SUBJECTS, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_title("Attendance Share by Subject", fontsize=14, fontweight="bold", color="#111827")
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "attendance_pie.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        absent,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ef4444", "#f97316", "#f59e0b", "#eab308", "#f43f5e", "#fb7185"],
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 10, "fontweight": "bold", "color": "#111827"},
    )
    for text in autotexts:
        text.set_color("#111827")
        text.set_fontsize(10)
        text.set_fontweight("bold")
    ax.legend(wedges, SUBJECTS, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_title("Absentee Share by Subject", fontsize=14, fontweight="bold", color="#111827")
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "absent_pie.png", dpi=160)
    plt.close()

generate_graphs()

X = data[feature_columns + ["Attended_Classes", "Absentee_Count"]]
y = data["Risk_Label"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    feature_values = []
    percent_map = {}
    weak = []

    total_present = 0
    total_absent = 0

    for sub in SUBJECTS:
        present = int(request.form[sub+"_present"])
        absent = int(request.form[sub+"_absent"])
        total = present + absent

        percent = (present/total)*100 if total!=0 else 0
        percent_map[sub] = round(percent, 2)
        feature_values.extend([present, absent])

        total_present += present
        total_absent += absent

        if percent < RISK_THRESHOLD:
            weak.append(sub)

    sample = pd.DataFrame([feature_values + [total_present, total_absent]], columns=X.columns)
    pred = model.predict(sample)[0]

    risk_state = "at_risk" if weak else "regular"
    result = "At Risk" if risk_state == "at_risk" else "Regular"
    model_result = "At Risk" if pred == 1 else "Regular"
    graph_files = ["attendance.png", "absent.png", "attendance_pie.png", "absent_pie.png"]
    graph_version = int(max((STATIC_DIR / name).stat().st_mtime for name in graph_files))

    return render_template(
        "result.html",
        result=result,
        risk_state=risk_state,
        model_result=model_result,
        weak=weak,
        percentages=percent_map,
        total_present=total_present,
        total_absent=total_absent,
        graph_version=graph_version,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)