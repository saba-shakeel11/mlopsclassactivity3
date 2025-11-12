import argparse
import json
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--metrics-path', type=str, default='metrics.json')
args = parser.parse_args()

data = pd.read_csv("data/preprocessed.csv")
X = data.drop('target', axis=1)
y = data['target']

with open(args.model, "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Model evaluation complete. Accuracy: {accuracy:.2f}")

metrics = {"accuracy": accuracy}
with open(args.metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f)

if accuracy < 0.9:
    raise SystemExit("Model accuracy below threshold. Failing pipeline.")
