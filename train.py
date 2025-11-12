import pandas as pd
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()

data = pd.read_csv("data/preprocessed.csv")
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model_path = f"model_v{args.version}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model trained and saved as {model_path}")
