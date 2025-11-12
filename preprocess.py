import os
import pandas as pd
from sklearn.datasets import load_iris

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = iris.target

output_path = os.path.join(DATA_DIR, "preprocessed.csv")
data.to_csv(output_path, index=False)
print(f"Data preprocessed and saved to {output_path}")
