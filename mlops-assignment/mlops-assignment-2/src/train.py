import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Loading dataset...")
df = pd.read_csv("data/dataset.csv")

X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = LogisticRegression()
model.fit(X_train, y_train)

print("Saving model to models/model.pkl...")
with open("models/model.pkl", "wb") as f: pickle.dump(model, f)

print("Done!")