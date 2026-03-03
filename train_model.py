import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset
data = {
    "study_hours": [5,2,7,1,6,3,8,2],
    "attendance": [80,60,90,50,85,65,95,55],
    "previous_marks": [70,50,88,40,75,60,92,48],
    "sleep_hours": [6,5,7,4,6,5,8,4],
    "assignment": [1,0,1,0,1,0,1,0],
    "result": [1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")