import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("StudentsPerformance.csv")
df['gender'] = df['gender'].map({'female': 0, 'male': 1})
df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})

df['result'] = df.apply(lambda row: 1 if row['math score'] >= 40 and row['reading score'] >= 40 and row['writing score'] >= 40 else 0, axis=1)

X = df[['math score', 'reading score', 'writing score']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

sample_student = [[67, 78, 76]]  # You can change these values

math, read, write = sample_student[0]
if math < 40 or read < 40 or write < 40:
    print("Manual Rule: Fail (one or more subjects below 40)")
else:
    prediction = model.predict(sample_student)
    print("Model Prediction:", "Pass" if prediction[0] == 1 else "Fail")
