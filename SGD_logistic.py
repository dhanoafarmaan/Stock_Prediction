import data
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

split = int(len(data.df)*.8)

train = data.df.iloc[:split]
test = data.df.iloc[split:]

X_train = train.drop(columns="Target")
X_test = test.drop(columns="Target")
y_train = train["Target"]
y_test = test["Target"]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


clf = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.1,
    max_iter=1000,
    random_state=0
)
clf.fit(X_train_scaled, y_train)

acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"SGD model accuracy: {acc:.2f}%")

latest_X = data.df.drop(columns=["Target"]).iloc[-1:]
latest_X_scaled = scaler.transform(latest_X)

prediction = clf.predict(latest_X_scaled)
probabilities = clf.predict_proba(latest_X_scaled)

print(f"Prediction for the next day: {prediction[0]}")
print(f"Probability of down: {probabilities[0][0] * 100:.2f}%")
print(f"Probability of up: {probabilities[0][1] * 100:.2f}%")