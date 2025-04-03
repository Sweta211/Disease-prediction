import pickle
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
data = load_diabetes()
X, y = data.data, data.target

# Binarize the target for a simple classification problem
y = (y > y.mean()).astype(int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('models/diabetes.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'models/diabetes.pkl'")
