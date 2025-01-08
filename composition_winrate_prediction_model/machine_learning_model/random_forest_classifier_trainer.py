import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_random_forest(X, y, test_size=0.2, n_estimators=100, random_state=42):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "random_forest_model_preprocessed.pkl")
    joblib.dump(rf, model_path)
    print("Model saved to", model_path)

def train_randon_forest_processed_data():
    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(csv_path, header=None)

    # Assign column names
    print(df.shape)
    columns = ["target"] + ["feature_" + str(i) for i in range(1, df.shape[1])]
    df.columns = columns
    print(columns)
    # Extract features and target
    X = df.iloc[:, 1:]
    y = df["target"]

    # Train model without One-Hot Encoding
    train_random_forest(X, y)

if __name__ == "__main__":
    train_randon_forest_processed_data()
