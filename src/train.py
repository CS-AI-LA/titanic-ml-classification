import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklvq import GMLVQ

# Fix for numpy <1.24 deprecation warning
if not hasattr(np, 'int'):
    np.int = int

def load_processed(path='data/processed_train.csv'):
    """Load preprocessed Titanic CSV file."""
    return pd.read_csv(path)

def main():
    # --- Load data ---
    df = load_processed()

    # --- Convert True/False string columns to numeric ---
    for col in df.columns:
        if df[col].dtype == object and df[col].isin(['True', 'False']).all():
            df[col] = df[col].apply(lambda x: 1 if x=='True' else 0)

    # Separate features and label
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Standardize features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Define GMLVQ model (fixed parameters) ---
    model = GMLVQ(
        distance_type="adaptive-squared-euclidean",
        activation_type="swish",
        activation_params={"beta": 2},
        solver_type="waypoint-gradient-descent",
        solver_params={"max_runs": 5, "k": 2, "step_size": np.array([0.1, 0.05])},
        random_state=42
    )

    # --- Train model ---
    print("Training GMLVQ model...")
    model.fit(X_train_scaled, y_train)
    print("Training completed.")

    # --- Predict ---
    y_pred = model.predict(X_test_scaled)

    # --- Classification report ---
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show(block=False)                  # Show figure
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

    # --- Feature relevance plot ---
    relevance_matrix = model.lambda_
    fig, ax = plt.subplots()
    ax.bar(X.columns, np.diagonal(relevance_matrix))
    ax.set_ylabel("Weight")
    plt.xticks(rotation=45, ha='right')
    plt.title("Feature Relevance")
    plt.tight_layout()
    plt.show(block=False)                  # Show figure
    plt.savefig("relevance_matrix.png")
    plt.close()
    print("Feature relevance plot saved as 'relevance_matrix.png'")

    # --- 2D discriminative projection ---
    transformed_test = model.transform(X_test_scaled, scale=True)
    x_test_proj = transformed_test[:, 0]
    y_test_proj = transformed_test[:, 1]

    prototypes_proj = model.transform(model.prototypes_, scale=True)
    x_proto = prototypes_proj[:, 0]
    y_proto = prototypes_proj[:, 1]

    fig, ax = plt.subplots()
    colors = ['blue', 'red']  # survived=0, survived=1
    labels_sorted = sorted(model.classes_)

    for i, cls in enumerate(labels_sorted):
        idx = y_test.values == cls
        ax.scatter(
            x_test_proj[idx], y_test_proj[idx],
            c=colors[i], s=80, alpha=0.7, edgecolors="white",
            label=f"Survived = {cls}"
        )

    # Plot prototypes
    for i, cls in enumerate(labels_sorted):
        ax.scatter(
            x_proto[i], y_proto[i],
            c=colors[i], s=180, alpha=0.9, edgecolors="black",
            linewidth=2.0, marker="X", label=f"Prototype: {cls}"
        )

    ax.set_xlabel("First Eigenvector")
    ax.set_ylabel("Second Eigenvector")
    ax.legend()
    ax.grid(True)
    plt.title("GMLVQ 2D Discriminative Projection")
    plt.tight_layout()
    plt.show(block=False)                  # Show figure
    plt.savefig("gmlvq_2d_projection.png")
    plt.close()
    print("2D projection plot saved as 'gmlvq_2d_projection.png'")

if __name__ == '__main__':
    main()
