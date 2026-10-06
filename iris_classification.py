import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def load_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = iris.target

    return df, iris.target_names


def explore_data(df):
    print("Dataset shape:", df.shape)
    print("\nFirst rows:")
    print(df.head())

    print("\nStatistics:")
    print(df.describe())

    print("\nClass distribution:")
    print(df["species"].value_counts())


def create_plot(df):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["sepal length (cm)"],
        df["petal length (cm)"],
        c=df["species"],
        cmap="viridis"
    )

    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.title("Iris Dataset Overview")

    plt.tight_layout()
    plt.savefig("iris_scatter.png")
    plt.close()


def train_model(df):

    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return model


def main():
    df, species_names = load_data()

    explore_data(df)

    create_plot(df)

    model = train_model(df)

    print("\nSpecies labels:")
    print(species_names)

    print("\nModel training completed.")


if __name__ == "__main__":
    main()