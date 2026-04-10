import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path):
    df = pd.read_csv(csv_path)

    df.set_index("pipeline").plot(kind="bar")
    plt.title("Guardrail Comparison")
    plt.ylabel("Score")
    plt.show()
