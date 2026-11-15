
"""Create bar charts comparing accuracy and interpretability."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("results/scores_two_datasets.csv")
OUTPUT_DIR = Path("results/figures")


def plot_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    """Create a grouped bar chart for one dataset."""

    # Select models with maximum depth 1 to 5.
    model_names = [f"tree_depth_{d}" for d in range(1, 6)]

    data = df[
        (df["dataset"] == dataset_name)
        & (df["model"].isin(model_names))
    ].copy()

    # Ensure the models appear in order.
    data["model"] = pd.Categorical(
        data["model"],
        categories=model_names,
        ordered=True,
    )
    data = data.sort_values("model")

    if len(data) != 5:
        raise ValueError(
            f"Expected 5 models for {dataset_name}, found {len(data)}."
        )

    # Convert accuracy to percentage.
    accuracy = data["accuracy"].to_numpy() * 100
    interpretability = data[
        "global_tree_interpretability"
    ].to_numpy()

    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_accuracy = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy (%)",
    )

    bars_interpretability = ax.bar(
        x + width / 2,
        interpretability,
        width,
        label="Interpretability Score",
    )

    # Display values above each bar.
    ax.bar_label(
        bars_accuracy,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )
    ax.bar_label(
        bars_interpretability,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )

    ax.set_title(
        f"Accuracy vs. Interpretability – {dataset_name}"
    )
    ax.set_xlabel("Maximum Tree Depth")
    ax.set_ylabel("Value (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(1, 6)])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    output = OUTPUT_DIR / f"{dataset_name}_bar_chart.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {output}")


def main() -> None:
    df = pd.read_csv(INPUT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_dataset(df, "titanic")
    plot_dataset(df, "breast_cancer")


if __name__ == "__main__":
    main()
