# Global Interpretability of Decision Trees

## 1. Approach

We developed a metric to estimate the **global interpretability of decision trees** based on three structural complexity components:

- **Tree Depth (d):** Maximum number of decisions along a root-to-leaf path.
- **Split Nodes (n):** Total number of decision nodes.
- **Used Features (f):** Number of distinct features used for splitting.

Each component is normalized, and the final interpretability score is calculated as:

$$
I(T) = 100 \left(1 - \frac{1}{3} \left(
\frac{d}{d+5} +
\frac{n}{n+10} +
\frac{f}{f+5}
\right)\right)
$$

The score ranges from 0 to 100, with **higher values indicating lower structural complexity and greater estimated interpretability**.

The normalization constants were chosen heuristically.

An earlier version also included Feature Combination Complexity. We removed this component because it was highly correlated with the number of used features (Pearson correlation ≈ 0.977), and its removal did not change the model rankings across our 18 Titanic configurations (Spearman correlation = 1.0).

## 2. Experiments and Results

We evaluated 18 `DecisionTreeClassifier` configurations on two datasets:

- **Titanic:** Passenger survival prediction.
- **Breast Cancer Wisconsin:** Classification of benign and malignant tumors.

We varied `max_depth`, `max_leaf_nodes`, and `min_samples_leaf`, using a stratified 75/25 train-test split with random seed 0.

The most accurate models were:

| Dataset | Model | Test Accuracy | Interpretability |
|---|---|---:|---:|
| Titanic | `tree_depth_3` | 83.23% | 57.11 |
| Breast Cancer | `tree_min_samples_5` | 92.31% | 45.36 |

Our experiments show that increasing tree complexity does not necessarily improve predictive performance.

For example, on Breast Cancer, `tree_leaves_5` and `tree_leaves_20` both achieved 91.61% accuracy, but their interpretability scores were 65.48 and 39.43, respectively.

## 3. Interpretability Threshold

We investigated an exploratory structural interpretability threshold using the following complexity budget:

- Maximum actual tree depth: **3**
- Maximum number of leaf nodes: **8**

Within this budget, `tree_depth_3` achieved 83.23% accuracy on Titanic, while `tree_leaves_5` achieved 91.61% on Breast Cancer.

This budget is a heuristic assumption, not a validated threshold of human understanding.

## 4. Reproducibility

Install dependencies:

```bash
uv sync --extra dev
```

Run the experiments:

```bash
uv run python scripts/run_experiment.py
```

Generate the bar charts:

```bash
uv run python scripts/plot_bar_charts.py
```

Results are saved in `results/scores_two_datasets.csv`.

The generated figures are available in `results/figures/`.

## 5. Limitations

Our metric measures structural complexity rather than actual human understanding. The normalization parameters and interpretability threshold were chosen heuristically.

We evaluated only two datasets with one train-test split and did not conduct a sensitivity analysis. Future work could investigate parameter sensitivity and compare the metric with human judgments.

## AI Statement

We used **ChatGPT (OpenAI)** to support the development of this exercise, including discussions of the metric design, explanations of mathematical and machine-learning concepts, assistance with Python implementation, creation of visualization code, and drafting the README.

We reviewed and adapted AI-generated suggestions and checked the implementation by running the experiments on both datasets and examining the generated results.

The interpretability scores are calculated using a deterministic mathematical formula rather than an LLM-based evaluation.

We take full responsibility for and ownership of the content of this repository.