# C4.5 Decision Tree Implementation in Python

## 1. Project Overview

This project is a **from-scratch implementation of the C4.5 Decision Tree algorithm** using Python. It is designed to demonstrate the internal mathematical logic of decision trees without relying on high-level machine learning libraries (such as `scikit-learn`).

Unlike standard implementations that often rely on the CART algorithm (using Gini Impurity), this project strictly adheres to the **C4.5 principles** introduced by Ross Quinlan, utilizing **Entropy** and **Gain Ratio** as the core splitting criteria.

### Key Features

- **Pure Python Logic:** No `sklearn.tree` or pre-built model libraries used.
- **C4.5 Metrics:** Implements **Entropy**, **Information Gain**, and **Gain Ratio**.
- **Continuous Data Support:** Dynamically calculates thresholds to split numerical data (e.g., `Temperature <= 75.5`).
- **Visualization:** Generates structural diagrams of the trained tree using Graphviz.

---

## 2. Directory Structure

The project is organized into modular components to separate mathematical logic from data handling and execution.

```text
DT-SCRATCH/
│
├── core/                  # The "Brain" of the algorithm
│   ├── metrics.py         # Mathematical formulas (Entropy, Gain Ratio)
│   ├── node.py            # The Node class structure
│   └── decision_tree.py   # The C4.5 Algorithm (fit, predict, build_tree)
│
├── utils/                 # Helper utilities
│   ├── data_loader.py     # CSV loading and categorical encoding
│   └── visualizer.py      # Graphviz tools to draw the tree
│
├── dataset/               # Input Data
│   └── iris.csv           # Example dataset
│
├── output/                # Results
│   └── tree_viz.png       # Generated visualization of the model
│
├── main.py                # Entry point to run the project
├── docs.md                # Project documentation
└── requirements.txt       # Dependencies
```
