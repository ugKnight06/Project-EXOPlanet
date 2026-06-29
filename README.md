# Project-EXOPlanet
A comparison the the kaggle kepler dataset between classical and quantum support vector machine models 

---

##  Overview

Exoplanets — planets orbiting stars outside our solar system — are detected by NASA's Kepler mission through the **transit method**: a planet passing in front of its star causes a tiny, periodic dip in brightness. The Kepler dataset contains thousands of these light-curve-derived measurements, labeled as `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE`.

This project builds a binary classifier to separate **confirmed exoplanets** from **non-exoplanets**, and then asks a more interesting question:

> **Does encoding this data into a quantum feature space give any classification advantage over a classical kernel?**

To answer that, two models are trained and compared head-to-head on identical preprocessed data:

| Model | Kernel |
|---|---|
| Classical SVM | RBF (or linear/poly) |
| Quantum SVM | `ZZFeatureMap` + `FidelityQuantumKernel` |

---

##  Dataset

- **Source:** [NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) (via Kaggle)
- **File used:** `cumulative.csv`
- **Target column:** `koi_disposition`
  - `CONFIRMED` → positive class (`1`)
  - `FALSE POSITIVE` + `CANDIDATE` → negative class (`0`)
- **Challenge:** Heavy class imbalance — confirmed exoplanets are a small minority of all observations, making naive accuracy a poor evaluation metric on its own.

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Inspected column types, missing values, and the class distribution of `koi_disposition`
- Visualized class imbalance before any preprocessing

### 2. Preprocessing
- Engineered a binary target from `koi_disposition`
- Dropped non-predictive columns (IDs, names, disposition labels)
- Handled missing values (with justification — see notebook)
- Separated features (`X`) and target (`y`)

### 3. Class Imbalance Handling
- Performed a **stratified train/validation split** *before* any resampling
- Applied **SMOTE** to the training set only — validation data was left untouched to avoid leakage
- Verified class distribution pre- and post-SMOTE

### 4. Feature Engineering & Selection
- Standardized features using `StandardScaler`
- Selected the top **8 features** using `SelectKBest` — chosen to keep the quantum circuit width computationally tractable (more qubits = exponentially higher simulation cost)

### 5. Classical SVM
- Trained an `SVC` on the preprocessed, balanced training data
- Evaluated on the held-out validation set using accuracy, precision, recall, and F1-score
- Visualized results with a confusion matrix

### 6. Quantum Feature Map & Kernel
- Encoded the 8 selected features into a quantum state using a **`ZZFeatureMap`** (8 qubits)
- Computed the quantum kernel matrix via **`FidelityQuantumKernel`**
- Subsampled training data (~100–200 rows) to keep kernel computation feasible

### 7. Quantum SVM (Q-SVM)
- Trained `SVC(kernel='precomputed')` on the quantum kernel matrix
- Predicted on the precomputed validation kernel matrix
- Evaluated using the same metric suite as the classical model

### 8. Classical vs Quantum — Comparison
- Built a side-by-side metrics table for both models
- Plotted a comparative bar chart
- Analyzed *why* one approach outperformed (or matched) the other

---

##  Results

> *Fill in after running the notebook — keep both models' numbers honest, including where Q-SVM underperforms.*

| Metric | Classical SVM | Q-SVM |
|---|---|---|
| Accuracy | 81.91 | 78.67 |
| Precision | 56.96 | 55.88 |
| Recall |97.36 | 95.00 |
| F1-score | 71.87 |70.37 |


- **Language:** Python 3.10+
- **Classical ML:** `scikit-learn` (SVC, StandardScaler, SelectKBest), `imbalanced-learn` (SMOTE)
- **Quantum ML:** `qiskit`, `qiskit-machine-learning` (`ZZFeatureMap`, `FidelityQuantumKernel`)
- **Data handling & viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`

---

##  Project Structure

```
exoplanet-detection-qsvm/
├── data/
│   └── cumulative.csv
├── notebook/
│   └── exoplanet_detection_classical_vs_quantum.ipynb
├── assets/
│   ├── classical_confusion_matrix.png
│   ├── quantum_confusion_matrix.png
│   └── model_comparison_bar_chart.png
├── requirements.txt
└── README.md
```

---

##  Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/exoplanet-detection-qsvm.git
cd exoplanet-detection-qsvm

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place cumulative.csv in data/

# Run the notebook
jupyter notebook notebook/exoplanet_detection_classical_vs_quantum.ipynb
```

**`requirements.txt`** (suggested)
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
qiskit
qiskit-machine-learning
```

---

##  What This Project Explores

- Real-world astrophysics data wrangling: noisy, imbalanced, high-dimensional
- Why preprocessing decisions (imputation, scaling, balancing) matter *before* any model touches the data
- How quantum kernels are constructed and where their computational bottlenecks lie
- A grounded, honest comparison of quantum vs classical ML — without hype, just metrics

---

##  Notes & Limitations

- Quantum kernel computation scales poorly with sample size, so the Q-SVM was trained on a subsampled dataset (~100–200 rows) — results should be read as a proof-of-concept, not a production-scale benchmark.
- Feature count was capped at 8 to match a simulatable qubit count; this constrains both models for a fairer comparison, but may limit overall classifier performance versus using the full feature set.

---

##  Acknowledgements

- NASA Kepler Mission & the Kepler Exoplanet Search Results dataset (via Kaggle)
- Quantum computation and Quantum Information by Nielsen and Chuang
