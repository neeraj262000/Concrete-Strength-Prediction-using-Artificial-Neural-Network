
# Concrete Strength Prediction using Artificial Neural Network

This project focuses on building and training an Artificial Neural Network (ANN) model to predict the compressive strength of concrete based on its ingredients.

## 🧠 Project Overview

Predicting the strength of concrete is critical in civil engineering and construction. This project utilizes machine learning techniques, specifically an ANN, to model the relationship between concrete mix components and its compressive strength.

## 📁 Files Included

- `Concrete_Strength_ANN.ipynb` — Jupyter Notebook with the entire workflow including data preprocessing, model creation, training, and evaluation.

## 📊 Dataset Description

The dataset includes the following features:

- **Cement (kg/m³)**
- **Blast Furnace Slag (kg/m³)**
- **Fly Ash (kg/m³)**
- **Water (kg/m³)**
- **Superplasticizer (kg/m³)**
- **Coarse Aggregate (kg/m³)**
- **Fine Aggregate (kg/m³)**
- **Age (days)**
- **Concrete Compressive Strength (MPa)** — *Target variable*

## 🔧 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn (for visualizations)

## 🚀 Workflow

1. **Data Exploration**: Inspect features and target variable.
2. **Data Preprocessing**: Handle scaling and splitting into training/testing sets.
3. **Model Building**: Design a feed-forward neural network using Keras.
4. **Training**: Train the model and monitor performance using loss metrics.
5. **Evaluation**: Evaluate the model on test data using MAE/MSE/R².
6. **Prediction & Visualization**: Predict concrete strength and visualize results.

## 📈 Performance Metrics

Common evaluation metrics for regression tasks:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

## ✅ How to Run

1. Clone this repository.
2. Make sure required libraries are installed (see below).
3. Open the notebook: `Concrete_Strength_ANN.ipynb`
4. Run all cells in order.

## 📦 Installation

Install the necessary Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

## ✍️ Author

Generated and structured using OpenAI's ChatGPT.

---

This project serves as a simple yet insightful example of how neural networks can model real-world engineering problems.
