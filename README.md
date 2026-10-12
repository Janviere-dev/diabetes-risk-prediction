# Multi-Class Diabetes Risk Prediction using CDC Health Indicators

## Project Overview
This project builds and evaluates both traditional machine learning models and a deep learning Multi-Layer Perceptron (MLP) to predict diabetes risk levels (Healthy, Prediabetes, and Diagnosed Diabetes) using behavioral and clinical survey indicators from the CDC BRFSS dataset.

## Dataset
* **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS 2015) via UCI / Kaggle.
* **Target Variable:** `Diabetes_012` (0 = Healthy, 1 = Prediabetes, 2 = Diabetes).
* **Instances:** 253,680 survey responses with 21 feature variables.

## Project Structure
- `notebooks/`: Contains exploratory data analysis, data cleaning, and model training notebooks.
- `reports/`: Final documentation.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/Janviere-dev/diabetes-risk-prediction.git](https://github.com/Janviere-dev/diabetes-risk-prediction.git)