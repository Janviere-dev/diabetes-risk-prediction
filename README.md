# Diabetes Health Indicators: Traditional ML vs. Deep Learning

## Project overview

This project compares traditional machine-learning algorithms and deep neural networks for classifying three diabetes-related categories using health indicators: **No diabetes (0), Prediabetes (1), and Diabetes (2)**.

The project is motivated by the idea that accessible technology might help people become more aware of their health and encourage them to seek professional screening when appropriate. The model developed here is an **exploratory classification model**, not a clinical diagnostic tool and not a validated predictor of future diabetes onset.

## Research questions

1. How do traditional machine-learning algorithms compare with a multilayer perceptron (MLP) on the three-class target?
2. How does the target-class imbalance affect macro F1, precision, and recall, especially for prediabetes?
3. Do class weighting, focal loss, oversampling, architectural regularization, or a changed prediction threshold improve minority-class detection?
4. How does removing exact duplicate rows and predictor patterns with conflicting labels affect the selected models?

## Demo Video

**Link to demo video:** https://www.youtube.com/watch?v=7CpLsjSJ1-I

## Dataset

The project uses the **Diabetes Health Indicators** dataset published by Alex Teboul on Kaggle, based on the 2015 U.S. Behavioral Risk Factor Surveillance System (BRFSS).

- Kaggle dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- CDC BRFSS 2015 annual data: https://www.cdc.gov/brfss/annual_data/annual_2015.html
- CDC BRFSS 2015 codebook: https://www.cdc.gov/brfss/annual_data/2015/pdf/CODEBOOK15_LLCP.pdf

The original CSV used in the main study contains **253,680 rows and 22 columns** (21 predictors and the `Diabetes_012` target). The original target distribution is approximately 84.24% no diabetes, 1.83% prediabetes, and 13.93% diabetes.

### Download and place the data

1. Download `diabetes_012_health_indicators_BRFSS2015.csv` from the Kaggle dataset page.
2. Create a local `data/` directory at the project root.
3. Place the CSV at:

   `data/diabetes_012_health_indicators_BRFSS2015.csv`

## Repository structure

```text
diabets-risk-prediction/
├── README.md
├── requirements.txt
│── 01_Diabetes_Health_Indicators_Classification_and_Evaluation.ipynb
│── 02_Duplicate_and_Conflicting_Label_Sensitivity_Analysis.ipynb
├── data/                         # local only; excluded from Git
├── results/                       # tables, figures, learning curves
├── results_deduplication/         # cleaning sensitivity outputs
├── saved_models/                  # selected original-data model artifacts
└── saved_models_deduplicated/     # optional cleaned-data artifacts
```

## Methods and experiments

### Traditional machine learning

The main notebook evaluates Logistic Regression, Random Forest, Gradient Boosting, and CatBoost. Experiments vary class weighting or sample weighting and selected tree hyperparameters such as depth, estimators, learning rate, and minimum leaf size.

### Deep learning

The neural-network experiments use TensorFlow/Keras multilayer perceptrons (MLPs) with the `tf.data` input pipeline. Experiments compare:

- a baseline MLP with balanced class weights;
- an unweighted MLP;
- inverse-square-root class weighting;
- focal loss;
- a deeper MLP architecture;
- random oversampling of the training set;
- a regularized MLP with Batch Normalization and L2 regularization.

Experiment 15 is a **prediction-threshold analysis**, not a separately trained model. It adjusts the prediabetes decision threshold for the Experiment 13 MLP using validation probabilities and then evaluates the fixed threshold on test data.

### Evaluation

Accuracy is reported but is not used alone because of class imbalance. The notebooks also use macro F1, weighted F1, per-class precision and recall, confusion matrices, ROC-AUC, and Average Precision (AP). Training/validation learning curves are examined to discuss generalization. Model selection and threshold selection use the validation set; the test set is reserved for final evaluation.

## Main findings

On the original unseen test set:

- Balanced Random Forest (Experiment 3): accuracy 0.7612, macro F1 0.4419, diabetes recall 0.6839.
- Deeper MLP with focal loss (Experiment 13): accuracy 0.8215, macro F1 0.4536, diabetes recall 0.5295.
- Experiment 13 with the validation-selected prediabetes threshold (Experiment 15): accuracy 0.8046, macro F1 0.4614, prediabetes precision 0.0390 and recall 0.0628.

The threshold adjustment slightly improved test macro F1 but prediabetes precision remained very low. Therefore, these models are not ready for clinical or real-world screening deployment. The findings are limited by the severe class imbalance, self-reported survey labels, the 2015 U.S. sample, and the lack of independent clinical validation.

### Duplicate/conflicting-label sensitivity study

Notebook 2 investigates removing records whose predictors repeat exactly with conflicting target labels and then removing exact duplicate rows. This reduced the dataset from 253,680 to 226,074 rows. On the cleaned test set, retraining had only small effects on overall macro F1; both models continued to struggle to predict prediabetes. This is a sensitivity analysis, not proof that conflicting-label respondents were incorrect.

## How to run the notebooks

### 1. Create an environment

Use a Python version supported by your installed TensorFlow release. The project was run in a Python 3.13 Linux environment; if TensorFlow installation fails in your environment, use a supported Python version listed by the TensorFlow installation documentation https://www.tensorflow.org/install .

Example using `venv` on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start Jupyter

```bash
jupyter notebook
```

Open the notebooks and run cells from top to bottom. Set the dataset path in the data-loading cell if your CSV is located elsewhere. Set random seeds as specified in the notebooks. Keep the original train/validation/test split logic intact when reproducing original-data results.

### 3. Weights and Biases

The notebooks use Weights and Biases (WandB) to record experiment configurations and metrics. If using online tracking, authenticate with WandB before running the experiment cells. For offline logging, set the relevant mode in the notebook to `offline`; synchronize runs later with:

```bash
wandb sync wandb/
```

Do not add API keys or secrets to the notebook.

## Reproducibility and limitations

The notebooks document the data preparation, model configurations, and evaluation procedure. The exact metrics may vary if package versions, random seeds, the data source, or split construction differ. The target is self-reported diabetes-related status, not a laboratory-confirmed diagnosis. Because prediabetes is rare and class-specific performance remains weak, the models should be used only for coursework and research, not for medical decisions.

## Acknowledgements

The project reuses a publicly available dataset from Kaggle based on CDC BRFSS 2015 data and uses open-source Python libraries including scikit-learn, TensorFlow/Keras, CatBoost, pandas, NumPy, Matplotlib, and Weights and Biases. No pretrained neural network was used in the experiments described here.


