CLARITY-AI 2.0: End-to-End Demo Pipeline
This repository contains the official demonstration script for the CLARITY-AI 2.0 research paper.

The purpose of this file is to provide a single, executable proof-of-concept that demonstrates the feasibility of our proposed framework. It shows the core end-to-end pipeline: from raw data loading to advanced XAI explanations.

Key Features Demonstrated
This script is a fully functional pipeline that:

Downloads Data: Automatically downloads the MIT-BIH Arrhythmia Database from PhysioNet.

Extracts Hybrid Features: Implements the novel 24+ feature extraction engine, combining statistical, wavelet (pywavelets), fiducial, and HRV (neurokit2) features.

Applies SMOTE: Uses imbalanced-learn to correct class imbalance in the training data.

Trains the Model: Trains the high-performance LightGBM (LGBM) classifier.

Evaluates Performance: Generates a full classification report, a confusion matrix, and an ROC curve.

Generates XAI: Uses the shap library to create and plot:

Global Explanations (a beeswarm plot, like Figure 11)

Local Explanations (a waterfall plot for a single anomaly, like Figure 12)

How to Run This Demo
The easiest way to run this script is in a cloud environment like Google Colab or a local Jupyter Notebook.

Step 1: Install Dependencies
Copy and paste the following command into a code cell and run it. This will install all required libraries.

Bash

!pip install wfdb numpy pandas matplotlib seaborn scikit-learn lightgbm imbalanced-learn shap neurokit2 pywavelets
Step 2: Run the Entire Script
Copy the entire Python script (.py file) into a new code cell and run it.

The script will execute from start to finish. It will:

Download the dataset (this may take a minute).

Process all 45 records and extract features.

Train the LightGBM model.

Print the evaluation metrics to the console.

Display the output plots (Confusion Matrix, ROC Curve, and SHAP plots) directly in your notebook.

Expected Output (The "Demo" Part)
This script is a working demo. It will execute and produce real results based on a single 70/30 split of the data. The output will include:

A Classification Report printed in the console (showing precision, recall, and F1-score similar to Table 6).

A Confusion Matrix plot (similar to Figure 7).

An ROC Curve plot (showing a high AUC, similar to Figure 8).

A SHAP Beeswarm Plot (similar to Figure 11), showing which features the model actually learned were important.

A SHAP Waterfall Plot (similar to Figure 12), explaining why a specific beat was flagged as an anomaly.

Note on Results: The metrics from this single 70/30 split are illustrative and will differ slightly from the final, averaged results reported in the paper. This demo proves the method, while the paper reports the rigorous, cross-validated results.

From Demo to Publication: How to Get the Final Paper Results
To reproduce the final, high-impact results presented in our paper (Tables 1-12, Figures 1-15), this demo script must be extended. This script provides the engine, but a full study requires a rigorous experimental harness.

Here are the key changes needed:

1. Rigorous Model Tuning (for Table 3)
The script has hard-coded LightGBM parameters (e.g., n_estimators=1450, max_depth=11). These values are the results of our tuning. To reproduce Table 3, you would need to run a hyperparameter search (e.g., using Optuna or GridSearchCV) on the training set to find these optimal values.

2. K-Fold Cross-Validation (for Tables 4, 5, 6, 9)
This demo uses a simple train_test_split. To get the robust, averaged metrics in the paper, you must wrap the model training and evaluation in a StratifiedKFold (e.g., 5-fold or 10-fold) cross-validation loop. You would then average the performance metrics (Accuracy, F1-Score, AUC) across all 5 or 10 folds.

3. Multi-Database Generalization (for Tables 1, 7, 10 & Figure 8)
This script only uses the MIT-BIH database. To get the results in Table 7 and Figure 8, you must:

Download the PTB-XL and Chapman datasets.

Adapt the load_and_segment_data function to read and process their different file formats and annotation styles.

Train the model only on MIT-BIH data.

Run model.predict() on the unseen PTB-XL and Chapman test sets to get the generalization scores.

4. On-Device Deployment (for Table 8, Figures 9, 10)
This script simulates the AI model in Python. It does not run on a real IoT device. To get the on-device metrics, you must:

Take the trained model (the lgbm.LGBMClassifier object).

Convert this model into an embedded format, such as LightGBM-Micro, TFLite, or C++ code.

Deploy this converted model onto a real microcontroller (like an ESP32).

Measure the actual inference time (in milliseconds) and energy consumption (in microJoules) on the hardware.

5. Clinical Utility Survey (for Table 12 & Figure 15)
This is an offline, human-based study. You would:

Run this script and use it to generate 10-20 high-quality SHAP waterfall plots (like Figure 12) for different anomalies.

Present these plots to a panel of cardiologists.

Have them fill out the survey questions listed in Table 12.

Core Dependencies
wfdb: For downloading and reading PhysioNet databases.

pandas & numpy: For data manipulation.

neurokit2: For advanced HRV and fiducial feature extraction.

pywavelets: For wavelet decomposition features.

imbalanced-learn: For the SMOTE algorithm.

scikit-learn: For scaling, splitting, and metrics.

lightgbm: The core classification model.

shap: For generating all XAI explanations and plots.

matplotlib & seaborn: For plotting.
