Data Classification Using AI


Goal

Build a basic classification model using a small dataset (the classic
Iris benchmark — 150 samples, 3 species, 4 features) and prove the full
supervised-learning pipeline: load data → split → scale → train (KNN) →
validate (confusion matrix + F1 score).


This follows the exact IPO Framework taught in the deck:


Stage	What happens
Input	Load Iris dataset, apply feature scaling
Process	Train-test split, K-Nearest Neighbors algorithm
Output	Confusion matrix, F1 score

Requirements / Setup

pip install scikit-learn pandas numpy matplotlib

(All are standard data-science libraries; no internet/API keys needed —
the Iris dataset ships built into scikit-learn.)


How to Run

python3 classification.py

Running it prints step-by-step output to the terminal and saves two
image files in the same folder:



k_tuning_curve.png — error rate vs. K (shows how the optimal K was chosen)

confusion_matrix.png — visual confusion matrix of the final model


Sample Output

STEP 1: LOAD & UNDERSTAND THE DATASET
Samples: 150 | Features: 4 | Classes: 3
...
STEP 2: TRAIN-TEST SPLIT
Training samples: 120 | Testing samples: 30

STEP 3: FEATURE SCALING (StandardScaler: mean=0, variance=1)

BONUS: TUNING K (Elbow Method via Cross-Validation)
Best K found: 5 (lowest CV error = 3.33%)

STEP 4: TRAIN THE MODEL (K-Nearest Neighbors, k=5)

STEP 5: OUTPUT VALIDATION
Accuracy: 93.33%
Macro F1 Score: 93.27%



• Load and understand a dataset — load_dataset() loads Iris, prints shape, class names, sample rows, and class balance.

• Split data into training and testing sets — split_data() does an 80/20 stratified, shuffled split (matches the "Structural Integrity: The Split" slide).

• Apply a simple classification algorithm — train_model() uses KNeighborsClassifier exactly per the "Workflow: Scikit-Learn" slide (Instantiate → Fit → Predict).

• Feature Scaling — scale_features() applies StandardScaler (the "Gatekeeper Rule" slide), fit only on training data to avoid leakage.

• Confusion Matrix & F1 Score — evaluate_model() reports both, per the "Output Validation" and "Diagnostic Tool" slides (accuracy alone can be misleading, especially on imbalanced data).


Bonus (per the Conclusion slide's "experiment with unique solutions"):



K is tuned automatically via 5-fold cross-validation on the training set only (not the test set — using the test set to pick a hyperparameter is data leakage and was an actual bug caught and fixed during development; it had wrongly picked K=1, the exact overfitting case the "Tuning the Engine" slide warns about).

A K-tuning curve plot is generated automatically.

