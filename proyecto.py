import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('online_shoppers_intention.csv')

# Convert boolean columns (True/False) to 1 and 0
df['Revenue'] = df['Revenue'].astype(int)
df['Weekend'] = df['Weekend'].astype(int)

# Separate features (X) and target variable (y)
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Group columns by their true data type
numeric_features = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

categorical_features = [
    'Month', 'OperatingSystems', 'Browser',
    'Region', 'TrafficType', 'VisitorType'
]

binary_features = ['Weekend']

# Stratified Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing rules 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bin', 'passthrough', binary_features)
    ]
)

# Build Pipeline: Preprocessing + Neural Network (MLP)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('mlp', MLPClassifier(
        max_iter=1000,
        early_stopping=True,     
        n_iter_no_change=15,     
        random_state=42
    ))
])

# Hyperparameter tuning with 5-Fold Stratified Cross-Validation
param_grid = {
    'mlp__hidden_layer_sizes': [(32,), (64, 32), (50, 25)],  
    'mlp__alpha': [0.0001, 0.01, 0.1],                       
    'mlp__learning_rate_init': [0.001, 0.01]                 
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running GridSearchCV (testing 18 MLP configurations with 5-fold CV)...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',       
    cv=cv,
    n_jobs=-1          
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Overfitting check
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1] 

# Save all results to resultados.txt instead of printing them to the terminal
output_lines = [
    "BEST MODEL CONFIGURATION:",
    str(grid_search.best_params_),
    f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}",
    "",
    "OVERFITTING CHECK (TRAIN VS. TEST):",
    f"Train Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.2f}% | Test Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.2f}%",
    f"Train F1-Score: {f1_score(y_train, y_train_pred):.4f} | Test F1-Score: {f1_score(y_test, y_test_pred):.4f}",
    "",
    "FINAL EVALUATION ON UNSEEN DATA (TEST SET)",
    f"Overall Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.2f}%",
    f"F1-Score (Class 1): {f1_score(y_test, y_test_pred):.4f}",
    f"F2-Score (Class 1): {fbeta_score(y_test, y_test_pred, beta=2):.4f}",
    f"ROC-AUC Score:      {roc_auc_score(y_test, y_test_proba):.4f}",
    f"PR-AUC Score:       {average_precision_score(y_test, y_test_proba):.4f}",
    f"MCC Score:          {matthews_corrcoef(y_test, y_test_pred):.4f}",
    "",
    "Confusion Matrix:",
    str(confusion_matrix(y_test, y_test_pred)),
    "",
    "Detailed Classification Report:",
    classification_report(y_test, y_test_pred),
]

with open("results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Done! Results saved to results.txt")