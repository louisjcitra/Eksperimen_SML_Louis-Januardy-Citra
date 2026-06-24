import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Setup dagshub tracking server dan eksperimen
DAGSHUB_USERNAME = "louisjcitra" 
REPO_NAME = "Membangun_Model_Louis-Januardy-Citra"

dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)

mlflow.set_experiment("Submission Membangun Sistem Machine Learning")

# Load data
try:
    data = pd.read_csv('Membangun_model/Social_Network_Ads_preprocessing.csv')
except FileNotFoundError:
    print("ERROR: File tidak ditemukan. Pastikan terminal di root folder.")
    exit()

# Split Data
X = data.drop("Purchased", axis=1)
y = data["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# hyperparameter tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Inisialisasi GridSearch
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

# Lakukan Tuning
grid_search.fit(X_train, y_train)

# Ambil hasil terbaik
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Parameter Terbaik: {best_params}")

# Logging manual ke dagshub
with mlflow.start_run():
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    # evaluasi model terbaik
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Akurasi Final: {acc:.4f}")

    # C. LOG METRIK (Manual)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(best_model, "model_best_tuning")

    # Membuat 2 artefak tambahan
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Purchased', 'Purchased'], 
                yticklabels=['Not Purchased', 'Purchased'])
    plt.title('Confusion Matrix - Best Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    cm_filename = "confusion_matrix_tuned.png"
    plt.savefig(cm_filename)
    mlflow.log_artifact(cm_filename) # Upload
    plt.close()

    importances = best_model.feature_importances_
    features = X.columns
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance - Best Model")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices])
    plt.tight_layout()
    
    fi_filename = "feature_importance_tuned.png"
    plt.savefig(fi_filename)
    mlflow.log_artifact(fi_filename)
    plt.close()
