import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Setup mlflow tracking server dan eksperimen
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Submission Membangun Sistem Machine Learning")

# Load data
data = pd.read_csv('Membangun_model/Social_Network_Ads_preprocessing.csv')

# Split Data
X = data.drop("Purchased", axis=1)
y = data["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Contoh input 
input_example = X_train.iloc[0:5]

print("Memulai Training dengan MLflow Autolog...")

# Training & Tracking
with mlflow.start_run():
    # Parameter Model
    n_estimators = 100
    max_depth = 10
    
    # Aktifkan Autolog (Wajib untuk poin Basic)
    mlflow.autolog()

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Validasi sederhana
    accuracy = model.score(X_test, y_test)
    print(f"Model berhasil dilatih. Akurasi: {accuracy:.4f}")