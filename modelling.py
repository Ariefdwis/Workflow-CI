import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn



MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Automated_CI_Experiment")

def main():
   
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("Eksperimen_Model_Iris") 

    print("Memulai proses training...")


    try:
        df = pd.read_csv('preprocessing/train_clean.csv')
    except FileNotFoundError:
        print("Error: File 'train_clean.csv' tidak ditemukan di folder 'preprocessing'.")
        return

    X = df.drop(columns=['Species'])
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [None, 5]
    }


    with mlflow.start_run():

        grid_search = GridSearchCV(rf, param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Parameter Terbaik: {best_params}")

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi: {acc}")

  
        

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)


        mlflow.sklearn.log_model(best_model, "model")


        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
 
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        print("Sukses! Hasil training terkirim ke DagsHub.")

if __name__ == "__main__":
    main()