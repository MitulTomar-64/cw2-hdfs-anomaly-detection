"""
Model training with MLFlow tracking
Following Lab 3 & Lab 8 patterns
"""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import argparse
import pickle
import os

def train_model(X_scaled, y, iteration=1, contamination=0.05, n_estimators=100):
    """Train and log model with MLFlow"""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration} - Training")
    print(f"{'='*70}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Setup MLFlow
    mlflow.set_experiment(f"hdfs_anomaly_v{iteration}")
    
    with mlflow.start_run(run_name=f"iteration_{iteration}"):
        
        # Train model
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train)
        print("✓ Training complete!")
        
        # Predict
        y_pred_test = model.predict(X_test)
        y_pred_binary = (y_pred_test == -1).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        
        decision = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, -decision)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
        
        # Log parameters
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        
        # Log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("tp", tp)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tn", tn)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/model_v{iteration}.pkl"
        pickle.dump(model, open(model_path, 'wb'))
        mlflow.log_artifact(model_path)
        
        print(f"\n{'='*70}")
        print(f"RESULTS - ITERATION {iteration}")
        print(f"{'='*70}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f} ⭐")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"{'='*70}\n")
        
        return model, {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_test': y_test,
            'y_pred': y_pred_binary
        }

def trainModel(X_scaled, y):
    """Wrapper for Azure ML job"""
    # Iteration 1: Baseline
    model_v1, metrics_v1 = train_model(X_scaled, y, iteration=1, 
                                       contamination=0.05, n_estimators=100)
    
    # Iteration 2: Improved
    model_v2, metrics_v2 = train_model(X_scaled, y, iteration=2, 
                                       contamination=0.08, n_estimators=150)
    
    # Show improvement
    print("\n" + "="*70)
    print("COMPARISON: ITERATION 1 vs ITERATION 2")
    print("="*70)
    improvement = ((metrics_v2['f1'] - metrics_v1['f1']) / metrics_v1['f1']) * 100
    print(f"F1 improvement: {metrics_v1['f1']:.4f} → {metrics_v2['f1']:.4f} (+{improvement:.1f}%)")
    print("="*70)
    
    return model_v1, model_v2, metrics_v1, metrics_v2

if __name__ == "__main__":
    from src.data_loader import load_data, validate_data
    from src.feature_engineering import engineer_features, create_feature_matrix, create_labels
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/final_structured_hdfs_logs-5.csv')
    args = parser.parse_args()
    
    df = load_data(args.data)
    validate_data(df)
    df_feat, _ = engineer_features(df)
    X_scaled, _, _ = create_feature_matrix(df_feat)
    y = create_labels(df)
    
    trainModel(X_scaled, y)
