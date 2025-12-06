"""
Feature engineering - 14 features
Following Lab 9 organization (functions for testing)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def engineer_features(df):
    """Engineer 14 features from HDFS logs"""
    df_feat = df.copy()
    df_feat['timestamp'] = pd.to_datetime(df_feat['Date & Time'])
    
    # TIME-BASED FEATURES (3)
    df_feat['hour'] = df_feat['timestamp'].dt.hour
    df_feat['day'] = df_feat['timestamp'].dt.day
    df_feat['dayofweek'] = df_feat['timestamp'].dt.dayofweek
    
    # LOG LEVEL FEATURES (3)
    df_feat['is_warning'] = (df_feat['Log Level'] == 'WARN').astype(int)
    df_feat['is_error'] = (df_feat['Log Level'] == 'ERROR').astype(int)
    df_feat['is_info'] = (df_feat['Log Level'] == 'INFO').astype(int)
    
    # COMPONENT ENCODING (1)
    le_comp = LabelEncoder()
    df_feat['component_code'] = le_comp.fit_transform(df_feat['Component'])
    
    # FREQUENCY FEATURES (3)
    logs_per_hour = df_feat.groupby('hour').size()
    df_feat['logs_per_hour'] = df_feat['hour'].map(logs_per_hour).fillna(0)
    
    warnings_per_hour = df_feat[df_feat['Log Level']=='WARN'].groupby('hour').size()
    df_feat['warnings_per_hour'] = df_feat['hour'].map(warnings_per_hour).fillna(0)
    
    df_feat['error_rate'] = df_feat['is_error'] * 100.0 / (df_feat['logs_per_hour'] + 1)
    
    # NETWORK FEATURES (2)
    ip_freq = df_feat['IP Address'].value_counts()
    df_feat['ip_frequency'] = df_feat['IP Address'].map(ip_freq).fillna(1)
    
    block_freq = df_feat['Block ID'].value_counts()
    df_feat['block_frequency'] = df_feat['Block ID'].map(block_freq).fillna(1)
    
    # MESSAGE FEATURES (2)
    df_feat['message_length'] = df_feat['Content/Message'].str.len()
    df_feat['contains_exception'] = df_feat['Content/Message'].str.contains(
        'exception|error|failed|failure', case=False, na=False).astype(int)
    
    print(f"✓ Engineered 14 features (total: {len(df_feat.columns)})")
    return df_feat, le_comp

def create_feature_matrix(df_feat):
    """Create normalized feature matrix - following Lab 9 structure"""
    feature_cols = [
        'hour', 'day', 'dayofweek',
        'is_warning', 'is_error', 'is_info',
        'component_code',
        'logs_per_hour', 'warnings_per_hour', 'error_rate',
        'ip_frequency', 'block_frequency',
        'message_length', 'contains_exception'
    ]
    
    X = df_feat[feature_cols].fillna(0).astype(float)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✓ Feature matrix: {X_scaled.shape}")
    return X_scaled, scaler, feature_cols

def create_labels(df):
    """Create binary labels - anomaly vs normal"""
    y = ((df['Log Level'] == 'WARN') | (df['Log Level'] == 'ERROR')).astype(int)
    
    print(f"✓ Normal: {(y==0).sum()}, Anomaly: {(y==1).sum()}")
    return y
