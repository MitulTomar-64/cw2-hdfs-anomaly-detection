"""
Load HDFS logs - adapted from Lab 3 pattern
"""
import pandas as pd
import argparse

def load_data(filepath):
    """
    Load HDFS logs from local or Azure blob
    Following Lab 3 structure for production code
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded: {len(df)} rows, {len(df.columns)} columns")
    return df

def validate_data(df):
    """
    Validate data quality - following Lab 9 testing patterns
    """
    # Check required columns exist
    required_cols = ['Date & Time', 'Process ID', 'Log Level', 'Component', 
                     'IP Address', 'Block ID', 'Content/Message']
    
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check no nulls in critical columns
    assert df[['Date & Time', 'Log Level', 'Component']].isnull().sum().sum() == 0
    
    # Check valid log levels
    valid_levels = {'INFO', 'WARN', 'ERROR'}
    assert set(df['Log Level'].unique()).issubset(valid_levels)
    
    print("✓ Data validation passed!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/final_structured_hdfs_logs-5.csv')
    args = parser.parse_args()
    
    df = load_data(args.data)
    validate_data(df)
