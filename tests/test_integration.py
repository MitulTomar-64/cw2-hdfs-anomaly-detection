"""
Integration Tests - End-to-end pipeline
"""
import unittest
import os
from src.data_loader import load_data, validate_data
from src.feature_engineering import engineer_features, create_feature_matrix, create_labels

class TestIntegration(unittest.TestCase):
    """Integration test cases"""
    
    def test_full_pipeline(self):
        """Test 1: Full pipeline runs without error"""
        # Load data
        csv_path = 'final_structured_hdfs_logs-5.csv'
        if not os.path.exists(csv_path):
            self.skipTest("Dataset not found")
        
        df = load_data(csv_path)
        validate_data(df)
        
        # Engineer features
        df_feat, _ = engineer_features(df)
        
        # Create feature matrix
        X_scaled, _, _ = create_feature_matrix(df_feat)
        
        # Create labels
        y = create_labels(df)
        
        # Verify shapes match
        self.assertEqual(X_scaled.shape, y.shape)
        self.assertEqual(X_scaled.shape, len(df))

if __name__ == '__main__':
    unittest.main()
