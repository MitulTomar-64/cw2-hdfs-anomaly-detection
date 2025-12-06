"""
Preprocessing Tests - Feature engineering validation
"""
import unittest
import numpy as np
from src.feature_engineering import engineer_features, create_feature_matrix
from src.data_loader import load_data

class TestPreprocessing(unittest.TestCase):
    """Preprocessing test cases"""
    
    @classmethod
    def setUpClass(cls):
        cls.df = load_data('final_structured_hdfs_logs-5.csv')
    
    def test_feature_engineering(self):
        """Test 1: Features engineered correctly"""
        df_feat, _ = engineer_features(self.df)
        self.assertGreaterEqual(len(df_feat.columns), len(self.df.columns) + 10)
    
    def test_feature_matrix_shape(self):
        """Test 2: Feature matrix has correct shape"""
        df_feat, _ = engineer_features(self.df)
        X, _, _ = create_feature_matrix(df_feat)
        self.assertEqual(X.shape, len(self.df))
        self.assertEqual(X.shape, 14)
    
    def test_scaler_normalization(self):
        """Test 3: Scaling produces normalized features"""
        df_feat, _ = engineer_features(self.df)
        X, _, _ = create_feature_matrix(df_feat)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        np.testing.assert_array_almost_equal(mean, np.zeros(14), decimal=10)
        np.testing.assert_array_almost_equal(std, np.ones(14), decimal=0)

if __name__ == '__main__':
    unittest.main()
