"""
Performance Tests - Speed and efficiency benchmarks
"""
import unittest
import time
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification

class TestPerformance(unittest.TestCase):
    """Performance test cases"""
    
    def test_model_training_time(self):
        """Test 1: Model trains in reasonable time"""
        X, y = make_classification(n_samples=1000, n_features=14, random_state=42)
        
        start = time.time()
        model = IsolationForest(contamination=0.05, n_estimators=100)
        model.fit(X)
        elapsed = time.time() - start
        
        # Should complete in less than 10 seconds
        self.assertLess(elapsed, 10.0)
    
    def test_prediction_speed(self):
        """Test 2: Predictions are fast"""
        X, y = make_classification(n_samples=1000, n_features=14, random_state=42)
        model = IsolationForest(contamination=0.05, n_estimators=100)
        model.fit(X)
        
        start = time.time()
        predictions = model.predict(X[:100])
        elapsed = time.time() - start
        
        # Should complete in less than 1 second
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(predictions), 100)

if __name__ == '__main__':
    unittest.main()
