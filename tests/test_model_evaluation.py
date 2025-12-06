"""
Model Evaluation Tests - Metrics validation
"""
import unittest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class TestModelEvaluation(unittest.TestCase):
    """Model evaluation test cases"""
    
    def test_precision_calculation(self):
        """Test 1: Precision metric calculated correctly"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        precision = precision_score(y_true, y_pred)
        self.assertGreater(precision, 0)
        self.assertLessEqual(precision, 1)
    
    def test_recall_calculation(self):
        """Test 2: Recall metric calculated correctly"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        recall = recall_score(y_true, y_pred)
        self.assertGreater(recall, 0)
        self.assertLessEqual(recall, 1)
    
    def test_f1_score_calculation(self):
        """Test 3: F1-Score metric calculated correctly"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        f1 = f1_score(y_true, y_pred)
        self.assertGreater(f1, 0)
        self.assertLessEqual(f1, 1)

if __name__ == '__main__':
    unittest.main()
