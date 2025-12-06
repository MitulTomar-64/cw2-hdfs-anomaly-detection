"""
Data Quality Tests - Lab 9 framework
Tests: load, columns, nulls, values, duplicates, distribution
"""
import unittest
import pandas as pd
import os

class TestDataQuality(unittest.TestCase):
    """Data quality test cases"""
    
    @classmethod
    def setUpClass(cls):
        csv_path = 'final_structured_hdfs_logs-5.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        cls.df = pd.read_csv(csv_path)
    
    def test_data_loads(self):
        """Test 1: Data loads successfully"""
        self.assertIsNotNone(self.df)
        self.assertGreater(len(self.df), 0)
    
    def test_required_columns(self):
        """Test 2: All required columns exist"""
        required = ['Date & Time', 'Process ID', 'Log Level', 'Component', 
                    'IP Address', 'Block ID', 'Content/Message']
        for col in required:
            self.assertIn(col, self.df.columns)
    
    def test_no_nulls_critical(self):
        """Test 3: No nulls in critical columns"""
        critical = ['Date & Time', 'Log Level', 'Component']
        for col in critical:
            self.assertEqual(self.df[col].isnull().sum(), 0)
    
    def test_valid_log_levels(self):
        """Test 4: Only valid log levels exist"""
        valid = {'INFO', 'WARN', 'ERROR'}
        actual = set(self.df['Log Level'].unique())
        self.assertTrue(actual.issubset(valid))
    
    def test_valid_timestamps(self):
        """Test 5: Timestamps are parseable"""
        try:
            pd.to_datetime(self.df['Date & Time'])
            self.assertTrue(True)
        except:
            self.fail("Invalid timestamp format")
    
    def test_no_duplicates(self):
        """Test 6: Reasonable duplicate ratio"""
        duplicates = self.df.duplicated().sum()
        ratio = duplicates / len(self.df)
        self.assertLess(ratio, 0.05)
    
    def test_anomaly_distribution(self):
        """Test 7: Reasonable anomaly count"""
        anomalies = ((self.df['Log Level'] == 'WARN') | 
                    (self.df['Log Level'] == 'ERROR')).sum()
        ratio = anomalies / len(self.df)
        self.assertGreater(ratio, 0.01)

if __name__ == '__main__':
    unittest.main()
