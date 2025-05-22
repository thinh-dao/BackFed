"""
Tests for defense categorization.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backfed.servers.defense_categories import (
    ClientSideDefenseServer,
    RobustAggregationServer,
    AnomalyDetectionServer,
    PostAggregationServer
)
from backfed.utils.defense_utils import (
    get_defense_category,
    get_defenses_by_category,
    is_defense_in_category
)
from backfed.servers import (
    TrimmedMeanServer,
    FedProxServer,
    FlameServer,
    FoolsGoldServer,
    WeakDPServer,
    FLTrustServer,
    FlareServer
)

class TestDefenseCategories(unittest.TestCase):
    """Test cases for defense categorization."""

    def test_defense_categories(self):
        """Test that defenses are correctly categorized."""
        # Test client-side defenses
        self.assertIn("client_side", get_defense_category(FedProxServer))
        self.assertIn("client_side", get_defense_category(WeakDPServer))

        # Test robust aggregation defenses
        self.assertIn("robust_aggregation", get_defense_category(TrimmedMeanServer))
        self.assertIn("robust_aggregation", get_defense_category(FLTrustServer))

        # Test anomaly detection defenses
        self.assertIn("anomaly_detection", get_defense_category(FoolsGoldServer))

        # Test hybrid defenses
        self.assertIn("anomaly_detection", get_defense_category(FlameServer))
        self.assertIn("robust_aggregation", get_defense_category(FlameServer))
        self.assertIn("anomaly_detection", get_defense_category(FlareServer))
        self.assertIn("robust_aggregation", get_defense_category(FlareServer))

    def test_get_defenses_by_category(self):
        """Test getting defenses by category."""
        client_side_defenses = get_defenses_by_category("client_side")
        self.assertIn("FedProxServer", client_side_defenses)
        self.assertIn("WeakDPServer", client_side_defenses)

        robust_agg_defenses = get_defenses_by_category("robust_aggregation")
        self.assertIn("TrimmedMeanServer", robust_agg_defenses)
        self.assertIn("FLTrustServer", robust_agg_defenses)

        anomaly_detection_defenses = get_defenses_by_category("anomaly_detection")
        self.assertIn("FoolsGoldServer", anomaly_detection_defenses)

        # Hybrid defenses should appear in multiple categories
        self.assertIn("FlameServer", robust_agg_defenses)
        self.assertIn("FlameServer", anomaly_detection_defenses)

    def test_is_defense_in_category(self):
        """Test checking if a defense is in a category."""
        self.assertTrue(is_defense_in_category(FedProxServer, "client_side"))
        self.assertTrue(is_defense_in_category(TrimmedMeanServer, "robust_aggregation"))
        self.assertTrue(is_defense_in_category(FoolsGoldServer, "anomaly_detection"))

        # Hybrid defenses should be in multiple categories
        self.assertTrue(is_defense_in_category(FlameServer, "anomaly_detection"))
        self.assertTrue(is_defense_in_category(FlameServer, "robust_aggregation"))

        # Negative tests
        self.assertFalse(is_defense_in_category(FedProxServer, "anomaly_detection"))
        self.assertFalse(is_defense_in_category(TrimmedMeanServer, "client_side"))

    def test_anomaly_detection_metrics(self):
        """Test anomaly detection metrics calculation."""
        # Create a mock server for testing
        server_config = {"device": "cpu"}
        server = AnomalyDetectionServer(server_config, "test_anomaly")

        # Test with no detections
        metrics = server.get_detection_performance()
        self.assertEqual(metrics["tpr"], 0.0)
        self.assertEqual(metrics["fpr"], 0.0)

        # Test with some detections - old format (just indices)
        detected = [0, 1, 2, 5]
        true_malicious = [0, 1, 3, 4]
        result = server.evaluate_detection(detected, true_malicious)

        # Test with new format (indices and info)
        detection_info = {"scores": [0.9, 0.8, 0.7, 0.2, 0.3, 0.6]}
        result2 = server.evaluate_detection((detected, detection_info), true_malicious)

        # Check round metrics for first result
        self.assertEqual(result["tp"], 2)  # 0, 1
        self.assertEqual(result["fp"], 2)  # 2, 5
        self.assertEqual(result["fn"], 2)  # 3, 4
        self.assertNotIn("detection_info", result)

        # Check round metrics for second result (with detection info)
        self.assertEqual(result2["tp"], 2)  # 0, 1
        self.assertEqual(result2["fp"], 2)  # 2, 5
        self.assertEqual(result2["fn"], 2)  # 3, 4
        self.assertIn("detection_info", result2)
        self.assertEqual(result2["detection_info"]["scores"], [0.9, 0.8, 0.7, 0.2, 0.3, 0.6])

        # Check overall metrics
        metrics = server.get_detection_performance()
        self.assertAlmostEqual(metrics["recall"], 0.5)  # 2/4 (recall is the same as TPR)
        self.assertAlmostEqual(metrics["precision"], 0.5)  # 2/4
        self.assertIn("counts", metrics)
        self.assertEqual(metrics["counts"]["tp"], 4)  # 2 from each test

        # Reset and check
        server.reset_detection_metrics()
        metrics = server.get_detection_performance()
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["rounds"], 0)

if __name__ == "__main__":
    unittest.main()
