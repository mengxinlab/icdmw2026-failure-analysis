"""Regression checks for complete patient mapping without requiring optional metadata."""
import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import require_patient_mapping


class PatientMappingTests(unittest.TestCase):
    def test_missing_case_cannot_hide_behind_unchanged_patient_set(self):
        meta = pd.DataFrame({"case_id": ["a", "b"], "patient_id": [10, 20]})
        self.assertEqual(set(meta.patient_id), {10, 20})
        with self.assertRaisesRegex(ValueError, "missing rows=1"):
            require_patient_mapping(["a", "b", "c"], meta)

    def test_invalid_patient_rejected(self):
        for patient in [None, float("nan"), float("inf"), "", "invalid", 10.5]:
            with self.subTest(patient=patient):
                meta = pd.DataFrame({"case_id": ["a"], "patient_id": [patient]})
                with self.assertRaises(ValueError):
                    require_patient_mapping(["a"], meta)

    def test_missing_optional_covariates_are_allowed(self):
        meta = pd.DataFrame({"case_id": ["a", "b"], "patient_id": [10, 10],
                             "size_mm": [None, 5], "density": ["Missing", "Solid"]})
        matched = require_patient_mapping(["a", "b"], meta)
        self.assertEqual(len(matched), 2)
        self.assertEqual(int(matched.size_mm.isna().sum()), 1)

    def test_duplicate_mapping_rejected(self):
        meta = pd.DataFrame({"case_id": ["a", "a"], "patient_id": [10, 10]})
        with self.assertRaises(ValueError):
            require_patient_mapping(["a"], meta)


if __name__ == "__main__":
    unittest.main()
