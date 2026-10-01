"""Checks against row-order dependence and accidental label-informed tie breaking."""
import unittest
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from referral import referral_order, random_expected_row
from utils import vote_entropy

class ReferralTests(unittest.TestCase):
    def test_missing_ids_rejected(self):
        for cid in [None, float("nan"), "", "  "]:
            df = pd.DataFrame({"case_id":[cid],"score":[0.5]})
            with self.assertRaises(ValueError):
                referral_order(df,df.score,"LUNA25")
    def test_score_ties_ignore_order_and_labels(self):
        df = pd.DataFrame({"case_id":["a","b","c","d"], "score":[0.9,0.9,0.5,0.9],
                           "y_true":[0,1,0,1]})
        expected = df.loc[referral_order(df,df.score,"LUNA25"),"case_id"].tolist()
        other = df.sample(frac=1,random_state=17).reset_index(drop=True)
        other["y_true"] = 1-other["y_true"]
        self.assertEqual(expected,other.loc[referral_order(other,other.score,"LUNA25"),"case_id"].tolist())
    def test_vote_complements_are_exact_ties(self):
        values = [vote_entropy(i,7) for i in range(8)]
        self.assertEqual(len(set(values)),4)
        self.assertEqual(values,values[::-1])
    def test_uniform_random_expectation_is_constant(self):
        df = pd.DataFrame({"mean_p":[0.1,0.9,0.6,0.2],"y_true":[0,1,0,1]})
        for pct in [0,20,40]:
            self.assertEqual(random_expected_row(df,pct)["auto_error_rate"],0.5)

if __name__ == "__main__":
    unittest.main()
