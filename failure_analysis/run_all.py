from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import config
from utils import ensure_output_dirs, required_output_paths


SCRIPTS = [
    "01_load_and_validate.py",
    "02_compute_model_behavior.py",
    "03_overall_model_performance.py",
    "04_error_cooccurrence.py",
    "05_consensus_failure_mining.py",
    "06_high_confidence_errors.py",
    "07_disagreement_and_selective_referral.py",
    "08_subgroup_analysis.py",
    "09_external_validation_if_available.py",
    "10_export_paper_tables.py",
    "11_patient_level_sensitivity.py",
    "12_malignancy_aware_referral.py",
]


def main() -> None:
    ensure_output_dirs()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(config.OUTPUT_DIR / "mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(config.OUTPUT_DIR / "mplconfig"))
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    script_dir = Path(__file__).resolve().parent / "scripts"
    for script in SCRIPTS:
        path = script_dir / script
        print(f"\n[run_all] Running {script}")
        result = subprocess.run([sys.executable, str(path)], cwd=config.PROJECT_ROOT, env=env)
        if result.returncode != 0:
            raise SystemExit(f"[run_all] {script} failed with exit code {result.returncode}")

    print("\n[run_all] Required output checklist")
    for path in required_output_paths():
        status = "OK" if path.exists() else "MISSING"
        print(f"  {status:7s} {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
