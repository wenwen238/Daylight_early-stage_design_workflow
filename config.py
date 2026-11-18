"""Paths and settings shared by every step of the workflow.

All paths are relative to this folder, so the workflow runs wherever the folder
is copied. The only machine-specific setting is the Radiance installation:
set the environment variable RADIANCE_ROOT (the folder that contains bin/ and
lib/), otherwise C:\\Radiance is used.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# -- inputs ---------------------------------------------------------------
DATA = ROOT / "thesis_data" / "inputs"
CASES_CSV = DATA / "fixed_design_cases_560.csv"
CASES_JSON = DATA / "fixed_design_cases_560.json"
WEA = DATA / "weather" / "milan.wea"          # from ITA_LM_Milano-Linate.AP.160800_TMYx.epw

# -- outputs --------------------------------------------------------------
RESULTS = ROOT / "thesis_data" / "results"
RADIANCE_560 = RESULTS / "radiance_560"            # one folder per model + sda_ase_560_cases.csv
RADIANCE_SUMMARY = RADIANCE_560 / "sda_ase_560_cases.csv"
RADIANCE_COMMON = RESULTS / "radiance_common"      # sky matrix and sun sources, shared by all runs
MCP_RUNS = RESULTS / "mcp_runs"                    # Radiance runs requested through MCP
ML_TRAINING = RESULTS / "ml_training"
ENSEMBLE_DIR = ML_TRAINING / "reduced" / "mlp_ensemble"
DECISION_CONFIG = ENSEMBLE_DIR / "decision_config.json"

# -- Radiance -------------------------------------------------------------
RADIANCE_ROOT = Path(os.environ.get("RADIANCE_ROOT", r"C:\Radiance"))
RAD_BIN = RADIANCE_ROOT / "bin"
RAD_LIB = RADIANCE_ROOT / "lib"

# -- Rhino bridge ---------------------------------------------------------
RHINO_URL = os.environ.get("RHINO_URL", "http://127.0.0.1:8765")
