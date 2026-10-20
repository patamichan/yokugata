"""Shared locations for the human-readable workspace layout (independent of cwd)."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = APP_DIR.parent
DATA_DIR = WORKSPACE_DIR / '02_データ'
DATABASE_DIR = DATA_DIR / 'v2'
INPUT_DIR = DATA_DIR / '人体プロファイル'
CONFIG_DIR = DATA_DIR / '設定'
CAD_DIR = WORKSPACE_DIR / '03_CAD出力'
RESEARCH_DIR = WORKSPACE_DIR / '05_研究記録'
LEGACY_DIR = WORKSPACE_DIR / '90_旧版' / 'v1'
DEFAULT_DATABASE = DATABASE_DIR / 'v2_Fairing_Airfoil_Database.csv'
DEFAULT_GRID_CONFIG = CONFIG_DIR / 'fairing_3d_grid_config.json'
