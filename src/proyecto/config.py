"""Constantes del proyecto RiskVision (Default of Credit Card Clients, UCI).

Un solo lugar para los nombres de columnas y la semilla, para que el EDA, la limpieza, la partición
y los modelos hablen de lo mismo.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_FILE = RAW_DIR / "default of credit card clients.xls"

RANDOM_STATE = 42
TEST_SIZE = 0.20

ID_COL = "ID"                              # identifica al cliente; NO es un predictor
TARGET = "default payment next month"      # 1 = incumple el próximo mes, 0 = paga

# Son códigos de categorías, no cantidades: se codifican (one-hot) dentro del pipeline
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
