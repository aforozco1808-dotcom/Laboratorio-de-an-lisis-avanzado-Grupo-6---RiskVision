"""Limpieza del dataset crudo Default of Credit Card Clients.

Cada regla sale de una decisión del EDA (notebook 02, reports/01_eda.md) y es una función pequeña
con su prueba en tests/test_cleaning.py:

  #3  Los códigos de EDUCATION sin documentar (0, 5, 6) pasan a "otros" (4); el código 0 de
      MARRIAGE pasa a "otros" (3).
  #3  SEX, EDUCATION y MARRIAGE son etiquetas, no cantidades: se guardan como categorías.
  --  Lo imposible se marca como vacío (no se borran filas). En estos datos no hay casos, pero la
      regla protege cuando lleguen datos nuevos.
  #8  No se borra ninguna fila: la limpieza devuelve exactamente las filas que recibe.

Lo que NO se hace aquí (depende de los datos de entrenamiento y va dentro del Pipeline, etapa 4):
imputar con la mediana, estandarizar, codificar one-hot y transformar los montos (#4, #5).
"""
import numpy as np
import pandas as pd

from proyecto.config import CATEGORICAL, ID_COL, RAW_FILE, TARGET

# Códigos que la documentación de UCI no define, y a qué categoría documentada se agrupan
EDUCATION_RARE = {0: 4, 5: 4, 6: 4}   # 4 = "otros"
MARRIAGE_RARE = {0: 3}                # 3 = "otros"

# Lo que es posible en el mundo real (no es el rango observado)
VALID_VALUES = {"SEX": {1, 2}}
VALID_RANGE = {"AGE": (18, 100), "LIMIT_BAL": (1, np.inf)}
NON_NEGATIVE = [f"PAY_AMT{i}" for i in range(1, 7)]   # un pago no puede ser negativo (BILL_AMT sí: saldo a favor)


def load_raw(path=RAW_FILE) -> pd.DataFrame:
    """Lee el archivo crudo. La primera fila del .xls es auxiliar (X1...X23/Y); los nombres reales están en la segunda.

    No hay datos personales: solo un ID anónimo, que se conserva para verificar la partición.
    """
    df = pd.read_excel(path, header=1)
    missing = {ID_COL, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"El archivo no tiene las columnas esperadas: {sorted(missing)}")
    return df


def group_rare_education(series: pd.Series) -> pd.Series:
    """EDUCATION 0, 5 y 6 (sin documentar) pasan a 4 ('otros')."""
    return series.replace(EDUCATION_RARE)


def group_rare_marriage(series: pd.Series) -> pd.Series:
    """MARRIAGE 0 (sin documentar) pasa a 3 ('otros')."""
    return series.replace(MARRIAGE_RARE)


def to_category(df: pd.DataFrame, columns=CATEGORICAL) -> pd.DataFrame:
    """Marca las columnas como categorías (sus números son etiquetas, no cantidades)."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("category")
    return df


def mark_impossible_as_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte en vacío lo que no puede ocurrir (edad fuera de 18-100, SEX distinto de 1 o 2, pagos negativos...).

    No borra filas: un vacío se imputa después, dentro del Pipeline.
    """
    df = df.copy()
    for col, allowed in VALID_VALUES.items():
        df[col] = df[col].where(df[col].isin(allowed))
    for col, (low, high) in VALID_RANGE.items():
        df[col] = df[col].where(df[col].between(low, high))
    for col in NON_NEGATIVE:
        df[col] = df[col].where(df[col] >= 0)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el dataset limpio, con el mismo número de filas que el crudo. No modifica el original."""
    out = df.copy()
    out["EDUCATION"] = group_rare_education(out["EDUCATION"])
    out["MARRIAGE"] = group_rare_marriage(out["MARRIAGE"])
    out = mark_impossible_as_missing(out)
    out = to_category(out)
    return out
