"""Variables candidatas de la etapa 4 (notebook 04, especificaciones/04-features.md).

Cada variable es una función pura que usa solo las columnas de la fila (historial de 6 meses y límite de
crédito), todo conocido al momento de decidir. Ninguna aprende nada de los datos ni mira el objetivo, así
que calcularlas antes de partir no filtra información. Lo que sí se aprende (medianas, escalas, categorías)
sigue ajustándose dentro del Pipeline, en `proyecto.models.train`.

Si el límite es vacío o ≤ 0, las razones salen vacías (no infinito); el Pipeline las imputa con la mediana.
"""
import numpy as np
import pandas as pd

from proyecto.data.cleaning import BILL_COLS, NON_NEGATIVE, PAY_COLS

PAY_AMT_COLS = NON_NEGATIVE
MIN_PAYMENT_SHARE = 0.05   # supuesto del grupo (no es un dato): pagar ≤ 5 % de la factura se trata como «pago mínimo»


def _limit(df: pd.DataFrame) -> pd.Series:
    """Límite de crédito; vacío si no es positivo para no dividir entre cero."""
    limit = df["LIMIT_BAL"].astype(float)
    return limit.where(limit > 0)


def utilization_last(df):
    """Factura del último mes sobre el límite."""
    return df["BILL_AMT1"] / _limit(df)


def utilization_mean(df):
    """Factura media de los 6 meses sobre el límite."""
    return df[BILL_COLS].mean(axis=1) / _limit(df)


def utilization_max(df):
    """Factura más alta de los 6 meses sobre el límite (puede ser > 1: sobrepasó el límite)."""
    return df[BILL_COLS].max(axis=1) / _limit(df)


def pending_balance_last(df):
    """Lo que queda sin pagar tras el último pago (factura de agosto menos pago de septiembre), sobre el límite."""
    return (df["BILL_AMT2"] - df["PAY_AMT1"]) / _limit(df)


def delay_streak(df):
    """Meses seguidos con atraso (PAY > 0) contando desde el más reciente (PAY_0) hacia atrás."""
    late = (df[PAY_COLS].to_numpy(dtype=float) > 0).astype(int)
    return pd.Series(np.cumprod(late, axis=1).sum(axis=1), index=df.index)


def delay_severity(df):
    """Suma de los meses de atraso de los 6 meses; -2, -1 y 0 (sin atraso) no suman."""
    return df[PAY_COLS].clip(lower=0).sum(axis=1)


def severe_delay_months(df):
    """Meses con 2 o más meses de atraso."""
    return (df[PAY_COLS] >= 2).sum(axis=1)


def n_delayed_months(df):
    """Meses con algún atraso (PAY > 0)."""
    return (df[PAY_COLS] > 0).sum(axis=1)


def n_zero_payments(df):
    """Meses en que el cliente no pagó nada."""
    return (df[PAY_AMT_COLS] == 0).sum(axis=1)


def n_minimum_payments(df):
    """De 5 meses, en cuántos pagó ≤ 5 % de una factura positiva del mes anterior (pago de k contra factura de k+1)."""
    bills = df[BILL_COLS[1:]].to_numpy(dtype=float)
    pays = df[PAY_AMT_COLS[:5]].to_numpy(dtype=float)
    return pd.Series(((bills > 0) & (pays <= MIN_PAYMENT_SHARE * bills)).sum(axis=1), index=df.index)


def log_limit(df):
    """Logaritmo del límite: acorta su cola larga (ayuda a los modelos lineales)."""
    return np.log(_limit(df))


FEATURES = {f.__name__: f for f in (
    utilization_last, utilization_mean, utilization_max, pending_balance_last, delay_streak, delay_severity,
    severe_delay_months, n_delayed_months, n_zero_payments, n_minimum_payments, log_limit,
)}

# Paquetes medidos en el notebook 04 (ver reports/04_features.md)
PACKAGE_E = ("utilization_mean", "pending_balance_last", "delay_streak", "n_zero_payments", "log_limit")
PACKAGE_D = ("utilization_last", "utilization_mean", "utilization_max", "pending_balance_last", "delay_severity", "severe_delay_months",
             "delay_streak", "n_zero_payments", "n_minimum_payments", "log_limit")

# Paquete elegido en el notebook 04 (reports/04_features.md). Se fija después de medir y confirmar.
FINAL_FEATURES = PACKAGE_E


def build_features(df: pd.DataFrame, names=()) -> pd.DataFrame:
    """Devuelve una copia de `df` con las variables pedidas añadidas; no borra filas ni toca el original."""
    names = tuple(names)
    unknown = [n for n in names if n not in FEATURES]
    if unknown:
        raise ValueError(f"Variables desconocidas: {unknown}. Disponibles: {sorted(FEATURES)}")
    if len(set(names)) != len(names):
        raise ValueError("Hay variables repetidas en la lista")
    clashes = [n for n in names if n in df.columns]
    if clashes:
        raise ValueError(f"Esas columnas ya existen en la tabla: {clashes}")
    new = pd.DataFrame({n: FEATURES[n](df) for n in names}, index=df.index)
    return pd.concat([df, new], axis=1)
