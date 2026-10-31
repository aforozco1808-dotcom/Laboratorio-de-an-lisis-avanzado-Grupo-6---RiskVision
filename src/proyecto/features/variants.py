"""Variantes de variables que mide el notebook 04: cada idea es una hipótesis que se compara con la referencia."""
from proyecto.features.build import PACKAGE_D, PACKAGE_E

REFERENCE_NAME = "referencia (23 variables)"

# nombre -> (variables que se añaden, columnas que se quitan)
VARIANTS = {
    REFERENCE_NAME: ((), ()),
    "+ utilización (utilization_last, utilization_mean, utilization_max)": (("utilization_last", "utilization_mean", "utilization_max"), ()),
    "+ pending_balance_last": (("pending_balance_last",), ()),
    "+ delay_streak": (("delay_streak",), ()),
    "+ delay_severity": (("delay_severity",), ()),
    "+ severe_delay_months": (("severe_delay_months",), ()),
    "+ n_delayed_months": (("n_delayed_months",), ()),
    "+ n_zero_payments": (("n_zero_payments",), ()),
    "+ n_minimum_payments": (("n_minimum_payments",), ()),
    "+ log_limit": (("log_limit",), ()),
    "paquete E (5 variables)": (PACKAGE_E, ()),
    "paquete D (10 variables)": (PACKAGE_D, ()),
    "sin SEX": ((), ("SEX",)),
    "sin PAY_0": ((), ("PAY_0",)),
}
