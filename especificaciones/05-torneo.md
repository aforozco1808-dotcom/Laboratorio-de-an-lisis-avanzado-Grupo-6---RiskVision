# Especificación · Etapa 5 · Torneo de modelos

> Se escribe **antes** de mirar los resultados del torneo. El torneo elige uno o dos finalistas para optimizar; no elige un campeón.
> Guía: `material-del-curso/kit/guia-modelado-y-optimizacion.md`, sección 2.

**Grupo:** 6 (RiskVision)   **Fecha:** 2026-10-31   **Estado:** ☒ borrador (el grupo debe revisarlo)  ☐ cerrada

## 1. Datos y validación
- **Entrada:** `data/processed/train.joblib` (24.000 clientes). La prueba (`test.joblib`) **no se abre**. Recordatorio de la revisión H1: ya se exploró y se evaluó antes, así que no es una evaluación final independiente; ninguna decisión del torneo se toma con ella.
- **Pliegues:** los mismos 15 de la etapa 4 (5 estratificados × 3 repeticiones, semilla 42), partidos por fila porque los 30.000 `ID` son únicos.
- **Variables:** el paquete final de la etapa 4 (`FINAL_FEATURES`: `utilization_mean`, `pending_balance_last`, `delay_streak`, `n_zero_payments`, `log_limit`) para todos los modelos. Excepción declarada: una variante de LightGBM con el paquete D (10 variables), para cerrar el pendiente de la etapa 4.

## 2. Candidatos (parámetros fijos, sin optimizar)
| Candidato | Papel |
|---|---|
| Línea base: siempre paga | Estrategia trivial del notebook 01 |
| Línea base: prior | Misma probabilidad para todos (con el umbral 1/6 equivale a marcar a todos) |
| Línea base: regla `PAY_0 ≥ 2` | La regla simple del EDA; ampliación posterior (ver sección 7) |
| Regresión logística (`C=0.5`, `PAY_*` como estados, números estandarizados) | Contraste explicable |
| Bosque aleatorio (300 árboles, `min_samples_leaf=30`) | Árboles en paralelo |
| LightGBM (300 árboles, `learning_rate=0.03`, `num_leaves=15`) | Boosting, candidato principal |
| XGBoost (300 árboles, `learning_rate=0.03`, `max_depth=4`) | Boosting alternativo |
| LightGBM + paquete D | Variante extra |
| LightGBM calibrado (isotónica, 3 pliegues internos) | Variante extra de la ampliación (sección 7) |

Sin pesos de clase en ningún modelo: el desbalance se trata con el umbral y las métricas.

## 3. Métricas
- **Decide quién avanza: ROC AUC** (ficha del caso, `reports/00_caso.md`).
- **Control, con probabilidades:** AUC-PR, log loss, Brier y tasa predicha media.
- **Control, con la decisión «marcar si proba ≥ 1/6»:** exactitud balanceada, F1 macro, precisión y recall por clase, % marcado y penalización relativa 5:1. El 5:1 es una **hipótesis del grupo, no un costo observado**, y el umbral 1/6 solo es óptimo con probabilidades calibradas.
- **Control independiente de la calibración:** recall y lift al revisar el 20 % de mayor riesgo (escenario supuesto, no una capacidad medida). La sección 7 documenta la ampliación posterior: también el 5 % y el 10 %, la penalización con 2 a 1 y 10 a 1, y la calibración por tramos.

## 4. Regla para elegir finalistas (escrita antes de ver los resultados)
Se eligen **1 o 2** mirando, en este orden:
1. **Consistencia:** gana en la mayoría de los 15 pliegues y su diferencia pareada con el siguiente es mayor que 1 (misma medida que la etapa 4).
2. **ROC AUC medio.**
3. **Control:** que las métricas técnicas no contradigan el ranking (por ejemplo, un recall de «paga» casi nulo, o probabilidades mal calibradas si se va a usar un umbral de costos).
4. **Tiempo de ajuste y simplicidad**, ante un empate.
5. **Interpretabilidad:** se mantiene la regresión logística como contraste explicable si su diferencia con el mejor es pequeña; esa diferencia se traduce a volumen por cada 1.000 clientes.

Un empate práctico (gana en menos de 12 de 15 pliegues o diferencia pareada ≤ 1) no se resuelve por AUC; se pasa al criterio siguiente.

## 5. Código y pruebas
| Pieza | Ruta |
|---|---|
| Métricas técnicas y medición detallada | `src/proyecto/models/evaluate.py` (`technical_metrics`, `score_folds_detailed`) |
| Modelos candidatos y sus parámetros | `src/proyecto/models/candidates.py` (`tournament_candidates`, `reference_models`) y `tests/test_candidates.py` |
| Tablas de comparación y del torneo | `src/proyecto/models/comparison.py`, `src/proyecto/models/tournament.py`, `src/proyecto/features/variants.py` y sus pruebas `tests/test_comparison.py`, `tests/test_tournament.py` |
| Regla base `PAY_0 ≥ 2` | `src/proyecto/models/baselines.py` |
| Pipeline | `src/proyecto/models/train.py` (`build_pipeline`) |
| Pruebas | `tests/test_evaluate.py` (valores calculados a mano) |
| Notebook que orquesta | `notebooks/05_torneo.ipynb` |
| Resultado por pliegue | `reports/05_torneo_por_pliegue.csv` |
| Informe | `reports/05_torneo.md` |

## 6. Criterios de cierre
- ☐ Al menos 3 modelos y una línea base, en los mismos pliegues, con una fila por pliegue guardada.
- ☐ Métricas técnicas de cada modelo y comparación entre rankings.
- ☐ Comparación pareada de los mejores.
- ☐ Diferencia traducida a volumen por cada 1.000 clientes (no se inventa un volumen mensual).
- ☐ 1–2 finalistas justificados en `reports/05_torneo.md`, con lo que podría cambiar al optimizar.
- ☐ `pytest` en verde y la prueba sin abrir.
- ☐ El grupo aprueba los finalistas y valida el 5 a 1 con el instructor.

## 7. Desviaciones declaradas (2026-10-31)
- **Control con pliegues nuevos para el paquete D.** La regla de la sección 4 no lo preveía. Se agregó **después de ver** que LightGBM + paquete D salía mejor que LightGBM en los pliegues de la etapa 4 (+0,0015, 14 de 15, diferencia pareada 1,26). Motivo: D se había elegido mirando esos mismos pliegues. Con las semillas 7 y 99 la ventaja se mantiene en signo y tamaño (+0,0013 y +0,0012) pero la diferencia pareada baja a 0,78 y 0,74. Por la regla escrita, D habría pasado la consistencia; se descartó por simplicidad (10 variables en vez de 5) y por no sostener el umbral con pliegues nuevos. Las semillas nuevas reparten los mismos 24.000 clientes, así que no eliminan el sesgo de selección.
- **Variante extra en el torneo.** LightGBM + paquete D no estaba en el plan inicial de candidatos; se incluyó para cerrar un pendiente de la etapa 4.
- **Estado.** Sigue en borrador hasta que el grupo lo apruebe; no se marca como cerrada.
- **Ampliación elegida por el grupo después de ver los resultados (2026-10-31).** Se agregaron: (1) sensibilidad a la hipótesis 5 a 1 (2:1, 5:1 y 10:1, cada una con su umbral teórico) y a la capacidad de revisión (5 %, 10 % y 20 %); (2) la regla `PAY_0 ≥ 2` como tercera línea base; (3) calibración por tramos y una variante calibrada de LightGBM. Ninguna cambia las reglas de elegir finalistas de la sección 4; sirven para comprobar que la elección no depende de un solo supuesto. Como se decidieron después de ver el primer resultado, no son pruebas ciegas.
- **Límites que siguen abiertos.** La calibración se midió en conjunto y por 10 tramos, no por subgrupos; no se evaluó equidad por grupo; los parámetros no están afinados.
- **Comparación con el notebook de selección de modelo (2026-10-31).** A pedido del grupo se comparó ese notebook con este torneo en los mismos pliegues y se hizo una afinación exploratoria de LightGBM y una mezcla con la logística. Resultado y límites en `reports/05_torneo.md`. Los scripts son exploratorios y no están en el repositorio; la afinación formal queda para la etapa 6.
- **Segunda pasada de revisión (2026-10-31).** La auditoría de fuga no halló fuga en la ampliación. La revisión de código halló dos fallos con empates de probabilidad (capacidad de revisión y tramos de calibración), que dependían del orden de las filas; se corrigieron (valor esperado en los empates y tramos por valor), y también la cuenta de penalización repetida. Cambiaron las cifras de captura al revisar y de *lift* en las líneas base, la regla `PAY_0 ≥ 2` y el LightGBM calibrado (por ejemplo, la regla pasó de 40,2 % a 40,0 % al revisar el 20 %); AUC, penalizaciones y calibración no cambiaron. La lógica reutilizable de los notebooks 04 y 05 se movió a `src/proyecto/` (`models/candidates.py`, `models/comparison.py`, `models/tournament.py` y `features/variants.py`).
- **Tercera pasada de revisión (2026-10-31).** Sin fuga. Se corrigió que `calibration_table` pudiera fundir dos tramos distintos cuando un valor repetido caía justo en el borde, y se reforzaron las pruebas (dirección de las comparaciones, variables que recibe cada variante, pliegues de cada semilla, nombres de las líneas base). La calibración, el AUC y las penalizaciones no cambiaron. Puntos conocidos que quedan: el rango 12–22 % de la «zona del umbral» está escrito a mano, `evaluate.py` sigue siendo largo (T504) y el CSV conserva valores sin sentido para las reglas 0/1.
