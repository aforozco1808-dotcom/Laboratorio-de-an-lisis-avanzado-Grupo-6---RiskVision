# Especificación · Etapa 4 · Variables (features)

> Se llena **antes** de pedirle código a Claude. Cada idea es una **hipótesis que se mide**: entra solo si mejora el
> modelo más que el ruido y se conoce al momento de predecir. Guía: `curso/kit-estudiante/guia-modelado-y-optimizacion.md`, sección 1.
> (Ruta real en este repo: `material-del-curso/kit/guia-modelado-y-optimizacion.md`.)

**Grupo:** 6 (RiskVision)   **Fecha:** 2026-10-31   **Estado:** ☐ borrador  ☒ medida y documentada (pendiente de revisión del código y de aprobación del grupo)

## 1. Qué se mide y cómo

- **Entrada:** `data/processed/train.joblib` (24.000 clientes). La prueba (`test.joblib`) **no se abre**. Ojo (revisión H1 de Juan David, 2026-10-31): la prueba original **ya fue explorada** (EDA sobre el dataset completo) y **evaluada** (notebook de selección de modelo), así que no es una evaluación final independiente. Todas las decisiones de variables se toman solo con validación dentro del 80 % de entrenamiento.
- **Validación común:** 5 pliegues estratificados × 3 repeticiones (15 valores por variante), semilla 42, los mismos en todas las comparaciones. Partir por fila es válido porque los `ID` son únicos.
- **Modelos de referencia (parámetros fijos, solo cambian las variables):**
  - LightGBM (`n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, colsample_bytree=0.8`).
  - Regresión logística (`C=0.5`) con `PAY_*`, `SEX`, `EDUCATION` y `MARRIAGE` como categorías y el resto estandarizado, todo dentro del `Pipeline`.
- **Métrica principal:** ROC AUC. **De control:** AUC-PR y penalización relativa a 5:1 (**hipótesis del grupo, no un costo observado del banco**; con un umbral fijo de antemano, no elegido en cada pliegue). El umbral teórico de referencia es 1/6 ≈ 0,167 y solo vale con probabilidades calibradas.
- **Referencia:** las 23 variables actuales (todas menos `ID` y objetivo).

## 2. Ideas candidatas (hipótesis)

Todas se calculan fila por fila con datos que el banco **ya tiene al decidir** (historial de 6 meses y límite), así que no hay fuga. Ninguna aprende nada de los datos, por eso son funciones deterministas y no necesitan ajustarse.

| # | Variable | Cómo se calcula | Por qué podría ayudar |
|---|---|---|---|
| 1 | `utilization_last`, `utilization_mean`, `utilization_max` | `BILL_AMT1`, media y máximo de `BILL_AMT1..6`, divididos entre `LIMIT_BAL` | Un cliente con la tarjeta casi llena tiene menos margen. El monto solo no dice nada sin el límite |
| 2 | `pending_balance_last` | `(BILL_AMT2 − PAY_AMT1) / LIMIT_BAL` | Lo que queda sin pagar tras el último pago, relativo al límite |
| 3 | `delay_streak` | Meses seguidos con `PAY_* > 0` contando desde `PAY_0` hacia atrás | Distingue un atraso reciente y continuo de uno viejo y ya superado |
| 4 | `delay_severity` | Suma de `max(PAY_*, 0)` en los 6 meses | Mezcla cuántos meses y qué tan largos fueron los atrasos |
| 5 | `severe_delay_months` | Número de meses con `PAY_* ≥ 2` | Un atraso de 2 meses o más concentra el incumplimiento (≥ 69 % en el EDA) |
| 6 | `n_zero_payments` | Número de meses con `PAY_AMT* = 0` | Un cliente que no paga nada es una señal directa |
| 7 | `n_minimum_payments` | Meses (de 5) con `PAY_AMT_k ≤ 5 % × max(BILL_AMT_{k+1}, 0)` | Quien paga solo lo mínimo acumula deuda. **El 5 % es un supuesto del grupo, no un dato.** Solo cuenta si la factura es positiva (con factura ≤ 0 no cuenta). No está en el paquete E |
| 8 | `log_limit` | `ln(LIMIT_BAL)` | Corrige la cola larga del límite; ayuda a los modelos lineales |
| 9 | `n_delayed_months` | Número de meses con `PAY_* > 0` | La correlación más alta con el objetivo (0,40) |
| 10 | Variantes sin `SEX` / sin `PAY_0` | Quitar la columna | Equidad (`SEX`) y dependencia del último estado de pago (`PAY_0`) |

**Ideas descartadas en la medición preliminar** (no se implementaron ni se volvieron a medir en el notebook; sus cifras están en §6): `max_atraso`, `atraso_medio`, `tendencia_pay`, `empeora`, conteos de estado, razón de pago, `pago_sobre_factura`, tendencia de factura, `n_factura_neg`, `crece_deuda`.

**Nombres en el código.**
El código usa nombres en inglés (regla del proyecto). Equivalencias con los nombres que aparecen en conversaciones y borradores anteriores:

| Nombre en el código | Nombre anterior | Qué es |
|---|---|---|
| `utilization_last` | `util_1` | Factura del último mes ÷ límite |
| `utilization_mean` | `util_media` | Factura media de 6 meses ÷ límite |
| `utilization_max` | `util_max` | Factura máxima de 6 meses ÷ límite |
| `pending_balance_last` | `saldo_pend_1` | (Factura de agosto − pago de septiembre) ÷ límite |
| `delay_streak` | `racha_atraso` | Meses seguidos con atraso desde el más reciente |
| `delay_severity` | `gravedad_atraso` | Suma de los meses de atraso |
| `severe_delay_months` | `atraso_grave` | Meses con 2 o más meses de atraso |
| `n_delayed_months` | `n_atraso` | Meses con algún atraso |
| `n_zero_payments` | `n_pago_cero` | Meses sin ningún pago |
| `n_minimum_payments` | `paga_minimo` | Meses con pago ≤ 5 % de una factura positiva |
| `log_limit` | `log_limite` | Logaritmo del límite |

(`n_atraso` sigue llamándose así dentro del EDA, notebook 02, donde nació.)

## 3. Regla de decisión

Una idea o paquete **se conserva** si cumple las tres:

1. **Se conoce al predecir** (§2, sin excepción).
2. **Mejora pareada > 1** en al menos un modelo, donde mejora pareada = mejora media del AUC ÷ desviación de las diferencias pliegue a pliegue (variante − referencia, sobre los mismos 15 pliegues).
3. **Gana en al menos 12 de los 15 pliegues** en ese modelo (80 %), para que no dependa de un solo pliegue, y **no empeora en promedio** en ningún modelo.

> **Cambio de regla, 2026-10-31 (decidido por el grupo después de ver el resultado).** La versión anterior pedía «mejora / ruido > 1», con el ruido igual a la desviación del AUC de la variante entre pliegues (regla del curso). Con ella, el paquete E no se confirmó con la semilla 99 (0,81), aunque su mejora fue casi igual en las tres semillas (+0,0088, +0,0088, +0,0089) y ganó en los 45 pliegues. La variación entre pliegues cambia con la semilla (0,006 a 0,011) y afecta igual a la referencia y a la variante, por lo que no sirve para comparar dos variantes sobre los mismos pliegues. El cambio se hizo **después** de conocer ese resultado, y por eso el notebook y el informe muestran las dos cifras. La regla pareada no mira el tamaño de la mejora, solo su consistencia.

> **Por paquete y por modelo.** La medición preliminar (§6) se hizo antes de fijar la regla y mostró que, con LightGBM, ninguna variable aislada la cumple. Por eso se evalúa por paquete y por modelo: un paquete que la cumpla en la logística y no empeore LightGBM se conserva.
>
> **Sesgo de selección:** el paquete E se eligió mirando los mismos 15 valores con los que se mide. Eso hace que su mejora (+0,0088 en la logística) sea **optimista**. La revisión H1 propone **validación cruzada anidada**: el paquete se elige en los pliegues internos y se evalúa en los externos, sin reajustar con ellos. Como mínimo, la medición definitiva se repite con **otras semillas de pliegues** (7 y 99) y el paquete solo se confirma si sigue cumpliendo la regla. Esto reduce el optimismo, pero no lo elimina. **La validación anidada no está hecha**: queda pendiente de aprobación. La prueba no se abrió.

## 4. Código y pruebas

| Pieza | Ruta |
|---|---|
| Funciones de variables (una por idea, puras, sin aprender de los datos) | `src/proyecto/features/build.py` |
| Constructor del `Pipeline` (variables + preprocesamiento + modelo) | `src/proyecto/models/train.py` (`build_pipeline`) |
| Pruebas (una por función, con un caso normal y uno borde) | `tests/test_features.py` |
| Notebook que orquesta (solo llama a las funciones) | `notebooks/04_features.ipynb` |
| Informe de decisiones | `reports/04_features.md` |

Reglas de implementación:
- Cada función recibe el `DataFrame` limpio (salida de `clean`) y devuelve **columnas nuevas** sin tocar el original ni borrar filas.
- División entre cero: si `LIMIT_BAL` es vacío o ≤ 0, el resultado es vacío (no infinito).
- Las funciones se aplican **dentro del `Pipeline`** (por ejemplo con `FunctionTransformer`), de modo que lo que se aprende de los datos (medianas, escalas) sigue ajustándose solo con entrenamiento.
- Prueba de seguridad: si una función usara el objetivo, `test_features_do_not_use_the_target` debe fallar (se calculan las variables con el objetivo alterado y deben salir idénticas).

## 5. Criterios de cierre

- ☒ Cada idea implementada tiene su fila en `reports/04_features.md` con su cifra (mejora pareada, mejora / ruido, pliegues) y su decisión escrita.
- ☒ Todas las variantes usan los **mismos pliegues** (§1).
- ☒ Ninguna variable usa información posterior a la decisión (`tests/test_features.py`).
- ☒ Paquete final elegido y escrito (E), con la métrica de LightGBM y la de la logística.
- ☒ `pytest` en verde desde la raíz.
- ☒ El conjunto de prueba sigue sin abrirse y `data/raw/` intacta (`git status`).
- ☐ Validación cruzada anidada (propuesta en la revisión H1, pendiente).
- ☐ Revisaron el código con `checklist-revision-ia.md` y cada integrante puede explicar una función.

## 6. Evidencia preliminar (solo entrenamiento, sin la prueba)

> **Las cifras definitivas están en `reports/04_features.md`** (notebook 04, con `build_pipeline`). Esta tabla es la medición previa que sirvió para elegir los candidatos; usa la regla original (mejora / ruido) y un modelo sin `Pipeline`, por eso difiere un poco.

Medición con 5 pliegues × 3 repeticiones, AUC. Referencia: LightGBM 0,7849 y logística 0,7697.

| Idea o paquete | LightGBM: mejora / ruido | LightGBM: % pliegues | Logística: mejora / ruido | Logística: % pliegues |
|---|---|---|---|---|
| `utilization_*` (3 variables) | 0,33 | 87 % | 0,12 | 87 % |
| `pending_balance_last` | 0,23 | 73 % | 0,10 | 80 % |
| `delay_streak` | 0,00 | 53 % | 0,34 | 100 % |
| `delay_severity` | 0,13 | 80 % | 0,00 | 27 % |
| `n_zero_payments` | 0,13 | 60 % | 0,82 | 100 % |
| `log_limit` | 0,00 | 53 % | 0,45 | 100 % |
| `n_delayed_months` | 0,04 | 53 % | −0,01 | 33 % |
| **Paquete E** (`utilization_mean`, `pending_balance_last`, `delay_streak`, `n_zero_payments`, `log_limit`) | 0,36 (+0,0022) | 80 % | **1,49 (+0,0088)** | 100 % |
| Paquete D (10 variables) | 0,49 (+0,0030) | 87 % | 1,45 (+0,0084) | 100 % |
| Sin `SEX` (1.ª medición, 5 pliegues) | −0,1 (−0,0008 de AUC) | — | no medido | — |
| Sin `PAY_0` (1.ª medición, 5 pliegues) | −3,1 (−0,030 de AUC) | — | no medido | — |

**Lectura:**
- LightGBM ya reconstruye casi todo a partir de las columnas originales: las variables nuevas aportan como máximo +0,003 de AUC.
- La logística sí gana: el paquete E sube su AUC de 0,7697 a 0,7785 (acorta la distancia con LightGBM) y cumple la regla completa.
- El paquete E es el **candidato** porque es corto (5 variables, fáciles de explicar al negocio) y rinde igual que el de 10.
- `n_delayed_months` y `delay_severity`, pese a su correlación alta con el objetivo, no suman: lo que dicen ya está en `PAY_*`.
- `PAY_0` domina: debe quedar escrito en el informe técnico como dependencia conocida.
