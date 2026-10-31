# Informe del EDA

> Resultado de `notebooks/02_eda.ipynb` (RiskVision, Grupo 6). Cada hallazgo termina en una decisión,
> que se implementará con pruebas en `src/proyecto/data/` en la etapa de preprocesamiento.

## Qué son los datos

- **Filas × columnas:** 30.000 × 25 (24 variables más `ID`).
- **Unidad de análisis (qué es una fila):** un cliente de tarjeta de crédito de un banco de Taiwán (abril a septiembre de 2005).
- **¿Una entidad aparece en varias filas?** No. Hay 30.000 `ID` distintos, uno por fila.
- **Objetivo y distribución:** `default payment next month`; 22,12 % incumple (6.636) y 77,88 % paga (23.364). Sin vacíos.
- **Línea base** (resultado de una regla trivial): «siempre paga» tiene exactitud 77,88 %, recall 0 % y ROC AUC 0,5000.

## Calidad de los datos

- Sin vacíos y sin texto: las 25 columnas son enteras. Sin duplicados completos; 35 filas idénticas si se ignora el `ID` (no se borran).
- `EDUCATION` tiene códigos sin documentar: 0 (14), 5 (280) y 6 (51), un total de 345 clientes (1,15 %). `MARRIAGE` tiene el código 0 en 54 clientes.
- `PAY_*` mezcla estados (-2, -1, 0) con meses de atraso (1 a 8); no existe `PAY_1` (se salta de `PAY_0` a `PAY_2`).
- 1.930 clientes (6,43 %) tienen algún `BILL_AMT` negativo; hay colas largas en `PAY_AMT*` (en `PAY_AMT2`, percentil 99 = 76.651 y máximo = 1.684.259).
- Las 6 facturas mensuales están muy correlacionadas entre sí (0,89 en promedio).
- 3.931 clientes facturan por encima de su límite en algún mes y 795 no tienen movimiento en 6 meses (plausibles).
- Línea base de negocio: la regla «marcar si `PAY_0` ≥ 2» marca al 10,4 %, tiene recall 32,8 % y precision 69,6 %, y su costo (5 a 1) es 0,775.
- Línea base de equidad (misma regla): recall 32,7 % en hombres y 32,9 % en mujeres; entre 30 % y 35 % según la edad.
- Sin datos personales (solo un `ID` anónimo). `SEX`, `AGE` y `MARRIAGE` son atributos sensibles.

## Sospechas de fuga de información

- No hay una fuga evidente: `PAY_*`, `BILL_AMT*` y `PAY_AMT*` cubren abril a septiembre, antes del mes que se predice.
- `PAY_0` (estado de septiembre) es la más fuerte (correlación 0,33; quien lleva 2 o más meses de atraso incumple el 69 % o más). Es legítima pero dominante; se mide el modelo con y sin ella.
- El riesgo está en el proceso: escalar o imputar antes de partir, o usar la prueba para decidir.

## Decisiones

| # | Hallazgo (con cifra) | Decisión | Motivo |
|---|---|---|---|
| 1 | 30.000 `ID` distintos: ningún cliente se repite | Partir por fila (80/20, estratificado, semilla 42) y excluir `ID` del modelo | No hay riesgo de que un cliente quede en entrenamiento y prueba; `ID` no describe comportamiento |
| 2 | 22,12 % incumple; «siempre paga» acierta 77,88 % con recall 0 % | Estratificar y usar ROC AUC (más recall y AUC-PR de control), no la exactitud | La exactitud premia no detectar a nadie; el error caro es el falso negativo |
| 3 | `EDUCATION` 0, 5 y 6 suman 345 clientes (1,15 %) y `MARRIAGE` 0 son 54, sin documentar | Agrupar esos códigos en «otros» (4 y 3) dentro del pipeline antes de codificar | Categorías sin significado conocido; evita categorías casi vacías |
| 4 | `PAY_*` mezcla estados con meses de atraso; la tasa de incumplimiento sube de 13–17 % a 69 % o más desde 2 meses de atraso | Conservarlas numéricas y ordenadas; probar una variable «atraso ≥ 2» en features | Es la señal más fuerte y el orden respeta el aumento del riesgo |
| 5 | 1.930 clientes (6,43 %) con `BILL_AMT` negativo; `PAY_AMT2` va de p99 = 76.651 a máximo = 1.684.259 | No aplicar `log(x)`; comparar `log1p` o Yeo-Johnson dentro del pipeline | Hay negativos y ceros; solo se acepta si mejora por validación cruzada |
| 6 | Mujeres incumplen 20,8 % y hombres 24,2 %; `AGE` y `MARRIAGE` también separan grupos | Revisar errores por grupo y probar con y sin `SEX` | Atributos sensibles: el modelo no debe discriminar |
| 7 | `PAY_0` es la más correlacionada con el objetivo (0,33) | Medir el modelo con y sin `PAY_0` | Que no dependa de una sola columna |
| 8 | 35 filas idénticas ignorando `ID`; 3.931 clientes facturan por encima del límite | No borrar filas | Parecen clientes reales distintos |
| 9 | `n_atraso` (meses con atraso) tiene correlación 0,40 con el objetivo, mayor que `PAY_0` (0,33); va de 11,7 % a 70,3 % | Crear `n_atraso` en la etapa de variables | Resume el historial de 6 meses en un número |
| 10 | Las 6 facturas `BILL_AMT` tienen correlación media 0,89; `ratio_pago` (−0,01) y `tendencia` (−0,03) casi no se relacionan con el objetivo | Resumir o regularizar las facturas; probar las nuevas sin esperar grandes mejoras | Redundancia confunde a la logística; sin señal lineal en las nuevas |
| 11 | La regla `PAY_0 >= 2` cuesta 0,775 (≈ «marcar a todos» 0,779); el modelo de referencia, 0,57 | Seguir con un modelo | El modelo ahorra cerca de un 27 % de costo sobre la mejor estrategia sin modelo |
| 12 | LightGBM 0,7813 y HistGradientBoosting 0,7806 empatan; XGBoost por defecto 0,7668; regresión logística 0,7272 | Gradient boosting como principal y regresión logística como contraste explicable | Mejor ROC AUC; la logística permite explicar |
| 13 | Calibración buena (predicho 0,568 frente a real 0,567 en el quintil más alto) | Usar probabilidades para el umbral de costos; confirmar con el modelo final | El umbral depende de que «30 %» signifique 30 % |
| 14 | De 14.400 a 19.200 filas el ROC AUC sube solo 0,0009 | Invertir en variables, no en más datos | La curva ya se aplanó |

## Qué queda abierto

- La proporción de costos 5 a 1 es un supuesto por validar; de ella depende el umbral.
- La comparación oficial de modelos (con optimización de cada candidato) se hace en `05_torneo.ipynb`.
- Medir el modelo con y sin `PAY_0` y con y sin `SEX` (etapa 4).
