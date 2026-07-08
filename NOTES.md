# H3 Notes

## Cache policy

Los scripts usan `H3/cache/` para evitar recalcular artefactos pesados.
El cache se construye automáticamente la primera vez. Para reconstruir
desde cero, eliminar los `.pkl` correspondientes.

En los notebooks, controlar con:

```python
FORCE_REBUILD = False  # reutiliza cache (normal)
FORCE_REBUILD = True   # recalcula todo desde cero
```

Cambiar a `True` solo si cambia el preprocesamiento, el split, la extracción
de categorías, la construcción de transiciones o la lógica de rankings
jerárquicos.

---

## Split val/test (cambio respecto a H2)

A partir de H3 final, los 50.000 usuarios evaluables se dividen en dos mitades:

- **val** (25.000): usado exclusivamente para tuning de hiperparámetros en
  `step2_tune_adaptive.py --mode tune`
- **test** (25.000): usado para reportar todas las métricas finales

La función `split_val_test(eval_users, seed)` en `cache_utils.py` implementa
este split de forma determinística (seed offset = seed + 1_000_003). Todos los
scripts (`step1`, `step2 --mode full`, `step3`, `step8`) la llaman
automáticamente y usan `final_test_users`.

Los archivos de salida llevan el sufijo `_test` (e.g.,
`step1_metrics_summary_sample_25000_test.csv`) para distinguirlos de las
corridas previas sin split (`_full`, `_sample_50000`).

**Importante:** los archivos con sufijo `_full` o `_sample_50000` en
`outputs/` son de corridas anteriores sin split correcto. No usarlos para
el paper.

---

## Corridas experimentales completadas

### Step 2 — tuning sobre val (25.000 usuarios)

```
outputs/step2_tuning_results_sample_25000_val.csv
outputs/step2_best_config_sample_25000_val.json
```

Mejor configuración por Recall@10:
```json
{
  "seq_min": 0.15,
  "seq_span": 0.7,
  "cat_base": 0.55,
  "cat_drop": 0.3,
  "hier_max": 0.2,
  "hier_power": 1.0,
  "pop": 0.05,
  "transition_mix": 0.75,
  "use_hierarchy": true
}
```

### Step 2 — evaluación final sobre test (25.000 usuarios)

```
outputs/step2_best_model_metrics_sample_25000_test.csv
```

| Modelo | Recall@10 | nDCG@10 |
|---|---:|---:|
| Adaptive Hierarchical Hybrid (tuned) | 0.1144 | 0.0686 |

### Step 3 — ItemKNN sobre test (25.000 usuarios)

```
outputs/step3_itemknn_summary_sample_25000_test.csv
```

| Modelo | Recall@10 | nDCG@10 |
|---|---:|---:|
| ItemKNN | 0.0791 | 0.0443 |

### Step 8 — GRU4Rec sobre test (25.000 usuarios)

Entrenado con 10 epochs (también disponible versión de 30 epochs en cache,
con resultados prácticamente iguales — el modelo convergió en ~20 epochs).

```
outputs/step8_gru4rec_summary_sample_25000_test.csv
cache/gru4rec_model_emb128_hid256_ep10_top50000.pt
```

| Modelo | Recall@10 | nDCG@10 | Observación |
|---|---:|---:|---|
| GRU4Rec (10 ep) | 0.0164 | 0.0100 | Convergido — 30 ep da resultado similar |

GRU4Rec queda por debajo de todos los baselines no-triviales. Esto es
consistente con la alta esparsidad del dataset (85% usuarios ≤2
interacciones) y se reporta como hallazgo: la señal categórica y jerárquica
es más informativa que la capacidad de representación neuronal cuando el
historial es escaso.

### Tablas consolidadas finales

```
outputs/final_metrics_summary_full.csv
outputs/final_metrics_short_history_full.csv
```

Incluyen todos los modelos: Random, Most Popular, Category Popularity,
Sequential Transition, Fixed Hybrid Seq+Category, ItemKNN, GRU4Rec,
Adaptive Hierarchical Hybrid (tuned).

---

## Figuras generadas (step9)

Script: `src/step9_figures.py`

Todas las figuras usan los datos de `final_metrics_summary_full.csv` y
`final_metrics_short_history_full.csv` — es decir, los números correctos
sobre el split test.

Figuras en `outputs/figures/`:
- `fig1_global_comparison.png`
- `fig2_short_history.png`
- `fig3_adaptive_weights.png`
- `fig4_ablation.png`
- `fig5_tuning_sensitivity.png`
- `fig6_tradeoff.png`
- `fig7_novelty_coverage.png`
