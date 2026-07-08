# H3 - Resumen final de experimentos

Este documento resume los pasos experimentales realizados para la entrega final
del proyecto RetailRocket Recommender System.

## Estructura

- `src/`: scripts reproducibles
- `notebooks/`: notebooks de análisis y visualización
- `outputs/`: resultados, tablas, figuras y ejemplos
- `cache/`: artefactos pesados precomputados para evitar recalcular

---

## Paso 1: métricas adicionales y ejemplos iniciales

Objetivo: evaluar los modelos existentes del H2 agregando métricas pedidas en
el feedback y generando ejemplos cualitativos.

Modelos:
- Random
- Most Popular
- Sequential Transition
- Category Popularity
- Fixed Hybrid Seq+Category

Métricas:
- Precision@10, Recall@10, nDCG@10
- Novelty@10, Category Diversity@10, Catalog Coverage@10

Archivos:
- Script: `src/step1_metrics_examples.py`
- Notebook: `notebooks/step1_metrics_examples.ipynb`

Outputs principales (sufijo `_sample_25000_test`):
- `outputs/step1_metrics_summary_sample_25000_test.csv`
- `outputs/step1_metrics_short_history_sample_25000_test.csv`
- `outputs/step1_metrics_detailed_sample_25000_test.csv`
- `outputs/step1_qualitative_examples_sample_25000_test.csv`

---

## Paso 2: híbrido adaptativo jerárquico (modelo propuesto)

Objetivo: transformar el híbrido fijo del H2 en una propuesta propia con pesos
adaptativos y backoff jerárquico de categorías.

Ideas principales:
- Usar `category_tree.csv` para construir ancestros de categorías
- Backoff jerárquico: categoría exacta → padre → ancestros → popularidad global
- Pesos adaptativos según largo del historial y confianza de transición del último ítem

Archivos:
- Notebook: `notebooks/step2_adaptive_hierarchical_hybrid.ipynb`
- Script de tuning/evaluación: `src/step2_tune_adaptive.py`

**Split val/test:** el tuning se realiza sobre 25.000 usuarios de validación
(`--mode tune`). Los resultados finales se reportan sobre los otros 25.000
usuarios de test (`--mode full`), garantizando que la selección de
hiperparámetros no contamina las métricas finales.

Grilla de tuning:
- `seq_min`: 0.15, 0.20, 0.25
- `seq_span`: 0.50, 0.60, 0.70
- `hier_max`: 0.08, 0.12, 0.16, 0.20

Mejor configuración (seleccionada sobre val):
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

Outputs:
- `outputs/step2_tuning_results_sample_25000_val.csv`
- `outputs/step2_best_config_sample_25000_val.json`
- `outputs/step2_best_model_metrics_sample_25000_test.csv`

---

## Paso 3: baseline colaborativo ItemKNN

Objetivo: agregar un baseline colaborativo más fuerte que los baselines iniciales.

Método: ItemKNN implícito basado en co-ocurrencias ítem-ítem con pesos de
eventos (view=1, addtocart=3, transaction=5).

Archivo: `src/step3_itemknn.py`

Outputs (sufijo `_sample_25000_test`):
- `outputs/step3_itemknn_summary_sample_25000_test.csv`
- `outputs/step3_itemknn_short_history_sample_25000_test.csv`
- `outputs/step3_itemknn_detailed_sample_25000_test.csv`

---

## Paso 4: tradeoff novedad/diversidad y análisis por usuario

Objetivo:
- Mostrar que novelty/diversity/coverage son métricas complementarias a recall
- Probar reranking post-hoc para promover novedad/diversidad
- Generar ejemplos por usuario

Archivos:
- Notebook: `notebooks/step4_analisis_cualitativo_y_tradeoffs.ipynb`
- Script de cache: `src/step4_cache_analysis_artifacts.py`

Outputs:
- `outputs/step4_reranking_tradeoffs_sample_50000.csv`
- `outputs/step4_user_level_analysis.csv`

Lectura:
- El reranking puede aumentar novelty/diversity levemente
- Ese aumento tiene un costo pequeño en Recall@10/nDCG@10
- El reranking no es parte del modelo final; es un análisis de tradeoff

---

## Paso 5: análisis final de métricas y gráficos

Objetivo: visualizar cada métrica y generar insumos para poster/paper.

Archivos:
- Notebook: `notebooks/step5_analisis_final_metricas.ipynb`

Outputs:
- `outputs/step5_metric_rankings.csv`
- `outputs/step5_relative_improvements.csv`
- Figuras en `outputs/figures/` (ver paso 9 para figuras actualizadas)

---

## Paso 6: casos de usuario explicados

Objetivo: análisis cualitativo interpretable comparando modelos por usuario.

Para cada usuario se explica:
- Largo de historial y confianza secuencial
- Pesos adaptativos asignados
- Recomendaciones de cada modelo con justificación
- Hit/miss por modelo

Archivos:
- Notebook: `notebooks/step6_casos_usuario_explicados.ipynb`

Outputs:
- `outputs/step6_casos_usuario_explicados.csv`
- `outputs/step6_comparacion_rankings_por_usuario.csv`
- `outputs/step6_explicaciones_por_modelo.csv`

---

## Paso 8: baseline neuronal GRU4Rec

Objetivo: agregar un baseline secuencial neuronal para comparar contra el
modelo propuesto y los baselines simples.

Método: GRU4Rec (Hidasi et al., 2015). RNN con celda GRU entrenada con
cross-entropy sobre secuencias de interacciones. Vocabulario: top-50.000 ítems.

Archivo: `src/step8_gru4rec.py`

Hiperparámetros:
- `embedding_dim`: 128, `hidden_dim`: 256, `num_layers`: 1
- `dropout`: 0.2, `epochs`: 10, `batch_size`: 512, `lr`: 1e-3

Resultado: Recall@10 = 0.0164, nDCG@10 = 0.0100. GRU4Rec queda por debajo de
todos los métodos basados en popularidad y transiciones, confirmando que en
datasets de alta esparsidad (85% usuarios con ≤2 interacciones), los modelos
neuronales no tienen suficiente señal para superar a métodos híbridos simples.

Para re-evaluar sin reentrenar:
```bash
python3 src/step8_gru4rec.py --max-eval-users 50000 --epochs 10 --skip-training
```

Outputs:
- `outputs/step8_gru4rec_summary_sample_25000_test.csv`
- `outputs/step8_gru4rec_short_history_sample_25000_test.csv`
- `outputs/step8_gru4rec_detailed_sample_25000_test.csv`

---

## Paso 9: figuras actualizadas

Objetivo: generar todas las figuras para poster y paper con los resultados
finales sobre el split test correcto.

Archivo: `src/step9_figures.py`

Figuras en `outputs/figures/`:
- `fig1_global_comparison.png`: Recall@10 + nDCG@10 todos los modelos
- `fig2_short_history.png`: usuarios con ≤2 interacciones
- `fig3_adaptive_weights.png`: pesos adaptativos para usuario con historial corto
- `fig4_ablation.png`: ablation study por componente
- `fig5_tuning_sensitivity.png`: sensibilidad a hiperparámetros (sobre val)
- `fig6_tradeoff.png`: scatter relevancia vs. diversidad
- `fig7_novelty_coverage.png`: novelty y catalog coverage por modelo

---

## Consolidación de tablas finales

Script: `src/collect_final_tables.py`

Une resultados de step1, step2, step3 y step8.

Outputs:
- `outputs/final_metrics_summary_full.csv`
- `outputs/final_metrics_short_history_full.csv`

---

## Resultados finales

### Global (25.000 usuarios de test)

| Modelo | Recall@10 | nDCG@10 | Novelty@10 | Cat.Div@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Adaptive Hierarchical Hybrid (tuned) | 0.1144 | 0.0686 | 13.60 | 0.507 | 0.230 |
| Fixed Hybrid Seq+Category | 0.1096 | 0.0685 | 13.16 | 0.623 | 0.228 |
| Sequential Transition | 0.1025 | 0.0664 | 13.24 | 0.730 | 0.256 |
| ItemKNN | 0.0791 | 0.0443 | 15.42 | 0.741 | 0.371 |
| Category Popularity | 0.0750 | 0.0399 | 13.51 | 0.070 | 0.059 |
| GRU4Rec | 0.0164 | 0.0100 | 13.78 | 0.931 | 0.143 |
| Most Popular | 0.0022 | 0.0009 | 10.20 | 0.955 | 0.000 |
| Random | 0.0000 | 0.0000 | 18.78 | 0.995 | 0.810 |

### Usuarios con historial corto (≤2 interacciones)

| Modelo | Recall@10 | nDCG@10 | Novelty@10 | Cat.Div@10 |
|---|---:|---:|---:|---:|
| Adaptive Hierarchical Hybrid (tuned) | 0.1263 | 0.0763 | 13.53 | 0.503 |
| Fixed Hybrid Seq+Category | 0.1209 | 0.0763 | 13.05 | 0.640 |
| Sequential Transition | 0.1120 | 0.0737 | 13.10 | 0.744 |
| ItemKNN | 0.0868 | 0.0498 | 15.08 | 0.755 |
| Category Popularity | 0.0832 | 0.0447 | 13.46 | 0.090 |
| GRU4Rec | 0.0173 | 0.0107 | 13.58 | 0.938 |
| Most Popular | 0.0025 | 0.0011 | 10.20 | 0.956 |
| Random | 0.0000 | 0.0000 | 18.79 | 0.995 |

---

## Cache

Artefactos importantes:
- `cache/events_preprocessed.pkl`
- `cache/temporal_leave_one_out_split.pkl`
- `cache/item_to_category.pkl`
- `cache/category_parent_map.pkl`
- `cache/category_to_ancestors.pkl`
- `cache/transition_artifacts.pkl`
- `cache/hierarchical_category_artifacts.pkl`
- `cache/direct_category_artifacts.pkl`
- `cache/seen_items.pkl`
- `cache/train_sequences.pkl`
- `cache/train_history_lengths.pkl`
- `cache/global_popularity.pkl`
- `cache/catalog_items.pkl`
- `cache/itemknn_neighbors_usercap50_n200.pkl`
- `cache/gru4rec_item_index_top50000.pkl`
- `cache/gru4rec_model_emb128_hid256_ep10_top50000.pt`
- `cache/step4_analysis_artifacts_sample_50000.pkl`

---

## Limitaciones

- Las categorías e ítems son anónimos; no se pueden interpretar como productos reales.
- El modelo final optimiza relevancia top-k, no novelty/diversity directamente.
- La diversidad interna es menor que en ItemKNN y Sequential Transition;
  esto se presenta como tradeoff relevancia/diversidad en el paper.
- GRU4Rec no converge a resultados competitivos en 10 epochs (ni en 30);
  esto es consistente con la alta esparsidad del dataset y se reporta como
  hallazgo, no como limitación del experimento.

---

## Archivos recomendados para el paper

Tablas:
- `outputs/final_metrics_summary_full.csv`
- `outputs/final_metrics_short_history_full.csv`
- `outputs/step5_relative_improvements.csv`
- `outputs/step6_casos_usuario_explicados.csv`

Figuras:
- `outputs/figures/fig1_global_comparison.png`
- `outputs/figures/fig2_short_history.png`
- `outputs/figures/fig3_adaptive_weights.png`
- `outputs/figures/fig5_tuning_sensitivity.png`
- `outputs/figures/fig6_tradeoff.png`

Notebooks para revisar:
- `notebooks/step2_adaptive_hierarchical_hybrid.executed.ipynb`
- `notebooks/step5_analisis_final_metricas.executed.ipynb`
- `notebooks/step6_casos_usuario_explicados.executed.ipynb`
