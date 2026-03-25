# Predicci?n de Inundaciones en Antioquia

Proyecto de Redes Neuronales con foco actual en **Fase 1** (modelo superficial tabular para clasificaci?n binaria `flood_next_24h`).

## Estructura

- `raw/` (datos crudos pesados, no versionar)
- `processed/fase1/`
  - `inventory/`
    - `dataset_inventory.csv`
    - `dataset_inventory_summary.json`
  - `phase1_dataset_municipio_fecha.csv`
  - `phase1_dataset_municipio_fecha.parquet`
  - `phase1_dataset_metadata.json`
- `models/fase1_mlp/`
  - modelos entrenados, scaler e imputer
- `notebooks/`
  - `01_preprocesamiento_fase1.ipynb`
  - `02_mlp_baseline_fase1.ipynb`
- `reports/`

## Flujo recomendado (Fase 1)

1. Ejecutar inventario:
   - `python scan_dataset_inventory.py --root raw --outdir processed/fase1/inventory`
2. Construir dataset de Fase 1:
   - `python build_phase1_dataset.py --raw-root raw --outdir processed/fase1 --start-date 2018-01-01 --end-date 2024-12-31`
3. Revisar y documentar dataset:
   - `notebooks/01_preprocesamiento_fase1.ipynb`
4. Entrenar baseline y analizar predicciones:
   - `notebooks/02_mlp_baseline_fase1.ipynb`

## Estado del proyecto

- **Activo:** Fase 1 tabular (MLP baseline).
- **No activo por ahora:** Fase 2 (LSTM / modelado profundo).

## GitHub (sin subir `raw/`)

- `.gitignore` ya incluye `raw/`.
- Repositorio actual:
  - `https://github.com/shcampinof/PREDICCION-INUNDACIONES-ANTIOQUIA`
