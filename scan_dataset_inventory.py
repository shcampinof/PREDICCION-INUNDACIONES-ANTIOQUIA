#!/usr/bin/env python3
"""
Scan project datasets and build a Phase-1 (MLP) oriented inventory.

Outputs:
1) CSV inventory with one row per file
2) JSON summary by extension/source/phase classification
3) Console summary
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

try:
    import rasterio  # type: ignore
except Exception:
    rasterio = None


SUPPORTED_EXTENSIONS = {
    "csv",
    "txt",
    "xlsx",
    "xls",
    "shp",
    "gpkg",
    "geojson",
    "json",
    "kml",
    "tif",
    "tiff",
    "tif.gz",
    "zip",
    "rar",
    "gz",
}


SOURCE_RULES = {
    "PRECIPITACION_IDEAM": {
        "default_phase": "NEEDS_LIGHT_PREPROCESSING",
        "phase1_reason": (
            "Series de precipitacion son valiosas para MLP, pero requieren limpieza "
            "y agregacion a municipio-fecha."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "ESTACIONES_IDEAM": {
        "default_phase": "NEEDS_LIGHT_PREPROCESSING",
        "phase1_reason": (
            "Metadatos y/o series de estaciones IDEAM; requieren unificacion de codigos, "
            "georreferenciacion y agregacion espacial."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "OPEN_METEO": {
        "default_phase": "NEEDS_LIGHT_PREPROCESSING",
        "phase1_reason": (
            "Datos meteorologicos tabulares suelen ser utiles para MLP, con mapeo "
            "a municipios y estandarizacion temporal."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "IGAC_TOPO_CARTO": {
        "default_phase": "NEEDS_HEAVY_GEOSPATIAL_PREPROCESSING",
        "phase1_reason": (
            "Topografia/cartografia es clave como covariable estatica, pero requiere "
            "procesamiento geoespacial (zonal stats) para volverlo tabular."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "MAPBIOMAS": {
        "default_phase": "NEEDS_HEAVY_GEOSPATIAL_PREPROCESSING",
        "phase1_reason": (
            "Cobertura del suelo aporta variables de susceptibilidad, pero requiere "
            "agregacion espacial por municipio (y eventualmente por anio)."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "TR_CENTROS_POBLADOS": {
        "default_phase": "NEEDS_LIGHT_PREPROCESSING",
        "phase1_reason": (
            "Capas de centros poblados sirven para exposicion/vulnerabilidad; "
            "normalmente necesitan joins espaciales y agregados simples."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "GROUND_TRUTH_INUNDACIONES": {
        "default_phase": "READY_FOR_PHASE1",
        "phase1_reason": (
            "Fuente candidata para variable objetivo (ocurrencia de inundacion), "
            "prioritaria para construir etiqueta binaria."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "CHIRPS": {
        "default_phase": "TOO_HEAVY_FOR_PHASE1_NOTEBOOK",
        "phase1_reason": (
            "Rasters diarios masivos; util para features de lluvia, pero conviene "
            "preagregar fuera de notebook a tabla municipio-fecha."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "SENTINEL1_GEE": {
        "default_phase": "NEEDS_HEAVY_GEOSPATIAL_PREPROCESSING",
        "phase1_reason": (
            "SAR aporta senales hidricas valiosas, pero requiere pipeline geoespacial "
            "y temporal para extraer estadisticos tabulares."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "SMAP_SUELO": {
        "default_phase": "NEEDS_HEAVY_GEOSPATIAL_PREPROCESSING",
        "phase1_reason": (
            "Humedad del suelo es util para inundaciones, pero suele venir en grillas "
            "o formatos pesados que requieren agregacion espacial/temporal."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "DHIME_NIVEL_IDEAM": {
        "default_phase": "NEEDS_LIGHT_PREPROCESSING",
        "phase1_reason": (
            "Niveles hidrologicos de IDEAM son utiles para predictor auxiliar; "
            "requieren limpieza temporal y consolidacion por area/municipio."
        ),
        "can_convert_to_municipio_fecha": True,
    },
    "DATOS_COMPRIMIDOS": {
        "default_phase": "IGNORE_FOR_NOW",
        "phase1_reason": (
            "Contenedor de archivos comprimidos (posibles duplicados o staging). "
            "Conviene priorizar datos ya extraidos para Fase 1."
        ),
        "can_convert_to_municipio_fecha": False,
    },
    "UNKNOWN": {
        "default_phase": "IGNORE_FOR_NOW",
        "phase1_reason": (
            "Fuente no identificada automaticamente; revisar manualmente si aporta "
            "a tabla municipio-fecha."
        ),
        "can_convert_to_municipio_fecha": False,
    },
}


SOURCE_KEYWORDS = [
    ("precipitacion ideam", "PRECIPITACION_IDEAM"),
    ("estaciones meteorologicas ideam", "ESTACIONES_IDEAM"),
    ("open-meteo", "OPEN_METEO"),
    ("topografia y cartografia igac", "IGAC_TOPO_CARTO"),
    ("mapbiomas", "MAPBIOMAS"),
    ("tr centros poblados", "TR_CENTROS_POBLADOS"),
    ("ground truth inundaciones", "GROUND_TRUTH_INUNDACIONES"),
    ("chirps", "CHIRPS"),
    ("sar sentinel-1", "SENTINEL1_GEE"),
    ("google earth engine", "SENTINEL1_GEE"),
    ("smap humedad suelo", "SMAP_SUELO"),
    ("dhime ideam nivel", "DHIME_NIVEL_IDEAM"),
    ("datos_comprimidos", "DATOS_COMPRIMIDOS"),
]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return normalized.lower()


def detect_extension(file_name: str) -> str:
    lower = file_name.lower()
    if lower.endswith(".tif.gz"):
        return "tif.gz"
    suffix = Path(lower).suffix
    return suffix.lstrip(".")


def guess_readable_type(extension: str) -> str:
    tabular = {"csv", "txt", "xlsx", "xls"}
    vector = {"shp", "gpkg", "geojson", "json", "kml"}
    raster = {"tif", "tiff", "tif.gz"}
    compressed = {"zip", "rar", "gz"}

    if extension in tabular:
        return "tabular"
    if extension in vector:
        return "geospatial_vector"
    if extension in raster:
        return "raster"
    if extension in compressed:
        return "compressed_archive"
    return "other"


def detect_source_group(relative_path: str) -> str:
    text = normalize_text(relative_path.replace("\\", "/"))
    for keyword, group in SOURCE_KEYWORDS:
        if keyword in text:
            return group
    return "UNKNOWN"


def detect_date_in_text(value: str) -> str:
    # yyyy-mm-dd or yyyy_mm_dd or yyyy.mm.dd or yyyymmdd
    patterns = [
        re.compile(r"(20\d{2})[._-](0[1-9]|1[0-2])[._-](0[1-9]|[12]\d|3[01])"),
        re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])"),
        re.compile(r"(20\d{2})[._-](0[1-9]|1[0-2])"),
        re.compile(r"\b(20\d{2})\b"),
    ]
    for regex in patterns:
        match = regex.search(value)
        if not match:
            continue
        parts = match.groups()
        if len(parts) == 3:
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}"
        if len(parts) == 1:
            return parts[0]
    return ""


def classify_phase(
    source_group: str,
    readable_type_guess: str,
    extension: str,
    size_mb: float,
) -> Tuple[str, str, bool]:
    rule = SOURCE_RULES[source_group]
    phase = rule["default_phase"]
    reason = rule["phase1_reason"]
    can_convert = bool(rule["can_convert_to_municipio_fecha"])

    if readable_type_guess == "other":
        return "IGNORE_FOR_NOW", "Archivo no identificado como insumo de datos principal.", False

    if extension in {"zip", "rar", "gz"} and source_group == "DATOS_COMPRIMIDOS":
        return phase, reason, can_convert

    if readable_type_guess == "compressed_archive":
        return "NEEDS_LIGHT_PREPROCESSING", (
            "Archivo comprimido; extraer y validar contenido antes de usarlo en tabla tabular."
        ), can_convert

    if source_group == "GROUND_TRUTH_INUNDACIONES" and readable_type_guess == "tabular":
        return "READY_FOR_PHASE1", reason, can_convert

    if readable_type_guess == "tabular" and size_mb > 250:
        return "NEEDS_LIGHT_PREPROCESSING", (
            "Tabular potencialmente util, pero voluminoso; conviene lectura por chunks y agregacion."
        ), can_convert

    if readable_type_guess == "raster":
        if source_group in {"CHIRPS", "SENTINEL1_GEE", "MAPBIOMAS"} or size_mb > 100:
            return "TOO_HEAVY_FOR_PHASE1_NOTEBOOK", (
                "Raster pesado; mejor preprocesar fuera del notebook y convertir a municipio-fecha."
            ), can_convert
        return "NEEDS_HEAVY_GEOSPATIAL_PREPROCESSING", (
            "Raster requiere extraccion geoespacial para generar features tabulares."
        ), can_convert

    if readable_type_guess == "geospatial_vector":
        if source_group in {"IGAC_TOPO_CARTO", "MAPBIOMAS", "SENTINEL1_GEE"}:
            return "NEEDS_HEAVY_GEOSPATIAL_PREPROCESSING", reason, can_convert
        return "NEEDS_LIGHT_PREPROCESSING", (
            "Vector util tras joins espaciales/agregacion a municipio."
        ), can_convert

    return phase, reason, can_convert


def inspect_tabular(path: Path, extension: str, sample_rows: int) -> str:
    if pd is not None:
        try:
            if extension in {"csv", "txt"}:
                df = pd.read_csv(
                    path,
                    nrows=sample_rows,
                    sep=None,
                    engine="python",
                    encoding="utf-8",
                    low_memory=False,
                )
            elif extension in {"xlsx", "xls"}:
                xls = pd.ExcelFile(path)
                first_sheet = xls.sheet_names[0] if xls.sheet_names else "Sheet1"
                df = pd.read_excel(path, sheet_name=first_sheet, nrows=sample_rows)
            else:
                return "tabular_inspection=unsupported_extension"
            cols = list(df.columns)
            return (
                f"sample_rows={len(df)}; n_columns={len(cols)}; "
                f"columns={cols[:12]}"
            )
        except Exception as exc:
            return f"tabular_inspection_error={type(exc).__name__}:{exc}"

    # Fallback without pandas (csv/txt only)
    if extension in {"csv", "txt"}:
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                head = fh.read(4096)
                fh.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(head)
                except csv.Error:
                    dialect = csv.excel
                reader = csv.reader(fh, dialect)
                rows = []
                for _, row in zip(range(sample_rows + 1), reader):
                    rows.append(row)
                if not rows:
                    return "sample_rows=0; n_columns=0; columns=[]"
                header = rows[0]
                return (
                    f"sample_rows={max(len(rows)-1, 0)}; n_columns={len(header)}; "
                    f"columns={header[:12]}"
                )
        except Exception as exc:
            return f"tabular_inspection_error={type(exc).__name__}:{exc}"

    return "tabular_inspection_skipped=pandas_not_available_for_excel"


def inspect_vector(path: Path, extension: str) -> str:
    if gpd is not None:
        try:
            try:
                sample = gpd.read_file(path, rows=1)
            except TypeError:
                sample = gpd.read_file(path).head(1)
            cols = list(sample.columns)
            geom_col = sample.geometry.name if hasattr(sample, "geometry") else "geometry"
            return f"n_columns={len(cols)}; geometry_col={geom_col}; columns={cols[:12]}"
        except Exception as exc:
            return f"vector_inspection_error={type(exc).__name__}:{exc}"

    if extension in {"geojson", "json"}:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
                features = payload.get("features") or []
                if features:
                    props = list((features[0].get("properties") or {}).keys())
                    return f"geojson_features~{len(features)}; properties={props[:12]}"
                return "geojson_features=0; properties=[]"
            if isinstance(payload, dict):
                return f"json_keys={list(payload.keys())[:12]}"
            return f"json_type={type(payload).__name__}"
        except Exception as exc:
            return f"vector_inspection_error={type(exc).__name__}:{exc}"

    return "vector_inspection_skipped=geopandas_not_available"


def inspect_raster(path: Path, extension: str) -> str:
    if extension == "tif.gz":
        return "raster_meta=compressed_tif_gz (extract required)"

    if rasterio is None:
        return "raster_inspection_skipped=rasterio_not_available"

    try:
        with rasterio.open(path) as ds:
            dtypes = list(ds.dtypes) if getattr(ds, "dtypes", None) else []
            return (
                f"width={ds.width}; height={ds.height}; bands={ds.count}; "
                f"crs={ds.crs}; dtype={dtypes[0] if dtypes else 'unknown'}"
            )
    except Exception as exc:
        return f"raster_inspection_error={type(exc).__name__}:{exc}"


def inspect_archive(path: Path, extension: str) -> str:
    if extension == "zip":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
            return f"zip_members={len(names)}; sample_members={names[:8]}"
        except Exception as exc:
            return f"zip_inspection_error={type(exc).__name__}:{exc}"
    if extension == "rar":
        return "rar_inspection=requires_optional_rar_support"
    if extension == "gz":
        stem = path.name[:-3] if path.name.lower().endswith(".gz") else path.name
        return f"gz_member_guess={stem}"
    return "archive_inspection_skipped"


def compact_note(note: str, max_len: int = 420) -> str:
    if len(note) <= max_len:
        return note
    return note[: max_len - 3] + "..."


def build_inventory(
    root: Path,
    outdir: Path,
    sample_rows: int,
    max_inspect_per_group: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    inventory: List[Dict[str, object]] = []
    inspect_counter: Dict[Tuple[str, str], int] = defaultdict(int)
    skipped_outdir = outdir.resolve()

    by_extension = Counter()
    by_source = Counter()
    by_phase = Counter()
    scan_errors: List[str] = []

    for current_dir, dirnames, files in os.walk(root):
        current_path = Path(current_dir)
        # Avoid scanning generated outputs.
        try:
            if current_path.resolve().is_relative_to(skipped_outdir):
                continue
        except Exception:
            pass

        # Skip hidden/system dirs.
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d.lower() not in {"__pycache__", "$recycle.bin"}
        ]

        for file_name in files:
            file_path = current_path / file_name
            if not file_path.exists():
                continue
            try:
                stat = file_path.stat()
            except Exception as exc:
                scan_errors.append(f"{file_path}: stat_error={type(exc).__name__}:{exc}")
                continue

            relative_path = file_path.relative_to(root).as_posix()
            extension = detect_extension(file_name)
            readable_type_guess = guess_readable_type(extension)
            source_group = detect_source_group(relative_path)
            date_detected = detect_date_in_text(f"{relative_path} {file_name}")
            size_mb = round(stat.st_size / (1024 * 1024), 4)

            suggested_phase, reason, can_convert = classify_phase(
                source_group=source_group,
                readable_type_guess=readable_type_guess,
                extension=extension,
                size_mb=size_mb,
            )

            notes = [
                f"phase1_reason={reason}",
                f"can_convert_to_municipio_fecha={'YES' if can_convert else 'NO'}",
            ]

            inspect_key = (source_group, readable_type_guess)
            should_inspect = readable_type_guess in {
                "tabular",
                "geospatial_vector",
                "raster",
                "compressed_archive",
            }
            if should_inspect:
                if inspect_counter[inspect_key] < max_inspect_per_group:
                    inspect_counter[inspect_key] += 1
                    try:
                        if readable_type_guess == "tabular":
                            inspect_result = inspect_tabular(file_path, extension, sample_rows)
                        elif readable_type_guess == "geospatial_vector":
                            inspect_result = inspect_vector(file_path, extension)
                        elif readable_type_guess == "raster":
                            inspect_result = inspect_raster(file_path, extension)
                        else:
                            inspect_result = inspect_archive(file_path, extension)
                        notes.append(inspect_result)
                    except Exception as exc:
                        notes.append(f"inspection_error={type(exc).__name__}:{exc}")
                else:
                    notes.append(
                        f"inspection_skipped=limit_reached({max_inspect_per_group}) for source/type"
                    )

            if extension in SUPPORTED_EXTENSIONS:
                by_extension[extension] += 1
            else:
                by_extension["untracked"] += 1
            by_source[source_group] += 1
            by_phase[suggested_phase] += 1

            inventory.append(
                {
                    "source_group": source_group,
                    "relative_path": relative_path,
                    "file_name": file_name,
                    "extension": extension if extension else "no_extension",
                    "size_mb": size_mb,
                    "readable_type_guess": readable_type_guess,
                    "date_detected_in_name": date_detected,
                    "suggested_phase": suggested_phase,
                    "notes": compact_note(" | ".join(notes)),
                }
            )

    source_summaries: Dict[str, Dict[str, object]] = {}
    for source_group, count in by_source.items():
        rule = SOURCE_RULES.get(source_group, SOURCE_RULES["UNKNOWN"])
        source_summaries[source_group] = {
            "file_count": count,
            "default_phase": rule["default_phase"],
            "phase1_reason": rule["phase1_reason"],
            "can_convert_to_municipio_fecha": rule["can_convert_to_municipio_fecha"],
        }

    summary: Dict[str, object] = {
        "scan_timestamp": dt.datetime.now().isoformat(),
        "root": str(root.resolve()),
        "total_files": len(inventory),
        "extension_counts": dict(sorted(by_extension.items())),
        "source_group_counts": dict(sorted(by_source.items())),
        "phase_counts": dict(sorted(by_phase.items())),
        "source_group_phase1_rationale": source_summaries,
        "libraries": {
            "pandas": pd is not None,
            "geopandas": gpd is not None,
            "rasterio": rasterio is not None,
        },
        "scan_errors": scan_errors[:1000],
    }
    return inventory, summary


def write_inventory_csv(out_csv: Path, inventory: List[Dict[str, object]]) -> None:
    fieldnames = [
        "source_group",
        "relative_path",
        "file_name",
        "extension",
        "size_mb",
        "readable_type_guess",
        "date_detected_in_name",
        "suggested_phase",
        "notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in inventory:
            writer.writerow(row)


def print_console_summary(summary: Dict[str, object], out_csv: Path, out_json: Path) -> None:
    print("\n=== Dataset Inventory Scan (Phase 1 / MLP) ===")
    print(f"Root: {summary['root']}")
    print(f"Total files scanned: {summary['total_files']}")
    print(f"Output CSV: {out_csv}")
    print(f"Output JSON: {out_json}")

    print("\n-- Phase counts --")
    for phase, count in summary["phase_counts"].items():
        print(f"{phase}: {count}")

    print("\n-- Top source groups --")
    source_counts = summary["source_group_counts"]
    sorted_sources = sorted(source_counts.items(), key=lambda kv: kv[1], reverse=True)
    for source, count in sorted_sources[:12]:
        print(f"{source}: {count}")

    print("\n-- Libraries --")
    libs = summary["libraries"]
    for lib_name, available in libs.items():
        print(f"{lib_name}: {'available' if available else 'missing'}")

    errors = summary.get("scan_errors", [])
    if errors:
        print(f"\nScan warnings/errors captured: {len(errors)} (see JSON)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan recursively and build a Phase-1 focused dataset inventory."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder to scan recursively.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for inventory CSV and summary JSON.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Sample rows for tabular inspection (default: 5).",
    )
    parser.add_argument(
        "--max-inspect-per-group",
        type=int,
        default=5,
        help="Max lightweight inspections per (source_group, readable_type) pair (default: 5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: root does not exist or is not a directory: {root}", file=sys.stderr)
        return 1

    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "dataset_inventory.csv"
    out_json = outdir / "dataset_inventory_summary.json"

    inventory, summary = build_inventory(
        root=root,
        outdir=outdir,
        sample_rows=max(args.sample_rows, 1),
        max_inspect_per_group=max(args.max_inspect_per_group, 1),
    )

    write_inventory_csv(out_csv, inventory)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print_console_summary(summary, out_csv, out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
