#!/usr/bin/env python3
"""
Build a minimum viable tabular dataset for Phase 1 (MLP) flood prediction.

Target grain:
    One row per municipio-fecha (daily).

Scope (MVP only):
    - Binary target from historical flood-related emergency events.
    - Daily meteorological features (Open-Meteo, station-level -> municipality daily agg).
    - Daily precipitation features (IDEAM, station/municipality -> municipality daily agg).
    - Optional DHIME level features from zipped CSV files.

No LSTM / no long sequence modeling logic is included.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover - handled at runtime
    pd = None  # type: ignore


FLOOD_EVENT_KEYWORDS = [
    "inund",
    "crecient",
    "avenida torrencial",
    "desbord",
]


@dataclass
class BuildContext:
    raw_root: Path
    outdir: Path
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]
    chunksize: int
    include_dhime: bool
    verbose: bool
    warnings: List[str]
    used_sources: Dict[str, Optional[str]]


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def normalize_key(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def clean_station_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    s = s.str.replace(r"[^\dA-Za-z]", "", regex=True)
    return s


def detect_delimiter(path: Path) -> str:
    sample = path.read_bytes()[:4096]
    text = sample.decode("utf-8", errors="replace")
    candidates = [",", ";", "\t", "|"]
    counts = {sep: text.count(sep) for sep in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def smart_read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    sep = detect_delimiter(path)
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=sep,
                encoding=enc,
                low_memory=False,
                nrows=nrows,
            )
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Cannot read CSV {path}: {last_exc}")


def smart_read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".csv", ".txt"}:
        return smart_read_csv(path, nrows=nrows)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def list_candidate_files(root: Path, suffixes: Sequence[str]) -> List[Path]:
    suffix_set = {s.lower() for s in suffixes}
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in suffix_set:
                out.append(p)
    return out


def pick_file_by_keywords(
    files: Sequence[Path],
    include_keywords: Sequence[str],
    exclude_keywords: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    include = [normalize_text(k) for k in include_keywords]
    exclude = [normalize_text(k) for k in (exclude_keywords or [])]
    scored: List[Tuple[int, int, Path]] = []
    for p in files:
        key = normalize_text(str(p))
        if exclude and any(k in key for k in exclude):
            continue
        score = sum(1 for k in include if k in key)
        if score > 0:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            scored.append((score, size, p))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def normalized_column_map(df: pd.DataFrame) -> Dict[str, str]:
    return {normalize_key(c): c for c in df.columns}


def find_column(
    df: pd.DataFrame,
    include_tokens: Sequence[str],
    exclude_tokens: Optional[Sequence[str]] = None,
) -> Optional[str]:
    colmap = normalized_column_map(df)
    include = [normalize_key(t) for t in include_tokens]
    exclude = [normalize_key(t) for t in (exclude_tokens or [])]

    best_col = None
    best_score = -1
    for ncol, original in colmap.items():
        if exclude and any(t in ncol for t in exclude):
            continue
        score = sum(1 for t in include if t in ncol)
        if score > best_score:
            best_score = score
            best_col = original
    return best_col if best_score > 0 else None


def coerce_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=False)
    try:
        dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt


def clip_dates(
    df: pd.DataFrame,
    date_col: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    out = df
    if start_date is not None:
        out = out[out[date_col] >= start_date]
    if end_date is not None:
        out = out[out[date_col] <= end_date]
    return out


def ensure_muni_key(df: pd.DataFrame, municipio_col: str) -> pd.DataFrame:
    out = df.copy()
    out["municipio_name"] = out[municipio_col].astype(str).str.strip()
    out["municipio_key"] = out["municipio_name"].map(normalize_key)
    out = out[out["municipio_key"] != ""]
    return out


def load_station_catalog(
    ctx: BuildContext, all_table_files: Sequence[Path]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    station_file = pick_file_by_keywords(
        all_table_files,
        include_keywords=["estaciones", "ideam"],
        exclude_keywords=["chirps", "sentinel", "mapbiomas"],
    )
    if station_file is None:
        raise FileNotFoundError("No station catalog file found (expected estaciones IDEAM).")
    ctx.used_sources["stations"] = str(station_file)
    log(f"[stations] Using: {station_file}", ctx.verbose)

    df = smart_read_table(station_file)
    code_col = find_column(df, ["codigo"], ["sensor"])
    muni_col = find_column(df, ["municipio"])
    dept_col = find_column(df, ["departamento"])
    lat_col = find_column(df, ["latitud"])
    lon_col = find_column(df, ["longitud"])
    alt_col = find_column(df, ["altitud"])

    if code_col is None or muni_col is None:
        raise ValueError(
            f"Station file missing key columns. code_col={code_col}, muni_col={muni_col}"
        )

    keep = [code_col, muni_col] + [c for c in [dept_col, lat_col, lon_col, alt_col] if c]
    out = df[keep].copy()
    out = out.rename(columns={code_col: "station_code", muni_col: "municipio_raw"})
    if dept_col:
        out = out.rename(columns={dept_col: "departamento_raw"})
    if lat_col:
        out = out.rename(columns={lat_col: "latitud_raw"})
    if lon_col:
        out = out.rename(columns={lon_col: "longitud_raw"})
    if alt_col:
        out = out.rename(columns={alt_col: "altitud_raw"})

    out["station_code"] = clean_station_code(out["station_code"])
    out = ensure_muni_key(out, "municipio_raw")

    if "departamento_raw" in out.columns:
        dept_norm = out["departamento_raw"].map(normalize_text)
        ant = out[dept_norm.str.contains("antioquia", na=False)]
        if len(ant) > 0:
            out = ant
        else:
            ctx.warnings.append(
                "Station catalog has no explicit Antioquia department match; using full file."
            )

    for c in ["latitud_raw", "longitud_raw", "altitud_raw"]:
        if c in out.columns:
            out[c] = (
                out[c]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            out[c] = pd.to_numeric(out[c], errors="coerce")

    agg_cols: Dict[str, str] = {"station_code": "nunique", "municipio_name": "first"}
    if "latitud_raw" in out.columns:
        agg_cols["latitud_raw"] = "mean"
    if "longitud_raw" in out.columns:
        agg_cols["longitud_raw"] = "mean"
    if "altitud_raw" in out.columns:
        agg_cols["altitud_raw"] = "mean"

    muni_static = (
        out.groupby("municipio_key", as_index=False)
        .agg(agg_cols)
        .rename(columns={"station_code": "stations_in_municipio"})
    )
    return out, muni_static


def build_target_from_ground_truth(
    ctx: BuildContext,
    all_table_files: Sequence[Path],
) -> pd.DataFrame:
    gt_file = pick_file_by_keywords(
        all_table_files,
        include_keywords=["historico", "emergencias", "antioquia"],
    )
    if gt_file is None:
        gt_file = pick_file_by_keywords(
            all_table_files,
            include_keywords=["ground", "truth", "inund"],
        )
    if gt_file is None:
        raise FileNotFoundError("No ground truth event table found.")

    ctx.used_sources["ground_truth"] = str(gt_file)
    log(f"[target] Using: {gt_file}", ctx.verbose)
    df = smart_read_table(gt_file)

    muni_col = find_column(df, ["municipio"])
    # Prefer explicit "Fecha Evento" style names first.
    date_col = (
        find_column(df, ["fecha_evento"])
        or find_column(df, ["fecha"], ["evento_norm", "tipo_evento", "evento"])
        or find_column(df, ["date"])
    )
    event_col = (
        find_column(df, ["evento_norm"])
        or find_column(df, ["tipo_evento"], ["fecha"])
        or find_column(df, ["evento"], ["fecha"])
    )
    dept_col = find_column(df, ["departamento"])

    if muni_col is None or date_col is None or event_col is None:
        raise ValueError(
            f"Ground truth missing required columns. municipio={muni_col}, fecha={date_col}, evento={event_col}"
        )

    work = df[[muni_col, date_col, event_col] + ([dept_col] if dept_col else [])].copy()
    work = work.rename(
        columns={
            muni_col: "municipio_raw",
            date_col: "event_time_raw",
            event_col: "event_type_raw",
        }
    )
    if dept_col:
        work = work.rename(columns={dept_col: "departamento_raw"})

    if "departamento_raw" in work.columns:
        dep = work["departamento_raw"].map(normalize_text)
        ant = work[dep.str.contains("antioquia", na=False)]
        if len(ant) > 0:
            work = ant

    work["event_time"] = coerce_datetime(work["event_time_raw"])
    work = work.dropna(subset=["event_time"])
    work["date"] = work["event_time"].dt.floor("D")
    work = clip_dates(work, "date", ctx.start_date, ctx.end_date)
    work = ensure_muni_key(work, "municipio_raw")

    event_norm = work["event_type_raw"].map(normalize_text)
    work["is_flood_event"] = 0
    for kw in FLOOD_EVENT_KEYWORDS:
        work.loc[event_norm.str.contains(kw, na=False), "is_flood_event"] = 1

    daily_event = (
        work.groupby(["municipio_key", "date"], as_index=False)
        .agg(
            flood_event_today=("is_flood_event", "max"),
            flood_events_count_today=("is_flood_event", "sum"),
            municipio_name=("municipio_name", "first"),
        )
    )

    target = daily_event[["municipio_key", "date", "flood_event_today"]].copy()
    target["date"] = target["date"] - pd.Timedelta(days=1)
    target = target.rename(columns={"flood_event_today": "flood_next_24h"})
    target = (
        target.groupby(["municipio_key", "date"], as_index=False)
        .agg(flood_next_24h=("flood_next_24h", "max"))
    )

    municipality_ref = daily_event[["municipio_key", "municipio_name"]].drop_duplicates()
    target = target.merge(municipality_ref, on="municipio_key", how="left")
    return target


def load_open_meteo_features(
    ctx: BuildContext,
    all_table_files: Sequence[Path],
    stations_lookup: pd.DataFrame,
) -> pd.DataFrame:
    meteo_file = pick_file_by_keywords(
        all_table_files,
        include_keywords=["meteo", "antioquia", "progreso"],
    )
    if meteo_file is None:
        meteo_file = pick_file_by_keywords(
            all_table_files,
            include_keywords=["open", "meteo"],
        )
    if meteo_file is None:
        raise FileNotFoundError("No Open-Meteo file found.")

    ctx.used_sources["open_meteo"] = str(meteo_file)
    log(f"[meteo] Using: {meteo_file}", ctx.verbose)

    station_map = stations_lookup[["station_code", "municipio_key", "municipio_name"]].drop_duplicates()
    sep = detect_delimiter(meteo_file)

    chunks: List[pd.DataFrame] = []
    reader = pd.read_csv(
        meteo_file,
        sep=sep,
        encoding="utf-8",
        low_memory=False,
        chunksize=ctx.chunksize,
    )

    processed_rows = 0
    for chunk in reader:
        processed_rows += len(chunk)
        col_time = find_column(chunk, ["time"]) or find_column(chunk, ["fecha"])
        col_station = find_column(chunk, ["codigoestacion"]) or find_column(chunk, ["codigo"])
        col_temp = find_column(chunk, ["temperature", "2m"]) or find_column(chunk, ["temp"])
        col_hum = find_column(chunk, ["humidity"]) or find_column(chunk, ["humedad"])
        col_wind = find_column(chunk, ["wind", "speed"]) or find_column(chunk, ["viento"])
        col_muni = find_column(chunk, ["municipio"])

        if col_time is None:
            continue
        if col_temp is None and col_hum is None and col_wind is None:
            continue

        keep_cols = [col_time] + [c for c in [col_station, col_temp, col_hum, col_wind, col_muni] if c]
        work = chunk[keep_cols].copy()
        work = work.rename(columns={col_time: "time_raw"})
        if col_station:
            work = work.rename(columns={col_station: "station_code_raw"})
        if col_temp:
            work = work.rename(columns={col_temp: "temp_raw"})
        if col_hum:
            work = work.rename(columns={col_hum: "rh_raw"})
        if col_wind:
            work = work.rename(columns={col_wind: "wind_raw"})
        if col_muni:
            work = work.rename(columns={col_muni: "municipio_raw"})

        work["time"] = coerce_datetime(work["time_raw"])
        work = work.dropna(subset=["time"])
        work["date"] = work["time"].dt.floor("D")
        work = clip_dates(work, "date", ctx.start_date, ctx.end_date)

        if "station_code_raw" in work.columns:
            work["station_code"] = clean_station_code(work["station_code_raw"])
            work = work.merge(station_map, on="station_code", how="left")

        if ("municipio_key" not in work.columns) and ("municipio_raw" in work.columns):
            work = ensure_muni_key(work, "municipio_raw")
        if "municipio_key" not in work.columns:
            continue

        for src, dst in [("temp_raw", "temp"), ("rh_raw", "rh"), ("wind_raw", "wind")]:
            if src in work.columns:
                work[dst] = pd.to_numeric(work[src], errors="coerce")

        agg_spec = {
            "meteo_obs_count_24h": ("date", "size"),
            "meteo_station_count_24h": ("station_code", "nunique"),
        }
        if "temp" in work.columns:
            agg_spec["meteo_temp_mean_24h"] = ("temp", "mean")
            agg_spec["meteo_temp_max_24h"] = ("temp", "max")
        if "rh" in work.columns:
            agg_spec["meteo_rh_mean_24h"] = ("rh", "mean")
            agg_spec["meteo_rh_max_24h"] = ("rh", "max")
        if "wind" in work.columns:
            agg_spec["meteo_wind_mean_24h"] = ("wind", "mean")
            agg_spec["meteo_wind_max_24h"] = ("wind", "max")

        part = (
            work.groupby(["municipio_key", "date"], as_index=False)
            .agg(**agg_spec)
        )
        if "municipio_name" in work.columns:
            muni_ref = work[["municipio_key", "municipio_name"]].dropna().drop_duplicates()
            part = part.merge(muni_ref, on="municipio_key", how="left")
        chunks.append(part)

    if not chunks:
        raise ValueError("No valid Open-Meteo chunks processed.")

    log(f"[meteo] Processed rows: {processed_rows:,}", ctx.verbose)
    out = pd.concat(chunks, ignore_index=True)

    agg_final = {
        "meteo_obs_count_24h": "sum",
        "meteo_station_count_24h": "max",
        "meteo_temp_mean_24h": "mean",
        "meteo_temp_max_24h": "max",
        "meteo_rh_mean_24h": "mean",
        "meteo_rh_max_24h": "max",
        "meteo_wind_mean_24h": "mean",
        "meteo_wind_max_24h": "max",
    }
    agg_final = {k: v for k, v in agg_final.items() if k in out.columns}
    out = out.groupby(["municipio_key", "date"], as_index=False).agg(agg_final)
    return out


def load_precip_features(
    ctx: BuildContext,
    all_table_files: Sequence[Path],
    stations_lookup: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    precip_file = pick_file_by_keywords(
        all_table_files,
        include_keywords=["precipitacion", "antioquia", "final"],
    )
    if precip_file is None:
        precip_file = pick_file_by_keywords(
            all_table_files,
            include_keywords=["precipitacion", "antioquia"],
        )
    if precip_file is None:
        ctx.warnings.append("Precipitation source not found. Continuing without precip features.")
        return None
    ctx.used_sources["precip_ideam"] = str(precip_file)
    log(f"[precip] Using: {precip_file}", ctx.verbose)

    ext = precip_file.suffix.lower()
    if ext == ".parquet":
        try:
            df = pd.read_parquet(precip_file)
        except Exception as exc:
            ctx.warnings.append(f"Cannot read parquet precipitation file: {exc}")
            return None
    elif ext in {".csv", ".txt", ".xlsx", ".xls"}:
        df = smart_read_table(precip_file)
    else:
        ctx.warnings.append(f"Unsupported precipitation format: {precip_file}")
        return None

    date_col = find_column(df, ["fecha", "observacion"]) or find_column(df, ["fecha"])
    value_col = (
        find_column(df, ["valor", "observado"])
        or find_column(df, ["precipit"])
        or find_column(df, ["lluvia"])
        or find_column(df, ["valor"])
    )
    station_col = find_column(df, ["codigo", "estacion"])
    muni_col = find_column(df, ["municipio"])
    sensor_col = find_column(df, ["descripcion", "sensor"])

    if date_col is None or value_col is None:
        ctx.warnings.append(
            f"Precip file missing required columns (fecha/valor). date={date_col}, value={value_col}"
        )
        return None

    keep = [date_col, value_col] + [c for c in [station_col, muni_col, sensor_col] if c]
    work = df[keep].copy()
    work = work.rename(columns={date_col: "time_raw", value_col: "precip_raw"})
    if station_col:
        work = work.rename(columns={station_col: "station_code_raw"})
    if muni_col:
        work = work.rename(columns={muni_col: "municipio_raw"})
    if sensor_col:
        work = work.rename(columns={sensor_col: "sensor_raw"})

    if "sensor_raw" in work.columns:
        sensor_norm = work["sensor_raw"].map(normalize_text)
        mask = sensor_norm.str.contains("precipit|lluvia", na=False)
        if mask.any():
            work = work[mask]

    work["time"] = coerce_datetime(work["time_raw"])
    work = work.dropna(subset=["time"])
    work["date"] = work["time"].dt.floor("D")
    work = clip_dates(work, "date", ctx.start_date, ctx.end_date)

    work["precip_mm"] = pd.to_numeric(work["precip_raw"], errors="coerce")
    work.loc[work["precip_mm"] < 0, "precip_mm"] = np.nan
    work = work.dropna(subset=["precip_mm"])

    station_map = stations_lookup[["station_code", "municipio_key", "municipio_name"]].drop_duplicates()
    if "station_code_raw" in work.columns:
        work["station_code"] = clean_station_code(work["station_code_raw"])
        work = work.merge(station_map, on="station_code", how="left")

    if ("municipio_key" not in work.columns) and ("municipio_raw" in work.columns):
        work = ensure_muni_key(work, "municipio_raw")
    if "municipio_key" not in work.columns:
        ctx.warnings.append("Precipitation file could not be mapped to municipio.")
        return None

    out = (
        work.groupby(["municipio_key", "date"], as_index=False)
        .agg(
            precip_mm_24h_sum=("precip_mm", "sum"),
            precip_mm_24h_max=("precip_mm", "max"),
            precip_obs_count_24h=("precip_mm", "size"),
            precip_station_count_24h=("station_code", "nunique"),
        )
    )
    out = out.sort_values(["municipio_key", "date"])
    out["precip_mm_48h_sum"] = (
        out.groupby("municipio_key")["precip_mm_24h_sum"]
        .transform(lambda s: s.rolling(2, min_periods=1).sum())
    )
    out["precip_mm_72h_sum"] = (
        out.groupby("municipio_key")["precip_mm_24h_sum"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    return out


def load_dhime_features(
    ctx: BuildContext,
    stations_lookup: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if not ctx.include_dhime:
        return None

    dhime_root = ctx.raw_root / "DHIME IDEAM nivel"
    if not dhime_root.exists():
        ctx.warnings.append("DHIME folder not found; skipping DHIME features.")
        return None

    zips = sorted(dhime_root.rglob("*.zip"))
    if not zips:
        ctx.warnings.append("No DHIME zip files found; skipping DHIME features.")
        return None
    ctx.used_sources["dhime"] = str(dhime_root)
    log(f"[dhime] Reading {len(zips)} zip parts", ctx.verbose)

    station_map = stations_lookup[["station_code", "municipio_key", "municipio_name"]].drop_duplicates()
    pieces: List[pd.DataFrame] = []
    for zpath in zips:
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                members = zf.namelist()
                if not members:
                    continue
                with zf.open(members[0]) as fh:
                    part = pd.read_csv(fh, low_memory=False)
        except Exception as exc:
            ctx.warnings.append(f"Could not read DHIME zip {zpath.name}: {exc}")
            continue

        code_col = find_column(part, ["codigo", "estacion"])
        date_col = find_column(part, ["fecha"])
        value_col = find_column(part, ["valor"])
        param_col = find_column(part, ["parametro"])
        if code_col is None or date_col is None or value_col is None:
            continue

        keep = [code_col, date_col, value_col] + ([param_col] if param_col else [])
        work = part[keep].copy()
        work = work.rename(
            columns={
                code_col: "station_code_raw",
                date_col: "time_raw",
                value_col: "level_raw",
            }
        )
        if param_col:
            work = work.rename(columns={param_col: "param_raw"})

        if "param_raw" in work.columns:
            pnorm = work["param_raw"].map(normalize_text)
            mask = pnorm.str.contains("nivel", na=False)
            if mask.any():
                work = work[mask]

        work["station_code"] = clean_station_code(work["station_code_raw"])
        work["time"] = coerce_datetime(work["time_raw"])
        work = work.dropna(subset=["time"])
        work["date"] = work["time"].dt.floor("D")
        work = clip_dates(work, "date", ctx.start_date, ctx.end_date)
        work["river_level"] = pd.to_numeric(work["level_raw"], errors="coerce")
        work = work.dropna(subset=["river_level"])
        work = work.merge(station_map, on="station_code", how="left")
        work = work.dropna(subset=["municipio_key"])
        pieces.append(work[["municipio_key", "date", "river_level", "station_code"]])

    if not pieces:
        ctx.warnings.append("DHIME files found but no valid rows were parsed.")
        return None

    data = pd.concat(pieces, ignore_index=True)
    out = (
        data.groupby(["municipio_key", "date"], as_index=False)
        .agg(
            river_level_mean_24h=("river_level", "mean"),
            river_level_max_24h=("river_level", "max"),
            river_level_station_count_24h=("station_code", "nunique"),
        )
    )
    out = out.sort_values(["municipio_key", "date"])
    out["river_level_delta_24h"] = (
        out.groupby("municipio_key")["river_level_mean_24h"].diff(1)
    )
    return out


def build_calendar_grid(
    muni_reference: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    muni_reference = muni_reference.dropna(subset=["municipio_key"]).drop_duplicates("municipio_key")
    grid = pd.MultiIndex.from_product(
        [muni_reference["municipio_key"].tolist(), dates],
        names=["municipio_key", "date"],
    ).to_frame(index=False)
    grid = grid.merge(
        muni_reference[["municipio_key", "municipio_name"]],
        on="municipio_key",
        how="left",
    )
    return grid


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    out["day"] = out["date"].dt.day.astype("int8")
    out["day_of_year"] = out["date"].dt.dayofyear.astype("int16")
    out["day_of_week"] = out["date"].dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    return out


def impute_numeric(df: pd.DataFrame, exclude_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [c for c in out.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    for c in numeric_cols:
        miss_col = f"{c}_missing"
        out[miss_col] = out[c].isna().astype("int8")
        out[c] = out.groupby("municipio_key")[c].transform(lambda s: s.fillna(s.median()))
        out[c] = out[c].fillna(out[c].median())
        out[c] = out[c].fillna(0.0)
    return out


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"municipio_key", "municipio_name", "date", "flood_next_24h"}
    return [c for c in df.columns if c not in excluded]


def build_phase1_dataset(ctx: BuildContext) -> Tuple[pd.DataFrame, Dict[str, object]]:
    table_files = list_candidate_files(ctx.raw_root, [".csv", ".txt", ".xlsx", ".xls", ".parquet"])
    if not table_files:
        raise FileNotFoundError(f"No tabular files found under {ctx.raw_root}")

    stations_rows, muni_static = load_station_catalog(ctx, table_files)
    target = build_target_from_ground_truth(ctx, table_files)
    meteo = load_open_meteo_features(ctx, table_files, stations_rows)
    precip = load_precip_features(ctx, table_files, stations_rows)
    dhime = load_dhime_features(ctx, stations_rows)

    muni_reference = pd.concat(
        [
            target[["municipio_key", "municipio_name"]],
            stations_rows[["municipio_key", "municipio_name"]],
        ],
        ignore_index=True,
    ).drop_duplicates("municipio_key")

    date_sources: List[pd.Series] = [target["date"], meteo["date"]]
    if precip is not None:
        date_sources.append(precip["date"])
    if dhime is not None:
        date_sources.append(dhime["date"])
    min_date = min(s.min() for s in date_sources if len(s) > 0)
    max_date = max(s.max() for s in date_sources if len(s) > 0)
    if ctx.start_date is not None:
        min_date = max(min_date, ctx.start_date)
    if ctx.end_date is not None:
        max_date = min(max_date, ctx.end_date)

    base = build_calendar_grid(muni_reference, min_date, max_date)
    base = base.merge(muni_static, on=["municipio_key", "municipio_name"], how="left")

    dataset = base.merge(meteo, on=["municipio_key", "date"], how="left")
    if precip is not None:
        dataset = dataset.merge(precip, on=["municipio_key", "date"], how="left")
    if dhime is not None:
        dataset = dataset.merge(dhime, on=["municipio_key", "date"], how="left")

    dataset = dataset.merge(
        target[["municipio_key", "date", "flood_next_24h"]],
        on=["municipio_key", "date"],
        how="left",
    )
    dataset["flood_next_24h"] = dataset["flood_next_24h"].fillna(0).astype("int8")
    dataset = add_calendar_features(dataset)
    dataset = impute_numeric(dataset, exclude_cols=["flood_next_24h"])
    dataset = dataset.sort_values(["municipio_key", "date"]).reset_index(drop=True)

    meta: Dict[str, object] = {
        "grain": "municipio-fecha (daily)",
        "target": "flood_next_24h (binary)",
        "rows": int(len(dataset)),
        "n_municipios": int(dataset["municipio_key"].nunique()),
        "date_min": str(dataset["date"].min().date()),
        "date_max": str(dataset["date"].max().date()),
        "feature_count": len(feature_columns(dataset)),
        "feature_columns": feature_columns(dataset),
        "used_sources": ctx.used_sources,
        "warnings": ctx.warnings,
    }
    return dataset, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 1 tabular dataset for MLP flood prediction.")
    parser.add_argument("--raw-root", default="raw", help="Root folder where raw data sources are located.")
    parser.add_argument("--outdir", default="processed/phase1", help="Output folder.")
    parser.add_argument("--start-date", default=None, help="Optional lower date bound (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Optional upper date bound (YYYY-MM-DD).")
    parser.add_argument("--chunksize", type=int, default=400000, help="Chunk size for large CSV ingestion.")
    parser.add_argument(
        "--include-dhime",
        action="store_true",
        help="Include DHIME river-level features from zip files (optional).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce log output.")
    return parser.parse_args()


def check_dependencies() -> Optional[str]:
    missing = []
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")
    if missing:
        return (
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install numpy pandas pyarrow openpyxl"
        )
    return None


def parse_date_arg(value: Optional[str], arg_name: str) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid {arg_name}: {value}. Expected YYYY-MM-DD.")
    return ts.floor("D")


def main() -> int:
    dep_err = check_dependencies()
    if dep_err:
        print(f"ERROR: {dep_err}", file=sys.stderr)
        return 1

    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        print(f"ERROR: raw root does not exist: {raw_root}", file=sys.stderr)
        return 1

    try:
        start_date = parse_date_arg(args.start_date, "--start-date")
        end_date = parse_date_arg(args.end_date, "--end-date")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if start_date is not None and end_date is not None and end_date < start_date:
        print("ERROR: --end-date must be >= --start-date", file=sys.stderr)
        return 1

    ctx = BuildContext(
        raw_root=raw_root,
        outdir=outdir,
        start_date=start_date,
        end_date=end_date,
        chunksize=max(50000, args.chunksize),
        include_dhime=args.include_dhime,
        verbose=not args.quiet,
        warnings=[],
        used_sources={},
    )

    try:
        dataset, meta = build_phase1_dataset(ctx)
    except Exception as exc:
        print(f"ERROR: dataset build failed: {exc}", file=sys.stderr)
        return 1

    csv_path = outdir / "phase1_dataset_municipio_fecha.csv"
    parquet_path = outdir / "phase1_dataset_municipio_fecha.parquet"
    json_path = outdir / "phase1_dataset_metadata.json"

    dataset.to_csv(csv_path, index=False)
    try:
        dataset.to_parquet(parquet_path, index=False)
        meta["parquet_saved"] = True
    except Exception as exc:
        meta["parquet_saved"] = False
        ctx.warnings.append(f"Parquet export skipped: {exc}")

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("\n=== Phase 1 dataset build complete ===")
    print(f"Rows: {meta['rows']}")
    print(f"Municipios: {meta['n_municipios']}")
    print(f"Date range: {meta['date_min']} -> {meta['date_max']}")
    print(f"Feature count: {meta['feature_count']}")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path} (saved={meta['parquet_saved']})")
    print(f"Metadata: {json_path}")
    if ctx.warnings:
        print("\nWarnings:")
        for w in ctx.warnings:
            print(f"- {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
