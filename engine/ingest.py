"""
ingest.py — Toutes les sources vers un seul DataFrame normalisé.

Schéma de sortie (une ligne = un point de mesure) :
    t          datetime UTC
    lat, lon   float, degrés (NaN si indoor / home-trainer)
    ele        float, mètres (altitude brute, non lissée)
    hr         float, bpm (NaN si absent)
    cad        float, spm course / rpm vélo (NaN si absent)
    power      float, watts (NaN si absent — vélo surtout)

Le lissage et le calcul de pente sont dans metrics.py, pas ici : on garde
l'ingestion aussi bête que possible pour pouvoir déboguer les sources
séparément du modèle.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

COLUMNS = ["t", "lat", "lon", "ele", "hr", "cad", "power"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}).assign(
        t=pd.Series(dtype="datetime64[ns, UTC]")
    )


def _finalize(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    return df


# ── GPX ───────────────────────────────────────────────────────────────────────

def from_gpx(source) -> pd.DataFrame:
    import gpxpy

    # Même précaution que pour le TCX : gpxpy s'appuie sur un analyseur XML
    # tout aussi strict sur les octets précédant la déclaration.
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            raw = f.read()
    else:
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    gpx = gpxpy.parse(io.BytesIO(_clean_xml(raw)))

    rows = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                hr = cad = power = np.nan
                for ext in (p.extensions or []):
                    for child in ext.iter():
                        tag = child.tag.split("}")[-1].lower()
                        try:
                            val = float(child.text)
                        except (TypeError, ValueError):
                            continue
                        if tag in ("hr", "heartrate"):
                            hr = val
                        elif tag in ("cad", "cadence", "runcadence"):
                            cad = val
                        elif tag in ("power", "watts"):
                            power = val
                rows.append({
                    "t": p.time, "lat": p.latitude, "lon": p.longitude,
                    "ele": p.elevation, "hr": hr, "cad": cad, "power": power,
                })
    return _finalize(rows)


# ── TCX ───────────────────────────────────────────────────────────────────────

def _clean_xml(raw: bytes) -> bytes:
    """
    Retire ce qui précède la déclaration XML.

    ElementTree applique la norme à la lettre : le moindre octet avant
    `<?xml` provoque « XML or text declaration not at start of entity ».
    Or beaucoup d'exportateurs écrivent un BOM UTF-8 (EF BB BF), parfois un
    saut de ligne ou une tabulation. Sur l'archive Strava, cela concernait
    57 fichiers sur 340 — soit un sixième de l'historique perdu pour trois
    octets invisibles.

    On accepte aussi les BOM UTF-16, qui imposent alors de laisser
    ElementTree lire l'encodage déclaré.
    """
    if raw[:3] == b"\xef\xbb\xbf":            # BOM UTF-8
        raw = raw[3:]
    elif raw[:2] in (b"\xff\xfe", b"\xfe\xff"):   # BOM UTF-16
        return raw

    # Espaces, sauts de ligne ou tabulations avant la déclaration.
    debut = raw.find(b"<")
    if debut > 0 and raw[:debut].strip() == b"":
        raw = raw[debut:]
    return raw


def from_tcx(source) -> pd.DataFrame:
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            raw = f.read()
    else:
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")

    root = ET.fromstring(_clean_xml(raw))

    def local(elem, name):
        """Cherche un descendant par nom local, en ignorant le namespace."""
        for child in elem.iter():
            if child.tag.split("}")[-1] == name:
                return child
        return None

    def num(elem, name):
        found = local(elem, name) if elem is not None else None
        if found is None or not found.text:
            return np.nan
        try:
            return float(found.text)
        except ValueError:
            return np.nan

    rows = []
    for tp in root.iter():
        if tp.tag.split("}")[-1] != "Trackpoint":
            continue
        t_el = local(tp, "Time")
        if t_el is None or not t_el.text:
            continue
        try:
            t = datetime.fromisoformat(t_el.text.strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)

        pos = local(tp, "Position")
        lat = num(pos, "LatitudeDegrees") if pos is not None else np.nan
        lon = num(pos, "LongitudeDegrees") if pos is not None else np.nan

        hr_el = local(tp, "HeartRateBpm")
        hr = num(hr_el, "Value") if hr_el is not None else np.nan

        cad = num(tp, "Cadence")
        if np.isnan(cad):
            cad = num(tp, "RunCadence")
        power = num(tp, "Watts")

        rows.append({
            "t": t, "lat": lat, "lon": lon, "ele": num(tp, "AltitudeMeters"),
            "hr": hr, "cad": cad, "power": power,
        })
    return _finalize(rows)


# ── FIT ───────────────────────────────────────────────────────────────────────

def from_fit(source) -> pd.DataFrame:
    from fitparse import FitFile

    data = source if isinstance(source, (str, os.PathLike)) else io.BytesIO(source.read())
    fit = FitFile(data)

    rows = []
    for rec in fit.get_messages("record"):
        v = {f.name: f.value for f in rec}
        lat, lon = v.get("position_lat"), v.get("position_long")
        semi = 180.0 / 2**31
        rows.append({
            "t": v.get("timestamp"),
            "lat": lat * semi if lat is not None else np.nan,
            "lon": lon * semi if lon is not None else np.nan,
            "ele": v.get("enhanced_altitude", v.get("altitude", np.nan)),
            "hr": v.get("heart_rate", np.nan),
            "cad": v.get("cadence", np.nan),
            "power": v.get("power", np.nan),
        })
    return _finalize(rows)


# ── Strava streams ────────────────────────────────────────────────────────────

def from_strava_streams(streams: dict, start_time: datetime) -> pd.DataFrame:
    """
    Convertit la réponse de /activities/{id}/streams en DataFrame.

    Source à privilégier : Strava renvoie des flux déjà alignés et
    dé-bruités côté altitude, ce qui évite le parsing XML et ses pièges.
    Demander keys=time,latlng,altitude,heartrate,cadence,watts,velocity_smooth
    """
    def col(key, default=np.nan):
        s = streams.get(key)
        return np.asarray(s["data"], dtype=float) if s else None

    time_s = streams.get("time", {}).get("data")
    if not time_s:
        return _empty()
    n = len(time_s)

    latlng = streams.get("latlng", {}).get("data")
    if latlng:
        arr = np.asarray(latlng, dtype=float)
        lat, lon = arr[:, 0], arr[:, 1]
    else:
        lat = lon = np.full(n, np.nan)

    def or_nan(key):
        a = col(key)
        return a if a is not None and len(a) == n else np.full(n, np.nan)

    start = pd.Timestamp(start_time).tz_convert("UTC") if pd.Timestamp(start_time).tzinfo \
        else pd.Timestamp(start_time).tz_localize("UTC")

    return _finalize([
        {
            "t": start + pd.Timedelta(seconds=float(time_s[i])),
            "lat": lat[i], "lon": lon[i],
            "ele": or_nan("altitude")[i],
            "hr": or_nan("heartrate")[i],
            "cad": or_nan("cadence")[i],
            "power": or_nan("watts")[i],
        }
        for i in range(n)
    ])


# ── Dispatch ──────────────────────────────────────────────────────────────────

def load(source, filename: str = "") -> pd.DataFrame:
    """Détecte le format depuis l'extension et parse."""
    name = filename or (source if isinstance(source, str) else getattr(source, "name", ""))
    ext = os.path.splitext(str(name))[-1].lower()
    if ext == ".tcx":
        return from_tcx(source)
    if ext == ".fit":
        return from_fit(source)
    return from_gpx(source)
