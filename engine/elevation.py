"""
elevation.py — Reconstitution d'altitude depuis un modèle de terrain public.

POURQUOI

Ta montre Polar n'a pas d'altimètre : les champs d'altitude des TCX et FIT
existent mais valent None. Sur 964 activités d'archive, cela concerne
environ 430 sorties trail — le cœur de l'historique. Les rejeter serait
perdre l'essentiel ; les traiter comme du plat serait pire encore, car
elles entreraient à 0 D+ et seraient classées « route », faussant l'effet
de terrain que le modèle mesure justement sur ce critère.

On interroge donc Open Topo Data (api.opentopodata.org), API publique sans
authentification, en privilégiant EU-DEM à 25 m et en repli SRTM à 30 m.
C'est exactement ce que fait Strava à l'import — d'où l'altitude présente
dans les GPX Strava et absente des fichiers originaux.

LES TROIS CONTRAINTES QUI DICTENT LA CONCEPTION

1. QUOTA. 100 points par requête, environ une requête par seconde, et de
   l'ordre de 1 000 requêtes par jour en usage courtois. Soit à peu près
   100 000 points par jour. Sans précaution, 430 sorties à 4 000 points
   représenteraient 1,7 million de points : dix-sept jours.

2. RÉSOLUTION. Le modèle a un pas de 25 à 30 m. Interroger un point tous
   les 3 m n'apporte rien : on lit trente fois la même cellule. On
   échantillonne donc tous les 30 m de distance parcourue, puis on
   interpole. Cela divise le volume par dix sans perte d'information.

3. RÉPÉTITION. Tu cours sur un nombre restreint de parcours. En arrondissant
   les coordonnées à une grille de 20 m — plus fine que le modèle, donc sans
   perte —, un point déjà vu n'est jamais redemandé. Après les premières
   sorties, le taux de réutilisation devient élevé et le coût s'effondre.

Le cache est un CSV sur disque : les altitudes sont figées une fois pour
toutes. Relancer un import ne redéclenche aucune requête.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://api.opentopodata.org/v1"
DATASET = "eudem25m,srtm30m"     # Europe d'abord, monde en repli
BATCH = 100                      # maximum accepté par requête
PAUSE = 1.15                     # secondes entre deux requêtes
GRID = 2e-4                      # ~20 m, sous la résolution du modèle
SAMPLE_M = 30.0                  # un point tous les 30 m parcourus

CACHE_PATH = Path("data") / "dem_cache.csv"


class Quota(Exception):
    """Quota journalier atteint. L'import doit s'arrêter proprement."""


class Reseau(Exception):
    """
    Le service d'altitude est injoignable.

    Exception DISTINCTE d'un échec d'activité, et c'est le point important.
    La version précédente laissait l'erreur réseau remonter dans le
    `except Exception` générique de l'import : chaque activité était alors
    marquée « échec » et l'import continuait, brûlant ainsi 138 sorties
    d'affilée pour une seule cause — le service indisponible. Il faut
    s'arrêter au premier constat d'indisponibilité durable, comme pour le
    quota, et reprendre plus tard.
    """


_SESSION = None


def _session():
    """
    Session HTTP réutilisée, avec reprises automatiques.

    Sans session, chaque requête rouvre une connexion TLS : coûteux, et
    surtout sans aucune reprise sur incident. Les échecs observés étaient
    des HTTPSConnectionPool, c'est-à-dire des ruptures au niveau transport,
    pas des refus du service.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    import requests
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    try:
        from urllib3.util.retry import Retry
        retry = Retry(total=4, backoff_factor=1.5,
                      status_forcelist=(500, 502, 503, 504),
                      allowed_methods=frozenset(["GET", "POST"]))
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
    except Exception:
        pass
    s.headers.update({"User-Agent": "trail-lab/1.0"})
    _SESSION = s
    return s


class ElevationCache:
    """Cache disque des altitudes, indexé sur une grille de coordonnées."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.points: dict[tuple[int, int], float] = {}
        self.new = 0
        if path.exists() and path.stat().st_size > 10:
            df = pd.read_csv(path)
            if {"gx", "gy", "ele"}.issubset(df.columns):
                self.points = {
                    (int(r.gx), int(r.gy)): float(r.ele)
                    for r in df.itertuples() if not np.isnan(r.ele)
                }

    @staticmethod
    def key(lat: float, lon: float) -> tuple[int, int]:
        return int(round(lat / GRID)), int(round(lon / GRID))

    def get(self, lat, lon):
        return self.points.get(self.key(lat, lon))

    def put(self, lat, lon, ele) -> None:
        self.points[self.key(lat, lon)] = float(ele)
        self.new += 1

    def save(self) -> None:
        if not self.points:
            return                      # ne pas écrire un fichier vide
        self.path.parent.mkdir(exist_ok=True)
        pd.DataFrame(
            [{"gx": k[0], "gy": k[1], "ele": v} for k, v in self.points.items()]
        ).to_csv(self.path, index=False)
        self.new = 0

    def __len__(self) -> int:
        return len(self.points)


def _query(coords: list[tuple[float, float]], session=None) -> list[float | None]:
    """Une requête, au plus 100 points. Repli exponentiel sur 429."""
    import requests

    s = session or requests
    locs = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in coords)

    for attempt in range(4):
        r = s.post(f"{API}/{DATASET}", data={"locations": locs}, timeout=60)
        if r.status_code == 429:
            if attempt == 3:
                raise Quota(
                    "Quota Open Topo Data atteint. Les altitudes déjà "
                    "obtenues sont enregistrées : relance demain, rien ne "
                    "sera redemandé."
                )
            time.sleep(5 * 2 ** attempt)
            continue
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            raise RuntimeError(f"Réponse inattendue : {data.get('error', data)}")
        return [res.get("elevation") for res in data["results"]]
    raise Quota("Quota atteint.")


def _sample_indices(cum_dist: np.ndarray, step_m: float) -> np.ndarray:
    """Indices des points espacés d'au moins step_m, premier et dernier inclus."""
    if len(cum_dist) < 2:
        return np.arange(len(cum_dist))
    targets = np.arange(0, cum_dist[-1] + step_m, step_m)
    idx = np.unique(np.searchsorted(cum_dist, targets, side="left"))
    idx = idx[idx < len(cum_dist)]
    return np.unique(np.concatenate([[0], idx, [len(cum_dist) - 1]]))


def enrich(raw: pd.DataFrame, cache: ElevationCache | None = None,
           step_m: float = SAMPLE_M, progress=None) -> tuple[pd.DataFrame, dict]:
    """
    Renvoie (DataFrame avec la colonne `ele` remplie, statistiques).

    Ne modifie rien si la trace porte déjà une altitude exploitable.
    Lève Quota si l'API refuse — l'appelant doit alors s'arrêter, pas
    réessayer en boucle.
    """
    from engine.metrics import haversine

    stats = {"queried": 0, "cached": 0, "sampled": 0, "skipped": False,
             "repli_cache": False}

    ele = raw["ele"].dropna()
    if len(ele) > 30 and float(ele.max() - ele.min()) >= 5.0:
        stats["skipped"] = True
        return raw, stats

    lat = raw["lat"].to_numpy(dtype=float)
    lon = raw["lon"].to_numpy(dtype=float)
    valid = ~np.isnan(lat) & ~np.isnan(lon)
    if valid.sum() < 30:
        raise ValueError("Pas de coordonnées GPS : altitude non reconstituable.")

    d = np.zeros(len(lat))
    d[1:] = np.nan_to_num(haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]))
    cum = np.cumsum(np.where(valid, d, 0.0))

    idx = _sample_indices(cum, step_m)
    idx = idx[valid[idx]]
    stats["sampled"] = len(idx)

    # `if cache is None`, surtout pas `cache or ...` : la classe définit
    # __len__, donc un cache vide est falsy et serait silencieusement
    # remplacé par un neuf à chaque appel. Toutes les altitudes déjà payées
    # seraient perdues et rien ne serait jamais réutilisé.
    if cache is None:
        cache = ElevationCache()
    todo, known = [], {}
    for i in idx:
        hit = cache.get(lat[i], lon[i])
        if hit is None:
            todo.append(i)
        else:
            known[i] = hit
    stats["cached"] = len(known)

    # REPLI SUR LE CACHE SEUL. Si le service tombe alors qu'une bonne part
    # des points est déjà connue, mieux vaut une altitude interpolée sur ces
    # points que de perdre la sortie. Le profil est un peu plus grossier,
    # le D+ reste correct — bien plus utile qu'un échec.
    couverture = len(known) / max(len(idx), 1)
    try:
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            values = _query([(lat[i], lon[i]) for i in chunk])
            for i, v in zip(chunk, values):
                if v is not None:
                    cache.put(lat[i], lon[i], v)
                    known[i] = float(v)
            stats["queried"] += len(chunk)
            if progress:
                progress(start + len(chunk), len(todo))
            if start + BATCH < len(todo):
                time.sleep(PAUSE)
    except (Reseau, Quota):
        if couverture >= 0.6 and len(known) >= 20:
            stats["repli_cache"] = True
        else:
            raise

    if len(known) < 10:
        raise ValueError("Le modèle de terrain n'a renvoyé aucune altitude "
                         "exploitable pour cette trace.")

    # Interpolation linéaire sur la distance parcourue, et non sur l'indice :
    # les points ne sont pas équidistants quand l'allure varie.
    ks = np.array(sorted(known))
    out = raw.copy()
    out["ele"] = np.interp(cum, cum[ks], np.array([known[k] for k in ks]))
    return out, stats


def coverage_estimate(n_activities: int, mean_km: float = 10.0,
                      reuse: float = 0.0) -> dict:
    """
    Estime le coût d'un enrichissement, pour décider avant de lancer.

    reuse : part attendue de points déjà en cache. Nulle au départ, elle
    monte vite si tu répètes les mêmes parcours.
    """
    per_activity = mean_km * 1000 / SAMPLE_M
    points = n_activities * per_activity * (1 - reuse)
    calls = np.ceil(points / BATCH)
    return {
        "points": int(points),
        "calls": int(calls),
        "minutes": round(calls * PAUSE / 60, 1),
        "jours_quota": round(calls / 1000, 2),
    }


def ping() -> dict:
    """
    Teste la joignabilité du service, indépendamment de tout import.

    Utile pour distinguer une panne du service d'un blocage local : sur un
    poste d'entreprise, un proxy peut laisser passer le trafic navigateur
    et bloquer les appels Python.
    """
    import requests
    try:
        r = _session().post(f"{API}/{DATASET}",
                            data={"locations": "44.10,6.90"}, timeout=20)
        return {"joignable": r.status_code == 200, "code": r.status_code,
                "reponse": r.text[:180]}
    except Exception as e:
        return {"joignable": False, "erreur": f"{type(e).__name__}: {e}",
                "piste": "Service indisponible, coupure réseau, ou proxy "
                         "bloquant api.opentopodata.org depuis Python."}
