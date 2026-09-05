"""
bike.py — Analyse vélo par la puissance.

Pourquoi le vélo ne passe PAS par le moteur trail : la VAP n'y a aucun
sens. On descend sans produire d'effort, l'aspiration modifie le coût de
30 %, et le rendement dépend du braquet. Toute la métrologie cycliste
sérieuse repose sur la puissance mécanique mesurée, pas sur la vitesse.

Deux garde-fous que la plupart des outils négligent.

1. PUISSANCE ESTIMÉE. Strava renvoie un flux `watts` même sans capteur :
   c'est alors une puissance reconstituée depuis la vitesse, la pente et
   un poids supposé. Sur route plate ou en peloton, l'erreur dépasse
   couramment 25 %. Le champ `device_watts` distingue les deux. Ce module
   REFUSE de calculer un TSS sur de la puissance estimée : mieux vaut pas
   de chiffre qu'un chiffre faux qu'on suivra pendant six mois.

2. CAPTEUR MONO-JAMBE. Un 4iiii Precision monté sur manivelle gauche
   mesure une jambe et double. L'asymétrie gauche/droite courante est de
   2 à 5 %, avec des cas à 10 %. Conséquence pratique : le biais est
   à peu près CONSTANT, donc le suivi longitudinal reste valable, mais
   la comparaison de ton FTP à des valeurs externes ne l'est pas. Le
   module affiche l'avertissement plutôt que de le taire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

# Facteurs de pondération TRIMP de Banister, coefficient masculin
TRIMP_B = 1.92


def normalized_power(power: np.ndarray, dt: np.ndarray,
                     window_s: float = 30.0) -> float:
    """
    Puissance normalisée (Coggan) : moyenne glissante 30 s, élevée à la
    puissance 4, moyennée, puis racine quatrième. L'exposant 4 traduit le
    fait que le coût physiologique croît beaucoup plus vite que la
    puissance mécanique — un effort en dents de scie coûte plus cher
    qu'un effort lisse de même moyenne.
    """
    p = pd.Series(np.asarray(power, dtype=float)).interpolate(limit_direction="both")
    if p.isna().all():
        return float("nan")
    step = float(np.median(dt[dt > 0])) if (dt > 0).any() else 1.0
    win = max(int(round(window_s / max(step, 0.1))), 1)
    rolled = p.rolling(win, min_periods=max(win // 2, 1)).mean().dropna()
    if rolled.empty:
        return float("nan")
    return float((rolled ** 4).mean() ** 0.25)


def summarize_ride(d: pd.DataFrame, ftp: float | None,
                   device_watts: bool = False,
                   hr_rest: float = 50, hr_max: float = 190) -> dict:
    """
    d : DataFrame déjà passé par metrics.prepare() (avec dt, dist, d_plus).
    ftp : Functional Threshold Power en watts, None si non testé.
    """
    dt = d["dt"].to_numpy()
    total_t = float(dt.sum())
    power = d["power"].to_numpy(dtype=float)
    has_power = device_watts and not np.isnan(power).all()

    out = {
        "distance_km": float(d["dist"].sum() / 1000),
        "d_plus": float(d["d_plus"].sum()),
        "d_minus": float(d["d_minus"].sum()),
        "duration_h": total_t / 3600,
        "speed_kmh": float(d["dist"].sum() / total_t * 3.6) if total_t else np.nan,
        "has_power": has_power,
        "totaux_source": "calcul",
        "power_source": "capteur" if has_power else
                        ("estimée — non exploitée" if not np.isnan(power).all() else "absente"),
    }

    if has_power:
        moving = dt > 0
        out["power_mean"] = float(np.nansum(power[moving] * dt[moving]) / dt[moving].sum())
        out["np"] = normalized_power(power, dt)
        out["variability_index"] = (
            out["np"] / out["power_mean"] if out["power_mean"] > 0 else np.nan
        )
        out["work_kj"] = float(np.nansum(power * dt) / 1000)
        if ftp and ftp > 0:
            intensity = out["np"] / ftp
            out["intensity_factor"] = intensity
            out["tss"] = float(total_t * out["np"] * intensity / (ftp * 3600) * 100)
        else:
            out["intensity_factor"] = np.nan
            out["tss"] = np.nan
    else:
        out.update(power_mean=np.nan, np=np.nan, variability_index=np.nan,
                   work_kj=np.nan, intensity_factor=np.nan, tss=np.nan)

    out["trimp"] = trimp(d["hr"].to_numpy(), dt, hr_rest, hr_max)
    # TOTAUX DE L'APPAREIL. Le chemin vélo ne passe pas par
    # metrics.summarize : il faut donc appliquer la substitution ici aussi,
    # sans quoi une sortie FIT affichait 2 289 m de dénivelé recalculé là
    # où le compteur en annonçait 1 370. Le défaut ne se voyait qu'à vélo,
    # puisque la course à pied emprunte l'autre chemin.
    out = metrics.appliquer_totaux(out, d)
    if out.get("duration_h"):
        out["speed_kmh"] = out["distance_km"] / out["duration_h"]
    return out

def trimp(hr: np.ndarray, dt: np.ndarray, hr_rest: float, hr_max: float) -> float:
    """
    TRIMP de Banister, pondéré exponentiellement par la réserve cardiaque.

    Sert de charge d'entraînement commune à TOUS les sports, y compris
    quand la puissance manque. C'est ce qui permet d'additionner une
    sortie trail et une sortie vélo dans un même bilan hebdomadaire —
    ce que ni la distance ni le D+ ne permettent.
    """
    hr = np.asarray(hr, dtype=float)
    m = ~np.isnan(hr) & (dt > 0)
    if m.sum() < 10:
        return float("nan")
    hrr = np.clip((hr[m] - hr_rest) / max(hr_max - hr_rest, 1.0), 0, 1.2)
    minutes = dt[m] / 60
    return float(np.sum(minutes * hrr * 0.64 * np.exp(TRIMP_B * hrr)))


def estimate_ftp(rides: pd.DataFrame, days: int = 90) -> dict:
    """
    Estimation de FTP depuis l'historique, à défaut d'un test formel.

    Convention retenue : 95 % de la meilleure puissance normalisée tenue
    au moins 20 minutes sur la fenêtre. C'est une approximation basse et
    assumée : sans effort maximal dans l'historique, elle sous-estime.
    Un test de 20 minutes reste très supérieur.
    """
    if rides.empty or "np" not in rides.columns:
        return {"ok": False, "reason": "Aucune sortie avec puissance capteur."}
    cutoff = rides["date"].max() - pd.Timedelta(days=days)
    d = rides[(rides["date"] >= cutoff) & (rides["duration_h"] >= 0.34)
              & rides["np"].notna()]
    if len(d) < 3:
        return {"ok": False, "reason":
                f"{len(d)} sortie(s) avec puissance sur {days} jours. "
                "Fais un test de 20 minutes, ce sera plus fiable."}
    best = float(d["np"].max())
    return {
        "ok": True,
        "ftp": round(best * 0.95),
        "n": int(len(d)),
        "note": "Estimation basse depuis l'historique. Un test de 20 min "
                "donnerait une valeur plus juste.",
    }


SINGLE_SIDED_WARNING = (
    "Capteur mono-jambe : la puissance est mesurée sur une manivelle puis "
    "doublée. L'asymétrie gauche/droite (2 à 5 % couramment) introduit un "
    "biais à peu près constant. Le suivi de TA progression reste valide ; "
    "la comparaison de ton FTP à des valeurs externes ne l'est pas."
)


# ── Meilleurs efforts de puissance ──────────────────────────────────────────

DUREES_W = (15, 30, 60)      # minutes
LISSAGE_MAX_S = 5            # la « puissance max » brute est du bruit


def best_power(d: pd.DataFrame, poids_kg: float | None = None,
               durees=DUREES_W) -> dict:
    """
    Meilleure puissance moyenne soutenue sur chaque durée, plus la pointe.

    Deux précautions.

    1. LA POINTE N'EST PAS LA VALEUR INSTANTANÉE. Un capteur renvoie des
       à-coups de 800 W sur un seul échantillon, sans signification
       physiologique. On lisse sur 5 secondes, convention usuelle.
    2. RÉÉCHANTILLONNAGE À LA SECONDE. Les fenêtres glissantes sur index de
       points seraient biaisées dès que la cadence d'enregistrement varie —
       cent points couvrent tantôt une minute, tantôt cinq.
    """
    out = {f"w{m}": np.nan for m in durees}
    out.update(w_max_5s=np.nan, w_moyen=np.nan)
    if poids_kg:
        out.update({f"wkg{m}": np.nan for m in durees})
        out.update(wkg_max_5s=np.nan, wkg_moyen=np.nan)

    if "power" not in d.columns or "dt" not in d.columns:
        return out
    p = d["power"].to_numpy(dtype=float)
    dt = d["dt"].to_numpy(dtype=float)
    ok = (dt > 0) & ~np.isnan(p)
    if ok.sum() < 60:
        return out

    t = np.cumsum(dt[ok])
    grille = np.arange(0, float(t[-1]), 1.0)
    if len(grille) < 60:
        return out
    w = np.interp(grille, t, p[ok])
    cum = np.concatenate([[0.0], np.cumsum(w)])

    out["w_moyen"] = float(np.mean(w))
    if len(w) > LISSAGE_MAX_S:
        liss = (cum[LISSAGE_MAX_S:] - cum[:-LISSAGE_MAX_S]) / LISSAGE_MAX_S
        out["w_max_5s"] = float(np.max(liss))

    for m in durees:
        n = int(m * 60)
        if len(w) < n:
            continue
        out[f"w{m}"] = float(np.max((cum[n:] - cum[:-n]) / n))

    if poids_kg and poids_kg > 20:
        for k in list(out):
            if k.startswith("w") and not k.startswith("wkg") and not np.isnan(out[k]):
                out["wkg" + k[1:]] = out[k] / poids_kg
    return out


def summarize_power(d: pd.DataFrame, poids_kg: float | None,
                    device_watts: bool) -> dict:
    """
    N'expose les puissances QUE si elles viennent d'un capteur.

    Strava renvoie un flux `watts` même sans capteur : puissance
    reconstituée depuis la vitesse, la pente et un poids supposé, avec plus
    de 25 % d'erreur en peloton. Stocker ces valeurs à côté des vraies
    créerait un historique dont on ne pourrait plus rien tirer.
    """
    if not device_watts:
        return {}
    return best_power(d, poids_kg)
