"""
physio.py — Base physiologique du moteur.

Corrige l'erreur centrale de la V2 précédente : la formule
    vap = vitesse / (1 + 3.5 * pente)
n'est valable ni en montée ni surtout en descente. En descente (pente < 0)
elle divise par un nombre < 1 et gonfle donc la vitesse équivalente, alors
que la descente coûte MOINS cher que le plat jusqu'à environ -20 %.
Résultat : tous les KPI de descente de la V2 étaient inversés.

On utilise ici les polynômes de Minetti et al. (2002), "Energy cost of
walking and running at extreme uphill and downhill slopes", J Appl Physiol
93:1039-1046. Domaine de validité : pente de -45 % à +45 %.
"""

from __future__ import annotations

import numpy as np

# Coût énergétique du plat, J/kg/m
CR_FLAT = 3.6
CW_FLAT = 2.5

# Plafond biomécanique de la marche (m/s) — au-delà, on court forcément.
# ~7,5 km/h sur le plat, décroissant avec la pente.
WALK_SPEED_CEILING_FLAT = 2.1


def cost_running(slope: np.ndarray | float) -> np.ndarray:
    """Coût énergétique de la course, J/kg/m, en fonction de la pente (fraction)."""
    i = np.clip(np.asarray(slope, dtype=float), -0.45, 0.45)
    return (
        155.4 * i**5
        - 30.4 * i**4
        - 43.3 * i**3
        + 46.3 * i**2
        + 19.5 * i
        + 3.6
    )


def cost_walking(slope: np.ndarray | float) -> np.ndarray:
    """Coût énergétique de la marche, J/kg/m, en fonction de la pente (fraction)."""
    i = np.clip(np.asarray(slope, dtype=float), -0.45, 0.45)
    return (
        280.5 * i**5
        - 58.7 * i**4
        - 76.8 * i**3
        + 51.9 * i**2
        + 19.6 * i
        + 2.5
    )


def gap(speed: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """
    Grade Adjusted Pace : vitesse équivalente sur le plat, en m/s.

    À puissance métabolique constante P = C(i) * v, la vitesse plat
    équivalente vaut v * C(i) / C(0). C'est la seule normalisation qui
    permette de comparer un segment à 15 % et un segment à -8 %.
    """
    return np.asarray(speed, dtype=float) * cost_running(slope) / CR_FLAT


def metabolic_power(speed: np.ndarray, slope: np.ndarray,
                    is_walking: np.ndarray | None = None) -> np.ndarray:
    """
    Puissance métabolique en W/kg. Utilise le coût marche si le segment est
    identifié comme marché — sinon le coût course.
    """
    speed = np.asarray(speed, dtype=float)
    cr = cost_running(slope)
    if is_walking is None:
        return speed * cr
    cw = cost_walking(slope)
    return speed * np.where(is_walking, cw, cr)


def walk_speed_ceiling(slope: np.ndarray) -> np.ndarray:
    """
    Vitesse maximale plausible en marche (m/s), décroissante avec la pente.
    Sert de garde-fou : au-delà, un point est nécessairement couru même si
    la cadence est basse (cadence manquante, capteur poignet, etc.).
    """
    i = np.asarray(slope, dtype=float)
    return WALK_SPEED_CEILING_FLAT * np.clip(1 - 1.8 * np.maximum(i, 0), 0.35, 1.0)


def classify_gait(cadence: np.ndarray, speed: np.ndarray,
                  slope: np.ndarray, run_cadence_min: float = 145.0
                  ) -> np.ndarray:
    """
    Sépare marche et course, point par point.

    Règle primaire : la cadence de course est bimodale et se sépare nettement
    de la marche autour de 145 spm (en spm double-jambe). Règle secondaire
    quand la cadence est absente : plafond de vitesse de marche.

    Retourne un masque booléen True = marche.
    """
    cadence = np.asarray(cadence, dtype=float)
    speed = np.asarray(speed, dtype=float)
    ceiling = walk_speed_ceiling(slope)

    has_cad = ~np.isnan(cadence)
    walking = np.zeros_like(speed, dtype=bool)

    # Avec cadence : marche = cadence basse ET vitesse compatible
    walking[has_cad] = (cadence[has_cad] < run_cadence_min) & (
        speed[has_cad] <= ceiling[has_cad] * 1.15
    )
    # Sans cadence : fallback vitesse seule
    walking[~has_cad] = speed[~has_cad] <= ceiling[~has_cad]
    return walking


def hr_reserve(hr: np.ndarray, hr_rest: float, hr_max: float) -> np.ndarray:
    """
    Fraction de réserve cardiaque (Karvonen). Nettement supérieur au %FCmax
    codé en dur dans la version précédente : la réserve est comparable entre
    individus et stable dans le temps pour un même athlète.
    """
    hr = np.asarray(hr, dtype=float)
    denom = max(hr_max - hr_rest, 1.0)
    return (hr - hr_rest) / denom


def theoretical_walk_crossover() -> float:
    """
    Pente (fraction) à partir de laquelle marcher devient plus économique
    que courir, à vitesse atteignable. Repère théorique uniquement : le
    seuil réel se mesure sur les données de l'athlète (cf. metrics.py).
    """
    slopes = np.linspace(0.0, 0.45, 451)
    # Vitesse atteignable = min(vitesse course libre, plafond marche)
    v_walk = walk_speed_ceiling(slopes)
    # À puissance égale, la course produit v_run = P / Cr
    p_ref = 12.0  # W/kg, allure trail soutenue
    v_run = p_ref / cost_running(slopes)
    idx = np.argmax(v_walk >= v_run)
    return float(slopes[idx]) if v_walk[-1] >= v_run[-1] else float("nan")
