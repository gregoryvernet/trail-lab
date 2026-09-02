"""
efforts.py — Meilleurs efforts soutenus, et vitesse de base.

VERSION 2026-08-25-A

POURQUOI CE MODULE EXISTE

Le modèle d'endurance ajuste T = a · KE^b sur des sorties ENTIÈRES jugées
continues. Sur 242 sorties, cela n'en retient que 25 : il faut que la
séance soit d'un seul tenant, assez longue, assez intense, sur le bon
terrain. Vingt-cinq points pour deux paramètres, dont un seul au-delà de
80 km-effort.

L'approche par meilleurs efforts renverse le problème. Au lieu de chercher
des sorties entières exploitables, on extrait de CHAQUE sortie le meilleur
bloc soutenu de 30, 60 et 120 minutes. Une séance de fractionné devient
alors une donnée valable via son meilleur bloc de 30 minutes. Une sortie
longue mal gérée fournit quand même son meilleur bloc de 2 heures.

Deux gains :

  - l'effectif passe de 25 à toutes les sorties assez longues ;
  - on lit directement l'ENVELOPPE de performance, au lieu d'essayer de la
    filtrer par des heuristiques d'intensité et de type de séance.

C'est le principe de la courbe puissance-durée du cyclisme, transposé au
GAP puisqu'on n'a pas de puissance à pied.

CE QUE ÇA NE RÉSOUT PAS

Un meilleur bloc de 120 minutes ne dit rien sur ce qui se passe à la
huitième heure. La courbe donne un niveau de référence, pas la
décroissance sur un ultra — celle-ci reste portée par l'exposant
d'endurance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "2026-08-25-A"

# Durées retenues, en minutes. 30 pour le seuil, 60 pour l'allure
# soutenue, 120 pour la cadence de fond.
DUREES = (30, 60, 120)


def best_efforts(d: pd.DataFrame, durees=DUREES) -> dict:
    """
    Meilleur GAP moyen soutenu sur chaque durée, en km/h.

    On travaille sur le GAP et non la vitesse brute : sans cela, le
    meilleur bloc de 30 minutes serait systématiquement la descente la plus
    longue de la sortie, ce qui ne mesure aucune capacité.

    Implémentation par somme cumulée sur une grille temporelle régulière à
    la seconde. Les fenêtres glissantes sur index de points seraient
    biaisées : à cadence d'enregistrement variable, cent points couvrent
    tantôt une minute tantôt cinq.
    """
    out = {f"v{m}": np.nan for m in durees}
    if "gap" not in d.columns or "dt" not in d.columns:
        return out

    dt = d["dt"].to_numpy(dtype=float)
    gap = d["gap"].to_numpy(dtype=float)
    ok = (dt > 0) & ~np.isnan(gap)
    if ok.sum() < 60:
        return out

    t = np.cumsum(dt[ok])
    total = float(t[-1])
    # Rééchantillonnage à la seconde : l'intégration devient exacte.
    grille = np.arange(0, total, 1.0)
    if len(grille) < 60:
        return out
    g = np.interp(grille, t, gap[ok])
    cum = np.concatenate([[0.0], np.cumsum(g)])

    for m in durees:
        w = int(m * 60)
        if len(g) < w:
            continue
        # Moyenne de chaque fenêtre de w secondes, en une opération.
        moyennes = (cum[w:] - cum[:-w]) / w
        out[f"v{m}"] = float(np.nanmax(moyennes) * 3.6)
    return out


def build_curve(hist: pd.DataFrame, months: int = 12,
                quantile: float = 0.95) -> dict:
    """
    Courbe vitesse-durée personnelle : le quantile haut des meilleurs
    efforts sur la fenêtre retenue.

    Pourquoi un quantile à 95 % et non le maximum. Le maximum absolu est
    porté par une seule sortie et hérite de tous ses accidents — une
    descente mal filtrée, un décrochage GPS, un jour exceptionnel. Le
    quantile à 95 % conserve le niveau de l'enveloppe en écartant le point
    unique aberrant.
    """
    if hist.empty:
        return {"ok": False, "reason": "Historique vide."}

    d = hist.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    if months:
        d = d[d["date"] >= d["date"].max() - pd.DateOffset(months=months)]
    d = d[d["sport"].isin(["trail", "rando"])]

    courbe, effectifs = {}, {}
    for m in DUREES:
        col = f"v{m}"
        if col not in d.columns:
            continue
        v = d[col].dropna()
        v = v[v > 1]
        if len(v) < 4:
            continue
        courbe[m] = float(np.quantile(v, quantile))
        effectifs[m] = int(len(v))

    if not courbe:
        return {"ok": False, "reason":
                "Aucun meilleur effort en base. Réimporte l'historique : "
                "ces colonnes sont calculées à l'import."}

    # Contrôle de cohérence : la courbe doit décroître avec la durée.
    durs = sorted(courbe)
    coherente = all(courbe[durs[i]] >= courbe[durs[i + 1]] - 0.05
                    for i in range(len(durs) - 1))

    return {"ok": True, "courbe": courbe, "n": effectifs,
             "coherente": coherente, "months": months,
             "decroissance": ({f"{durs[i]}->{durs[i+1]}":
                               round((courbe[durs[i + 1]] / courbe[durs[i]] - 1) * 100, 1)
                               for i in range(len(durs) - 1)} if len(durs) > 1 else {})}


def base_speed(curve: dict, duree_cible_h: float) -> dict:
    """
    Vitesse de référence pour une durée donnée, par ajustement log-log de la
    courbe.

    LA PREMIÈRE VERSION ÉTAIT FAUSSE. Elle pondérait les trois blocs par
    leur proximité en log avec la cible. À neuf heures, les trois blocs
    étant tous très éloignés, les poids devenaient presque égaux et la
    vitesse prédite ressortait PLUS ÉLEVÉE qu'à trois heures. Absurde.

    On ajuste donc la décroissance :

        log v = alpha + beta · log(durée)

    beta vaut typiquement −0,10 à −0,14, soit 7 à 9 % de perte par
    doublement de durée. La cible est évaluée sur cette droite, ce qui
    permet une extrapolation cohérente au-delà du bloc le plus long.

    VÉRIFICATION CROISÉE GRATUITE

    beta et l'exposant d'endurance b mesurent le même phénomène par deux
    chemins indépendants. Si v ∝ T^beta, alors la distance D = v·T suit
    D ∝ T^(1+beta), donc T ∝ D^(1/(1+beta)). D'où :

        b = 1 / (1 + beta)

    beta = −0,120 donne b = 1,136. Les deux estimations doivent
    concorder : l'une vient des meilleurs blocs intra-sortie, l'autre de
    l'ajustement sur des sorties entières. Un écart important signale un
    problème dans l'une des deux, pas une découverte.
    """
    if not curve.get("ok"):
        return {"ok": False, "reason": curve.get("reason")}

    c = curve["courbe"]
    dispo = sorted(c)
    cible_min = max(duree_cible_h * 60, 15.0)

    if len(dispo) < 2:
        return {"ok": True, "v_kmh": float(c[dispo[0]]), "beta": float("nan"),
                "b_implicite": float("nan"), "extrapolation": True,
                "note": "Un seul bloc de référence : aucune décroissance "
                        "estimable, la vitesse est reprise telle quelle."}

    x = np.log(np.array(dispo, dtype=float))
    y = np.log(np.array([c[m] for m in dispo], dtype=float))
    beta, alpha = np.polyfit(x, y, 1)
    v = float(np.exp(alpha + beta * np.log(cible_min)))

    b_impl = 1.0 / (1.0 + beta) if beta > -1 else float("nan")
    facteur = cible_min / max(dispo)

    plausible = -0.20 <= beta <= -0.04
    return {
        "ok": True,
        "v_kmh": v,
        "beta": float(beta),
        "perte_par_doublement_pct": float((2 ** beta - 1) * 100),
        "b_implicite": float(b_impl),
        "beta_plausible": bool(plausible),
        "extrapolation": facteur > 2.0,
        "facteur_extrapolation": float(facteur),
        "note": (
            (f"Cible de {duree_cible_h:.1f} h contre un bloc mesuré maximal "
             f"de {max(dispo)} min, soit un facteur {facteur:.1f}. "
             "L'extrapolation est réelle." if facteur > 2.0 else "")
            + ("" if plausible else
               f" Décroissance de {beta:.3f} hors plage attendue "
               "[−0,20 ; −0,04] : la courbe n'est pas exploitable en l'état.")
        ).strip(),
    }


def cross_check(curve: dict, b_riegel: float,
                duree_cible_h: float = 9.0) -> dict:
    """
    Confronte l'exposant issu des meilleurs efforts à celui du modèle
    d'endurance. Deux chemins indépendants vers la même quantité.
    """
    bs = base_speed(curve, duree_cible_h)
    if not bs.get("ok") or np.isnan(bs.get("b_implicite", np.nan)):
        return {"ok": False, "reason": "Courbe insuffisante."}
    b_eff = bs["b_implicite"]
    ecart = abs(b_eff - b_riegel)
    return {
        "ok": True,
        "b_efforts": b_eff,
        "b_riegel": b_riegel,
        "ecart": ecart,
        "accord": ecart <= 0.05,
        "verdict": ("Les deux méthodes concordent : le modèle est cohérent "
                    "de bout en bout." if ecart <= 0.05 else
                    "Écart notable entre les deux méthodes. L'une des deux "
                    "repose sur un échantillon non représentatif — à "
                    "regarder avant de se fier à la prédiction."),
    }


def summarize(d: pd.DataFrame) -> dict:
    """Colonnes à stocker à l'import."""
    return best_efforts(d)
