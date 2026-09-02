"""
strava.py — Intégration Strava.

Décision d'architecture, à valider avant d'écrire une ligne de plus :

Tu demandes Strava + Polar + Bryton. Trois intégrations, c'est trois OAuth,
trois formats, trois systèmes de quotas et trois sources de panne. Or :

  - Polar Flow synchronise nativement vers Strava.
  - Bryton Active synchronise nativement vers Strava.
  - L'API Polar AccessLink ne fournit pas de rattrapage d'historique
    correct : elle livre les séances au fil de l'eau via un système de
    transactions. Elle est inadaptée à la reconstitution d'un passé.
  - Bryton n'a pas d'API publique documentée.

Recommandation : UNE seule intégration, Strava, en hub. Polar et Bryton y
poussent leurs séances. Tu divises la surface technique par trois pour une
perte de données nulle.

Contrainte à connaître : les quotas Strava sont limités (de l'ordre de
quelques centaines de requêtes par quart d'heure et quelques milliers par
jour — à vérifier sur la doc courante, ils ont évolué). Chaque activité
détaillée coûte 1 à 2 requêtes. Un rattrapage de 300 activités doit donc
être étalé et mis en cache, jamais rejoué à chaque ouverture de l'app.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
API = "https://www.strava.com/api/v3"

STREAM_KEYS = "time,latlng,altitude,heartrate,cadence,watts,velocity_smooth,moving"

# Scope minimal. activity:read_all inclut les activités privées.
SCOPE = "read,activity:read_all"

SPORT_MAP = {
    "Run": "trail", "TrailRun": "trail", "VirtualRun": "trail",
    "Ride": "velo", "VirtualRide": "velo", "GravelRide": "velo",
    "MountainBikeRide": "velo", "EBikeRide": "velo",
    "Hike": "rando", "Walk": "rando",
}


def authorize_url(client_id: str, redirect_uri: str) -> str:
    return f"{AUTH_URL}?" + urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": SCOPE,
    })


def exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    r = requests.post(TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "code": code, "grant_type": "authorization_code",
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def refresh(client_id: str, client_secret: str, refresh_token: str) -> dict:
    r = requests.post(TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "refresh_token": refresh_token, "grant_type": "refresh_token",
    }, timeout=30)
    r.raise_for_status()
    return r.json()


class Strava:
    def __init__(self, token: dict, client_id: str, client_secret: str,
                 on_refresh=None):
        self.token = token
        self.client_id = client_id
        self.client_secret = client_secret
        self.on_refresh = on_refresh
        self.last_rate: dict | None = None

    def _headers(self) -> dict:
        if self.token.get("expires_at", 0) - 60 < time.time():
            self.token = refresh(self.client_id, self.client_secret,
                                 self.token["refresh_token"])
            if self.on_refresh:
                self.on_refresh(self.token)
        return {"Authorization": f"Bearer {self.token['access_token']}"}

    def _get(self, path: str, **params):
        r = requests.get(f"{API}{path}", headers=self._headers(),
                         params=params, timeout=60)
        self._record_rate(r)
        if r.status_code == 429:
            raise RuntimeError(
                "Quota Strava atteint. Attends le prochain quart d'heure, "
                "puis relance la synchronisation — les activités déjà "
                "importées ne seront pas retéléchargées."
            )
        r.raise_for_status()
        return r.json()

    def _record_rate(self, r) -> None:
        """
        Mémorise le quota courant depuis les en-têtes.

        Strava expose deux jeux de compteurs : le quota global et le quota
        "read", plus restrictif (100 / 15 min et 1 000 / jour par défaut).
        Toutes les requêtes de ce client étant des GET, c'est le quota read
        qui contraint : on le privilégie quand il est présent.
        """
        lim = r.headers.get("X-ReadRateLimit-Limit") or r.headers.get("X-RateLimit-Limit")
        use = r.headers.get("X-ReadRateLimit-Usage") or r.headers.get("X-RateLimit-Usage")
        if not lim or not use:
            return
        try:
            self.last_rate = {
                "limit": [int(v) for v in lim.split(",")],
                "usage": [int(v) for v in use.split(",")],
            }
        except ValueError:
            pass

    def activities(self, after: datetime | None = None, per_page: int = 100,
                   max_pages: int = 20):
        """Itère les activités, de la plus récente à la plus ancienne."""
        params = {"per_page": per_page}
        if after:
            params["after"] = int(after.replace(tzinfo=timezone.utc).timestamp())
        for page in range(1, max_pages + 1):
            batch = self._get("/athlete/activities", page=page, **params)
            if not batch:
                return
            yield from batch

    def streams(self, activity_id: int) -> dict:
        return self._get(f"/activities/{activity_id}/streams",
                         keys=STREAM_KEYS, key_by_type="true")

    def athlete_zones(self) -> dict:
        """Zones FC configurées dans Strava — utile pour calibrer FCmax."""
        try:
            return self._get("/athlete/zones")
        except requests.HTTPError:
            return {}


def normalize_activity(a: dict) -> dict:
    """Métadonnées d'une activité Strava, dans notre schéma."""
    return {
        "activity_id": str(a["id"]),
        "source": "strava",
        "sport": SPORT_MAP.get(a.get("sport_type") or a.get("type"), "autre"),
        "strava_type": a.get("sport_type") or a.get("type"),
        "date": a.get("start_date"),
        "name": a.get("name"),
        "distance_km": (a.get("distance") or 0) / 1000,
        "d_plus": a.get("total_elevation_gain"),
        "duration_h": (a.get("moving_time") or 0) / 3600,
        "has_hr": bool(a.get("has_heartrate")),
        "device_watts": bool(a.get("device_watts")),
    }
