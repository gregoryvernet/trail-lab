# Trail Lab

Analyse de performance trail et vélo, prédiction de temps de course et suivi
de plan d'entraînement. Application personnelle, préparation du Trail des
Templiers 2026 — 80 km, 3 400 D+, 18 octobre.

## Ce que fait l'application

| Onglet | Contenu |
|---|---|
| Sortie | Analyse d'une trace GPX, TCX ou FIT. La discipline choisie conditionne les indicateurs affichés et le plafond anti-décrochage GPS |
| Historique | État par bande de pente : distance, vitesse, FC, cadence, coût cardiaque, seuil marche/course, courbe vitesse-durée |
| Tendance | Progression sur 4 horizons (3, 5, 10, 20 sorties), coût de relance après montée et descente, découplage cardiaque, charge |
| Vélo | Puissance, W/kg, meilleurs efforts 15/30/60 min, courbe puissance-durée |
| Course | Distance équivalente, temps prédit avec intervalle, plan par segment |
| Plan | Profil de charge, séances semaine par semaine, saisie du réalisé |
| Réglages | Profil physiologique, import d'archive, cohérence des disciplines |

## Fondations

- **Coûts énergétiques** : polynômes de Minetti et al. (2002), valides de −45 % à +45 %
- **Unité de référence** : le kilomètre-effort (distance + D+/100), retenu après comparaison de sept formules par validation croisée
- **Prédiction** : `T = a · KE^b`, deux paramètres calibrés sur les sorties trail dans la plage de distance de la cible
- **Charge** : TRIMP de Banister, seul agrégat commun au trail et au vélo

Le référentiel complet des indicateurs, avec leur construction et leur
fiabilité, se trouve dans `indicateurs.html`.

## Mise en place

Voir `SETUP.md`. En résumé :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-min.txt
python check.py
streamlit run app.py
```

## Outils en ligne de commande

| Script | Rôle |
|---|---|
| `check.py` | Autodiagnostic — à lancer avant tout |
| `backfill.py` | Import de l'archive Strava ou d'un dossier local |
| `repair.py` | Renseigne les disciplines sans retraiter les traces |
| `disciplines.py` | Export et réimport CSV pour reclasser à la main |
| `migrate.py` | Récupère les saisies faites avant un changement de clé |
| `diagnose.py` | Diagnostic du modèle d'endurance |

## Déploiement

Le dépôt est public, les données ne le sont pas : `.gitignore` exclut
`secrets.toml` et `data/`. Les identifiants Supabase et le mot de passe
d'accès vivent dans les secrets de Streamlit Cloud, côté serveur.

Le schéma de base est dans `schema.sql`, réexécutable sans risque.

## Limites connues

- 25 sorties de calibrage, une seule au-delà de 80 km-effort
- Minetti est calibré sur tapis roulant : le sol technique n'est pas modélisé
- La détection du type de séance rate les côtes courtes, d'où la confirmation manuelle
- Altitude reconstituée par modèle de terrain à 25 m sur les sorties sans altimètre
