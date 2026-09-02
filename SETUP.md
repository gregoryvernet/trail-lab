# Mise en place — pas à pas

Six étapes, chacune avec un **point de contrôle**. Ne passe à la suivante que si le contrôle passe. Si une étape bloque, le problème est forcément dans celle-là, pas ailleurs.

Compte 30 à 45 minutes pour les étapes 1 à 3, qui suffisent à tout tester en local. Les étapes 4 à 6 mettent l'application en ligne.

---

## Étape 1 — Installer (15 min)

### 1.1 Python

Vérifie s'il est déjà là. Ouvre **PowerShell** (touche Windows, tape `powershell`) :

```powershell
python --version
```

Si tu vois `Python 3.10` ou plus, passe à 1.2. Sinon, télécharge Python sur python.org et **coche « Add Python to PATH »** pendant l'installation — c'est la case que tout le monde rate, et sans elle rien ne marchera ensuite.

### 1.2 Le projet

Décompresse le dossier `trail-lab` où tu veux, par exemple `D:\Greg\trail-lab`. Puis :

```powershell
cd D:\Greg\trail-lab
python -m venv .venv
.venv\Scripts\activate
```

Ton invite doit maintenant commencer par `(.venv)`. C'est un environnement isolé : les paquets installés ici ne toucheront pas le reste de ta machine.

```powershell
pip install -r requirements.txt
```

Trois à cinq minutes. Huit paquets, tous très courants : `streamlit` (l'interface web), `pandas` et `numpy` (le calcul), `plotly` (les graphiques), `gpxpy` et `fitparse` (la lecture des fichiers de trace), `supabase` et `requests` (uniquement pour le déploiement).

**Sur un poste d'entreprise**, commence par la version réduite — six paquets, pas de dépendance réseau ni base de données :

```powershell
pip install -r requirements-min.txt
```

Tout fonctionne en local avec celle-là. Tu ajouteras les deux autres au moment du déploiement.

Si `pip` échoue avec une erreur de connexion ou de certificat, c'est le proxy de l'entreprise. Deux contournements, dans cet ordre :

```powershell
pip install -r requirements-min.txt --proxy http://proxy.tonentreprise.fr:8080
pip install -r requirements-min.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

Si les deux échouent, il faut passer par le service informatique — ou revenir sur le poste perso.

> **À retenir** : à chaque nouvelle session PowerShell, il faut refaire `cd D:\Greg\trail-lab` puis `.venv\Scripts\activate`. Sans le `(.venv)`, les commandes échoueront.

### Point de contrôle 1

```powershell
python check.py
```

Attendu :

```
[  OK   ] Version de Python
[  OK   ] Dépendances
[  OK   ] Import des modules — 10 modules
[  OK   ] Coûts énergétiques (Minetti)
[  OK   ] Analyse d'une trace synthétique
[  OK   ] Modèle d'endurance
[  OK   ] Base locale (CSV)
[  OK   ] Plan Templiers — 56 séances · 8 semaines
```

La ligne `ATTENTION supabase absent` est normale à ce stade.

---

## Étape 2 — Tester sur un vrai fichier (10 min)

### 2.1 Un fichier d'abord

Prends **un seul** TCX Polar, pas tout le dossier :

```powershell
python check.py "D:\Greg\Code_Indice_UTMB\1. Fichiers GPX\TCX Polar\ta_sortie.tcx"
```

Regarde la ligne `Lecture`. Elle indique les capteurs trouvés. Si tu vois `MANQUE : cadence`, la détection marche/course sera dégradée — dis-le-moi, on ajustera.

### 2.2 L'application

```powershell
streamlit run app.py
```

Ton navigateur s'ouvre sur `localhost:8501`. Onglet **Sortie**, dépose ton TCX.

### Point de contrôle 2

Tu dois voir : distance, D+, GAP, VAM, découplage, le profil altimétrique, le tableau par bande de pente, et un bouton **Enregistrer dans l'historique**.

Vérifie deux choses :

- **La distance et le D+ correspondent à Polar Flow** (à 2-3 % près, l'altimétrie diffère selon les algorithmes de lissage). Si l'écart dépasse 10 %, arrête et signale-le.
- **Le type de séance suggéré**. Corrige-le si besoin dans le menu — c'est manuel, comme convenu.

Clique sur Enregistrer. Un fichier `data\activities.csv` apparaît. Ouvre-le : une ligne, tes chiffres.

Redépose le même fichier : tu dois lire « Déjà dans l'historique ».

---

## Étape 3 — Charger l'historique (20 min + traitement)

### 3.1 Inventaire d'abord

```powershell
python backfill.py "D:\Greg\Code_Indice_UTMB\1. Fichiers GPX\TCX Polar" --dry-run
```

Aucune écriture, juste le compte des fichiers par format.

### 3.2 Essai sur 20

```powershell
python backfill.py "D:\Greg\...\TCX Polar" --hr-rest 48 --hr-max 188 --limit 20
```

Remplace 48 et 188 par tes vraies valeurs — **c'est important**, elles conditionnent tout le calcul d'intensité et donc le filtrage du modèle.

### 3.3 Tout

Retire `--limit 20`. Compte 2 à 5 secondes par sortie.

### Point de contrôle 3

En fin de script :

```
Modèle d'endurance
  exposant b        : 1.1xx
  R²                : 0.9xx
  base              : NN trail / NN route
  surcharge terrain : +NN %
```

Trois lectures :

| Signal | Interprétation |
|---|---|
| `b` entre 1,00 et 1,30 | modèle exploitable |
| `b` hors plage, ou `HORS PLAGE PLAUSIBLE` | historique non représentatif — envoie-moi la sortie du script |
| moins de 15 sorties exploitables | prédiction non calibrable, on garde le suivi de plan seul |

Regarde aussi la répartition des types de séance. Le backfill classe automatiquement ; si tu vois beaucoup de « fractionné », c'est probablement faux — corrigeable dans `data\activities.csv`, colonne `session_type`.

Relance `streamlit run app.py` : les onglets **Historique**, **Course** et **Plan** ont maintenant de quoi travailler.

---

## Étape 4 — Persister les données (20 min)

Le disque de Streamlit Cloud est **éphémère** : ton `data\activities.csv` disparaîtrait au premier redémarrage du conteneur. À faire avant la mise en ligne, pas après.

1. Créer un compte sur **supabase.com**, offre gratuite, puis un projet.
2. Onglet **SQL Editor** → coller le contenu de `SUPABASE_SCHEMA` (dans `engine/store.py`, tout en bas) → **Run**.
3. **Project Settings → API** → copier `Project URL` et la clé `anon public`.
4. Créer `.streamlit\secrets.toml` à partir de `secrets.toml.example`, remplir `SUPABASE_URL`, `SUPABASE_KEY` et `APP_PASSWORD`.

### Point de contrôle 4

```powershell
python check.py
```

La ligne supabase doit passer en `OK`. Relance ensuite le backfill : il réécrira dans Supabase.

> `.streamlit\secrets.toml` est dans le `.gitignore`. Il ne doit **jamais** partir sur GitHub.

---

## Étape 5 — GitHub (15 min)

Sans ligne de commande.

1. Compte sur **github.com**.
2. **New repository** → nom `trail-lab` → **Public** → Create.
3. **Add file → Upload files** → glisser **tout le dossier sauf** :
   - `.streamlit\secrets.toml`
   - le dossier `data\`
   - le dossier `.venv\`
4. **Commit changes**.

### Point de contrôle 5

Sur la page du dépôt tu dois voir `app.py`, `check.py`, `backfill.py`, `requirements.txt`, les dossiers `engine` et `integrations`. Tu ne dois voir **ni** `secrets.toml`, **ni** `data`.

Si `secrets.toml` est passé : supprime-le du dépôt, puis régénère la clé Supabase — une clé publiée est une clé compromise.

Pour la suite, installe **GitHub Desktop** : tu édites en local, tu cliques Commit puis Push.

---

## Étape 6 — Mettre en ligne et sur iPhone (10 min)

1. **share.streamlit.io** → *Create app* → connecter GitHub → dépôt `trail-lab`, branche `main`, fichier `app.py` → **Deploy**.
2. Premier démarrage : 3 à 5 minutes, le temps d'installer les dépendances.
3. **Settings → Secrets** → coller le contenu de ton `secrets.toml` → l'app redémarre.

### iPhone et iPad

Safari → ton URL → bouton **Partager** → **Sur l'écran d'accueil**. Tu obtiens une icône qui lance l'application en plein écran, sans barre d'adresse.

### Point de contrôle 6

Depuis l'iPhone : mot de passe, puis l'onglet Plan affiche les séances de la semaine.

Le geste hebdomadaire :

1. Safari → `flow.polar.com` → ta séance → **Export TCX** (atterrit dans l'app Fichiers)
2. Ton application → onglet **Sortie** → dépôt → vérifier le type de séance → **Enregistrer**

> Après quelques jours sans usage, Streamlit Cloud met l'application en veille. Le premier chargement prend alors 30 à 60 secondes. C'est normal et gratuit.

---

## Si ça bloque

| Symptôme | Cause la plus probable |
|---|---|
| `python n'est pas reconnu` | case « Add Python to PATH » non cochée à l'installation |
| `ModuleNotFoundError` | environnement non activé — refaire `.venv\Scripts\activate` |
| `streamlit n'est pas reconnu` | idem |
| Distance très différente de Polar | signale-le, ne corrige pas toi-même |
| `Trace trop courte ou illisible` | fichier indoor, tapis ou home-trainer, sans GPS |
| Page blanche sur Streamlit Cloud | onglet *Manage app* en bas à droite → lire les logs |
| Historique vide après un redémarrage | Supabase non configuré — étape 4 |

Quand tu me signales un blocage, envoie la **sortie complète de `python check.py`**. C'est ce qui permet de localiser le problème sans deviner.

---

## Ordre recommandé

Les étapes 1 à 3 suffisent à tout utiliser en local, y compris le suivi du plan. Rien ne t'oblige à mettre en ligne tout de suite.

Le plan démarre le 24 août. La priorité est donc : **étapes 1 à 3 d'abord**, pour que la semaine 1 soit déjà rapprochée. Le déploiement peut attendre le week-end suivant.
