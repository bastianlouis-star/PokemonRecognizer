# Pokédex

Un Pokédex web (façon jeu vidéo) qui permet de :

- **Rechercher** un Pokémon de la 1ère génération par nom (français) ou par numéro, et afficher sa fiche (types, stats, évolutions, sprites...) via les API [Tyradex](https://tyradex.app/) et [PokéAPI](https://pokeapi.co/).
- **"Qui est ce Pokémon ?"** : envoyer une photo et laisser un modèle de deep learning (TensorFlow / Keras, transfer learning sur MobileNetV2) deviner de quelle espèce il s'agit parmi les 151 Pokémon de la génération 1.

## Structure du projet

| Fichier / dossier | Rôle |
|---|---|
| `pokemon.html`, `pokemon.css`, `pokemon.js` | Interface web (le Pokédex) |
| `server.js` | Serveur HTTP Node.js : sert les fichiers statiques et expose `POST /api/identifier` |
| `whoIsThatPokemon.py` | Script Python : téléchargement du dataset, entraînement et prédiction du modèle |
| `pokemon_model.keras` | Modèle entraîné (suivi via [Git LFS](https://git-lfs.com/)) |
| `pokemon_class_names.json` | Liste des classes (id + nom) reconnues par le modèle |
| `pokemon_dataset/` | Images d'entraînement téléchargées depuis PokéAPI (non versionné, voir `.gitignore`) |
| `requirements.txt` | Dépendances Python |

## Prérequis

- [Node.js](https://nodejs.org/) (testé avec Node 24)
- [Python 3.12](https://www.python.org/) — TensorFlow ne supporte pas encore les versions plus récentes (ex. 3.14)
- [Git LFS](https://git-lfs.com/) pour cloner le modèle entraîné

## Installation

1. Cloner le dépôt (Git LFS récupère automatiquement `pokemon_model.keras` si `git lfs install` a été fait au moins une fois sur la machine) :

   ```bash
   git lfs install
   git clone <url-du-depot>
   cd Pokemon
   ```

2. Créer l'environnement virtuel Python et installer les dépendances :

   ```bash
   py -3.12 -m venv .venv
   ./.venv/Scripts/pip install -r requirements.txt
   ```

## Lancer le Pokédex

```bash
node server.js
```

Puis ouvrir [http://localhost:3000](http://localhost:3000).

- La recherche par nom/numéro fonctionne directement (appels aux API publiques Tyradex/PokéAPI).
- Le bouton en forme d'œil permet d'envoyer une image ; le serveur exécute `whoIsThatPokemon.py predict` via l'environnement virtuel (`.venv`) et renvoie le top 3 des Pokémon les plus probables.

## Entraîner le modèle

Le modèle fourni (`pokemon_model.keras`) est déjà entraîné, mais peut être régénéré :

```bash
./.venv/Scripts/python whoIsThatPokemon.py train
```

- Télécharge automatiquement le dataset d'images (sprites de plusieurs générations/styles par espèce) depuis PokéAPI dans `pokemon_dataset/`.
- Ajoute pour chaque espèce les 20 premiers résultats d'une recherche d'images web (`<nom> pokemon`, via DuckDuckGo : Google/Bing Images ne sont plus accessibles par script), enregistrés en `web_XX.png`. Ces images (artworks, cartes, captures, fan arts...) rapprochent le dataset des photos réellement envoyées. Les espèces déjà traitées sont ignorées aux lancements suivants.
- Entraîne le modèle en deux phases (tête de classification puis fine-tuning des dernières couches de MobileNetV2).
- Sauvegarde le résultat dans `pokemon_model.keras` et `pokemon_class_names.json`.

Options utiles :

- `--limit N` : n'entraîner que sur les N premiers Pokémon (pour un test rapide).
- `--epochs N` : nombre d'époques de la phase 1 (30 par défaut).

## Prédire en ligne de commande

```bash
./.venv/Scripts/python whoIsThatPokemon.py predict chemin/vers/image.png
```

Ajouter `--json` pour une sortie JSON (c'est ce que `server.js` utilise en interne).

## Variante PyTorch (EfficientNet-B0)

`whoIsThatPokemon_torch.py` entraîne un second modèle avec PyTorch / torchvision (EfficientNet-B0 pré-entraîné sur ImageNet) sur le même dataset et le même découpage entraînement/test, pour comparer les deux approches. Le dataset doit déjà être téléchargé (via `whoIsThatPokemon.py train`).

```bash
./.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python whoIsThatPokemon_torch.py train
./.venv/Scripts/python whoIsThatPokemon_torch.py predict chemin/vers/image.png --json
```

Le modèle est sauvegardé dans `pokemon_model_torch.pt` ; la sortie `--json` a le même format que celle de `whoIsThatPokemon.py`.
