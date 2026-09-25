"""
Who's That Pokemon? -- image recognition with TensorFlow.

Inspired by:
https://medium.com/@nawat.sun/capturing-pokemon-exploring-image-recognition-with-tensorflow-and-python-dca4da45e0d6

That article trains a flatten+dense network to guess a Pokemon's elemental
TYPE (18 classes) from a grayscale image, reaching ~12% accuracy. Here we
adapt the same overall pipeline (gather images, preprocess, train, evaluate,
predict) to a harder goal: recognizing the SPECIES itself (1025 classes,
every Pokemon through Gen 9). That needs a convolutional network instead of
the article's flat dense layers, since a flatten+dense model does not scale
to that many classes.

Usage:
    python whoIsThatPokemon.py train [--limit N] [--epochs N]
    python whoIsThatPokemon.py predict <image_path>
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import requests
from PIL import Image

IMG_SIZE = 96
POKEDEX_RANGE = range(1, 1026)  # National Pokedex, Gen 1 through Gen 9
DATA_DIR = Path(__file__).parent / "pokemon_dataset"
IDS_PATH = DATA_DIR / "ids.json"
MODEL_PATH = Path(__file__).parent / "pokemon_model.keras"
CLASS_NAMES_PATH = Path(__file__).parent / "pokemon_class_names.json"

POKEAPI_POKEMON_URL = "https://pokeapi.co/api/v2/pokemon/"

# Official sprites alone are clean renders on flat backgrounds, while user
# uploads are photos, screenshots, cards, fan art... Adding the top web image
# search results per species brings that real-world variety into training.
# Google/Bing Images only serve JS-rendered pages to scripts now, so results
# come from DuckDuckGo image search (largely backed by Bing's index).
IMAGES_WEB_PAR_ESPECE = 20
PREFIXE_IMAGE_WEB = "web_"
TAILLE_MAX_IMAGE_WEB = 256  # stored downscaled: training only uses IMG_SIZE
EN_TETES_HTTP = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
}

# Raster variants per species, spanning several generations' art styles
# (8-bit through modern official artwork) so each class has real visual
# variety instead of near-duplicate renders. Not every variant exists for
# every Pokemon (e.g. no back sprite past Gen 5); missing ones are skipped.
SPRITE_PATHS = [
    lambda s: s["front_default"],
    lambda s: s["back_default"],
    lambda s: s["front_shiny"],
    lambda s: s["back_shiny"],
    lambda s: s["other"]["official-artwork"]["front_default"],
    lambda s: s["other"]["official-artwork"]["front_shiny"],
    lambda s: s["other"]["home"]["front_default"],
    lambda s: s["other"]["home"]["front_shiny"],
    lambda s: s["versions"]["generation-i"]["red-blue"]["front_default"],
    lambda s: s["versions"]["generation-i"]["red-blue"]["back_default"],
    lambda s: s["versions"]["generation-i"]["yellow"]["front_default"],
    lambda s: s["versions"]["generation-ii"]["crystal"]["front_default"],
    lambda s: s["versions"]["generation-ii"]["crystal"]["front_shiny"],
    lambda s: s["versions"]["generation-iii"]["emerald"]["front_default"],
    lambda s: s["versions"]["generation-iii"]["emerald"]["back_default"],
    lambda s: s["versions"]["generation-iii"]["emerald"]["front_shiny"],
    lambda s: s["versions"]["generation-iv"]["diamond-pearl"]["front_default"],
    lambda s: s["versions"]["generation-iv"]["diamond-pearl"]["back_default"],
    lambda s: s["versions"]["generation-iv"]["diamond-pearl"]["front_shiny"],
    lambda s: s["versions"]["generation-v"]["black-white"]["front_default"],
    lambda s: s["versions"]["generation-v"]["black-white"]["back_default"],
    lambda s: s["versions"]["generation-v"]["black-white"]["front_shiny"],
    lambda s: s["versions"]["generation-vi"]["x-y"]["front_default"],
    lambda s: s["versions"]["generation-vi"]["x-y"]["front_shiny"],
    lambda s: s["versions"]["generation-vii"]["ultra-sun-ultra-moon"]["front_default"],
    lambda s: s["versions"]["generation-vii"]["ultra-sun-ultra-moon"]["front_shiny"],
    lambda s: s["versions"]["generation-viii"]["brilliant-diamond-shining-pearl"]["front_default"],
    lambda s: s["versions"]["generation-viii"]["brilliant-diamond-shining-pearl"]["front_shiny"],
    lambda s: s["versions"]["generation-ix"]["scarlet-violet"]["front_default"],
    lambda s: s["versions"]["generation-ix"]["scarlet-violet"]["front_shiny"],
]

# SPRITE_PATHS indices that are back views. They're kept out of the test set
# so every one of them is used for training: nobody uploads a photo of a
# Pokemon's back, so holding them out would waste training data and give a
# test score that doesn't reflect real use.
INDICES_SPRITES_DE_DOS = {1, 3, 9, 14, 17, 20}


def session_http():
    """One shared Session: a bare requests.get() builds a new SSL context per
    call (~1s, holding the GIL), which serializes concurrent downloads."""
    session = requests.Session()
    session.headers.update(EN_TETES_HTTP)
    adaptateur = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("https://", adaptateur)
    session.mount("http://", adaptateur)
    return session


def telecharger_image(session, chemin_image, url, tentatives=3):
    """Download one sprite, tolerating transient network errors.

    With ~3000+ concurrent downloads, an occasional dropped connection is
    expected; letting one bad request crash the whole batch (and lose
    everything already downloaded that run) isn't acceptable.
    """
    if chemin_image.exists():
        return

    for tentative in range(tentatives):
        try:
            image_response = session.get(url, timeout=15)
            if image_response.ok:
                chemin_image.write_bytes(image_response.content)
            return
        except requests.exceptions.RequestException:
            if tentative == tentatives - 1:
                print(f"Échec du téléchargement (ignoré) : {url}")


def telecharger_dataset(limit=None):
    """Download sprite variants per species into DATA_DIR (concurrently)."""
    ids = list(POKEDEX_RANGE)[:limit] if limit else list(POKEDEX_RANGE)
    DATA_DIR.mkdir(exist_ok=True)

    noms_vers_ids = json.loads(IDS_PATH.read_text(encoding="utf-8")) if IDS_PATH.exists() else {}
    taches = []
    session = session_http()

    for pokedex_id in ids:
        for tentative in range(3):
            try:
                response = session.get(f"{POKEAPI_POKEMON_URL}{pokedex_id}/", timeout=15)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if tentative == 2:
                    raise
        data = response.json()
        nom = data["name"]
        noms_vers_ids[nom] = pokedex_id

        dossier_espece = DATA_DIR / nom
        dossier_espece.mkdir(exist_ok=True)

        for index, extraire_url in enumerate(SPRITE_PATHS):
            try:
                url = extraire_url(data["sprites"])
            except (KeyError, TypeError):
                url = None

            if url:
                taches.append((dossier_espece / f"{index}.png", url))

        print(f"#{pokedex_id} {nom} référencé")

    IDS_PATH.write_text(json.dumps(noms_vers_ids), encoding="utf-8")

    print(f"Téléchargement de {len(taches)} images (les fichiers déjà présents sont ignorés)...")
    with ThreadPoolExecutor(max_workers=32) as pool:
        for _ in pool.map(lambda t: telecharger_image(session, *t), taches):
            pass


def rechercher_images_web(requete, tentatives=5):
    """Return image URLs for a web image search, retrying on rate limits.

    Uses Bing's cached thumbnail of each result rather than the original:
    originals are often multi-MB wallpapers or dead links, while thumbnails
    (~300px) are always served, fast, and still well above IMG_SIZE.
    """
    from ddgs import DDGS

    for tentative in range(tentatives):
        try:
            resultats = DDGS().images(requete, max_results=IMAGES_WEB_PAR_ESPECE * 2)
            return [r.get("thumbnail") or r["image"] for r in resultats]
        except Exception as erreur:
            if tentative == tentatives - 1:
                print(f"Recherche échouée (ignorée) pour '{requete}' : {erreur}")
                return []
            time.sleep(5 * 2 ** tentative)


def telecharger_image_web(session, url):
    """Download one web image and return it as a downscaled RGBA image, or
    None if it's unreachable or not a readable image."""
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.thumbnail((TAILLE_MAX_IMAGE_WEB, TAILLE_MAX_IMAGE_WEB))
        return image.convert("RGBA")
    except Exception:
        return None


def telecharger_images_web_espece(session, nom):
    """Save the first IMAGES_WEB_PAR_ESPECE usable search results for one
    species as web_XX.png (PNG so charger_dataset picks them up)."""
    dossier_espece = DATA_DIR / nom
    if any(dossier_espece.glob(f"{PREFIXE_IMAGE_WEB}*.png")):
        return  # already done on a previous run

    urls = rechercher_images_web(f"{nom.replace('-', ' ')} pokemon")

    # Download all candidates concurrently, then keep the first valid ones
    # in search-rank order so "first N" means the top-ranked results.
    with ThreadPoolExecutor(max_workers=16) as pool:
        images = [image for image in pool.map(lambda url: telecharger_image_web(session, url), urls) if image]

    for index, image in enumerate(images[:IMAGES_WEB_PAR_ESPECE]):
        image.save(dossier_espece / f"{PREFIXE_IMAGE_WEB}{index:02d}.png")

    print(f"{nom} : {min(len(images), IMAGES_WEB_PAR_ESPECE)} images web")


def telecharger_images_web():
    """Add web image search results to every species folder in DATA_DIR.

    Searches run one at a time: the search endpoint rate-limits bursts.
    """
    noms = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    session = session_http()
    for nom in noms:
        telecharger_images_web_espece(session, nom)


def charger_image(chemin):
    """Load an image, flatten transparency onto white, resize, normalize."""
    image = Image.open(chemin).convert("RGBA")
    fond = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(fond, image).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    return np.array(image, dtype=np.float32) / 255.0


def charger_dataset():
    """Build (images, labels, testables, class_names) arrays from DATA_DIR.

    testables flags the front-view sprites: the only images eligible for the
    test set (back views and web images always go to training).

    class_names holds {"id": pokedex_id, "name": species_name} so callers
    can look a Pokemon up by number (Tyradex doesn't resolve every PokeAPI
    name, e.g. "mr-mime" or "farfetchd").
    """
    noms_vers_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    noms = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    class_names = [{"id": noms_vers_ids[nom], "name": nom} for nom in noms]
    images, labels, testables = [], [], []

    for label_index, nom in enumerate(noms):
        for fichier in (DATA_DIR / nom).glob("*.png"):
            try:
                images.append(charger_image(fichier))
            except Exception:
                print(f"Image corrompue ignorée : {fichier}")
                fichier.unlink()
                continue
            labels.append(label_index)
            testables.append(fichier.stem.isdigit() and int(fichier.stem) not in INDICES_SPRITES_DE_DOS)

    return np.array(images), np.array(labels), np.array(testables), class_names


def separer_train_test(images, labels, testables, num_classes, max_test_par_classe=3):
    """Hold out up to a few front-view sprites per class for testing, rest
    for training.

    A plain random/stratified split breaks here: with only a handful of
    images per species, some classes would end up with zero test samples.
    """
    train_idx, test_idx = [], []

    for classe in range(num_classes):
        indices = np.where(labels == classe)[0]
        candidats = indices[testables[indices]]
        n_test = min(max_test_par_classe, len(candidats), len(indices) - 1)
        test_idx.extend(candidats[:n_test])
        train_idx.extend(np.setdiff1d(indices, candidats[:n_test]))

    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]


def creer_modele(num_classes):
    """Transfer learning on MobileNetV2 (ImageNet weights) instead of a
    from-scratch CNN: with only ~20 images per species, a small custom CNN
    has too little data to learn useful visual features on its own, while a
    pretrained backbone already knows shapes/textures/colors and only needs
    a small classification head trained on top.
    """
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    entrees = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(entrees)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)  # [0, 1] -> [-1, 1] expected by MobileNetV2
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    sorties = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(entrees, sorties)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model, base_model


def entrainer(limit=None, epochs=30):
    telecharger_dataset(limit)
    telecharger_images_web()

    global tf
    import tensorflow as tf

    images, labels, testables, class_names = charger_dataset()
    print(f"{len(images)} images chargées pour {len(class_names)} Pokémon")

    X_train, y_train, X_test, y_test = separer_train_test(images, labels, testables, len(class_names))
    print(f"{len(X_train)} images d'entraînement, {len(X_test)} de test")

    model, base_model = creer_modele(len(class_names))

    print("Phase 1 : entraînement de la tête de classification (MobileNetV2 gelé)")
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    print("Phase 2 : ajustement fin des dernières couches de MobileNetV2")
    base_model.trainable = True
    for couche in base_model.layers[:100]:
        couche.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    perte, precision = model.evaluate(X_test, y_test, verbose=0)
    print(f"Précision sur le jeu de test : {precision:.2%}")
    print(f"(hasard pur : {1 / len(class_names):.2%})")

    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, ensure_ascii=False), encoding="utf-8")
    print(f"Modèle sauvegardé dans {MODEL_PATH}")


def predire(chemin_image, sortie_json=False):
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))

    image = charger_image(chemin_image)
    logits = model.predict(image[np.newaxis, ...], verbose=0)[0]
    probabilites = tf.nn.softmax(logits).numpy()

    top3 = probabilites.argsort()[-3:][::-1]
    resultats = [
        {**class_names[index], "confidence": float(probabilites[index])}
        for index in top3
    ]

    if sortie_json:
        print(json.dumps(resultats))
    else:
        for rang, resultat in enumerate(resultats, start=1):
            print(f"{rang}. {resultat['name']} ({resultat['confidence']:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Who's That Pokemon? image recognition")
    sous_commandes = parser.add_subparsers(dest="commande", required=True)

    parser_train = sous_commandes.add_parser("train", help="Télécharge le dataset et entraîne le modèle")
    parser_train.add_argument("--limit", type=int, default=None, help="Limiter à N Pokémon (pour un test rapide)")
    parser_train.add_argument("--epochs", type=int, default=30)

    parser_predict = sous_commandes.add_parser("predict", help="Identifie le Pokémon dans une image")
    parser_predict.add_argument("image", help="Chemin vers l'image à identifier")
    parser_predict.add_argument("--json", action="store_true", help="Sortie JSON (pour un appel programmatique)")

    args = parser.parse_args()

    if args.commande in ("train",):
        entrainer(limit=args.limit, epochs=args.epochs)
    elif args.commande == "predict":
        if not MODEL_PATH.exists():
            sys.exit("Aucun modèle entraîné. Lancez d'abord : python whoIsThatPokemon.py train")
        import tensorflow as tf
        predire(args.image, sortie_json=args.json)
