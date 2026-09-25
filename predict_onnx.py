"""
Who's That Pokemon? -- lightweight prediction used by the website.

Runs the EfficientNet-B0 model trained by whoIsThatPokemon_torch.py, exported
to ONNX (`python whoIsThatPokemon_torch.py export-onnx`). Only needs numpy,
pillow and onnxruntime, so it fits on a free host (see Dockerfile), where
PyTorch or TensorFlow would not. Preprocessing mirrors TRANSFORM_EVAL there.

Usage:
    python predict_onnx.py predict <image_path> [--json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime
from PIL import Image

ONNX_PATH = Path(__file__).parent / "pokemon_model.onnx"


def preparer_image(chemin, img_size, moyenne, ecart_type):
    """Same steps as whoIsThatPokemon_torch: transparency flattened onto
    white, bilinear resize, [0, 1] scaling, ImageNet normalization, NCHW."""
    image = Image.open(chemin).convert("RGBA")
    fond = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(fond, image).convert("RGB")
    image = image.resize((img_size, img_size), Image.BILINEAR)

    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - np.array(moyenne, dtype=np.float32)) / np.array(ecart_type, dtype=np.float32)
    return x.transpose(2, 0, 1)[np.newaxis, ...]


def predire(chemin_image, sortie_json=False):
    session = onnxruntime.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    meta = {cle: json.loads(valeur) for cle, valeur in session.get_modelmeta().custom_metadata_map.items()}
    class_names = meta["class_names"]

    x = preparer_image(chemin_image, meta["img_size"], meta["mean"], meta["std"])
    logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]

    exp = np.exp(logits - logits.max())
    probabilites = exp / exp.sum()

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
    parser = argparse.ArgumentParser(description="Who's That Pokemon? (modèle ONNX)")
    sous_commandes = parser.add_subparsers(dest="commande", required=True)

    parser_predict = sous_commandes.add_parser("predict", help="Identifie le Pokémon dans une image")
    parser_predict.add_argument("image", help="Chemin vers l'image à identifier")
    parser_predict.add_argument("--json", action="store_true", help="Sortie JSON (pour un appel programmatique)")

    args = parser.parse_args()

    if not ONNX_PATH.exists():
        sys.exit("Aucun modèle ONNX. Lancez : python whoIsThatPokemon_torch.py train puis export-onnx")
    predire(args.image, sortie_json=args.json)
