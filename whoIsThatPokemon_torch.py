"""
Who's That Pokemon? -- PyTorch / torchvision variant.

Same goal, dataset and train/test split as whoIsThatPokemon.py (TensorFlow,
MobileNetV2), but with a different backbone, EfficientNet-B0, to compare
the two. The dataset must already be downloaded (run
`python whoIsThatPokemon.py train` once, or at least its download steps).

The machine this runs on has no GPU, so training is organized around that:
- Phase 1 (frozen backbone): backbone features are computed ONCE, then the
  classification head trains on those cached features. Each epoch takes
  seconds instead of a full forward pass over ~40k images.
- Phase 2 (fine-tuning): the last blocks of EfficientNet are unfrozen and
  the whole network trains on augmented images.

Usage:
    python whoIsThatPokemon_torch.py train [--limit N] [--epochs N] [--epochs-fine N]
    python whoIsThatPokemon_torch.py predict <image_path> [--json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

from whoIsThatPokemon import DATA_DIR, IDS_PATH, INDICES_SPRITES_DE_DOS, separer_train_test

IMG_SIZE = 128
MODEL_PATH = Path(__file__).parent / "pokemon_model_torch.pt"
BATCH_SIZE = 64
BLOCS_A_AJUSTER = 2  # last EfficientNet feature blocks unfrozen in phase 2

# ImageNet statistics expected by torchvision's pretrained weights
MOYENNE = [0.485, 0.456, 0.406]
ECART_TYPE = [0.229, 0.224, 0.225]

TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MOYENNE, ECART_TYPE),
])

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10, fill=255),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MOYENNE, ECART_TYPE),
])


def ouvrir_image(chemin):
    """Open an image with transparency flattened onto white (sprites are
    transparent PNGs; black-filled transparency would look like a dark
    background the model could latch onto)."""
    image = Image.open(chemin).convert("RGBA")
    fond = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(fond, image).convert("RGB")


def lister_dataset(limit=None):
    """List (files, labels, testables, class_names) without decoding pixels.

    Mirrors whoIsThatPokemon.charger_dataset so both models are trained and
    evaluated on exactly the same images.
    """
    noms_vers_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    noms = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())[:limit]
    class_names = [{"id": noms_vers_ids[nom], "name": nom} for nom in noms]
    fichiers, labels, testables = [], [], []

    for label_index, nom in enumerate(noms):
        for fichier in (DATA_DIR / nom).glob("*.png"):
            fichiers.append(fichier)
            labels.append(label_index)
            testables.append(fichier.stem.isdigit() and int(fichier.stem) not in INDICES_SPRITES_DE_DOS)

    return np.array(fichiers), np.array(labels), np.array(testables), class_names


class DatasetImages(Dataset):
    def __init__(self, fichiers, labels, transform):
        self.fichiers, self.labels, self.transform = fichiers, labels, transform

    def __len__(self):
        return len(self.fichiers)

    def __getitem__(self, index):
        try:
            image = ouvrir_image(self.fichiers[index])
        except Exception:
            print(f"Image corrompue remplacée par une image blanche : {self.fichiers[index]}")
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        return self.transform(image), int(self.labels[index])


def creer_modele(num_classes):
    """EfficientNet-B0 with ImageNet weights and a new classification head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    entree_tete = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(entree_tete, num_classes))
    return model


def extraire_features(model, chargeur):
    """Run the frozen backbone once over a dataset: (features, labels)."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, cibles in chargeur:
            x = model.avgpool(model.features(images)).flatten(1)
            features.append(x)
            labels.append(cibles)
    return torch.cat(features), torch.cat(labels)


def evaluer(fonction_logits, chargeur):
    correct = total = 0
    with torch.no_grad():
        for x, y in chargeur:
            correct += (fonction_logits(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def entrainer(limit=None, epochs=30, epochs_fine=5):
    torch.manual_seed(0)
    fichiers, labels, testables, class_names = lister_dataset(limit)
    num_classes = len(class_names)
    print(f"{len(fichiers)} images pour {num_classes} Pokémon")

    f_train, y_train, f_test, y_test = separer_train_test(fichiers, labels, testables, num_classes)
    print(f"{len(f_train)} images d'entraînement, {len(f_test)} de test")

    model = creer_modele(num_classes)
    perte = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("Phase 1 : extraction des features (EfficientNet-B0 gelé)...")
    debut = time.time()
    # Features are computed on un-augmented images: augmentation would need a
    # fresh backbone pass every epoch, which is what caching avoids.
    x_train, t_train = extraire_features(model, DataLoader(DatasetImages(f_train, y_train, TRANSFORM_EVAL), batch_size=BATCH_SIZE))
    x_test, t_test = extraire_features(model, DataLoader(DatasetImages(f_test, y_test, TRANSFORM_EVAL), batch_size=BATCH_SIZE))
    print(f"Features extraites en {time.time() - debut:.0f}s")

    chargeur_features = DataLoader(TensorDataset(x_train, t_train), batch_size=BATCH_SIZE, shuffle=True)
    chargeur_features_test = DataLoader(TensorDataset(x_test, t_test), batch_size=BATCH_SIZE)
    optimiseur = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.classifier.train()
        for x, y in chargeur_features:
            optimiseur.zero_grad()
            perte(model.classifier(x), y).backward()
            optimiseur.step()
        model.classifier.eval()
        print(f"Époque {epoch}/{epochs} - précision test : {evaluer(model.classifier, chargeur_features_test):.2%}")

    print(f"Phase 2 : ajustement fin des {BLOCS_A_AJUSTER} derniers blocs d'EfficientNet-B0")
    for parametre in model.features.parameters():
        parametre.requires_grad = False
    blocs_ajustes = model.features[-BLOCS_A_AJUSTER:]
    for parametre in blocs_ajustes.parameters():
        parametre.requires_grad = True

    chargeur_train = DataLoader(DatasetImages(f_train, y_train, TRANSFORM_TRAIN), batch_size=BATCH_SIZE, shuffle=True)
    chargeur_test = DataLoader(DatasetImages(f_test, y_test, TRANSFORM_EVAL), batch_size=BATCH_SIZE)
    optimiseur = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )

    for epoch in range(1, epochs_fine + 1):
        debut = time.time()
        # Frozen blocks stay in eval mode so their BatchNorm statistics
        # (learned on ImageNet) aren't overwritten by our small batches.
        model.eval()
        blocs_ajustes.train()
        model.classifier.train()
        for x, y in chargeur_train:
            optimiseur.zero_grad()
            perte(model(x), y).backward()
            optimiseur.step()
        model.eval()
        precision = evaluer(model, chargeur_test)
        print(f"Époque {epoch}/{epochs_fine} - précision test : {precision:.2%} ({time.time() - debut:.0f}s)")

    print(f"Précision sur le jeu de test : {precision if epochs_fine else evaluer(model, chargeur_test):.2%}")
    print(f"(hasard pur : {1 / num_classes:.2%})")

    torch.save({"state_dict": model.state_dict(), "class_names": class_names, "img_size": IMG_SIZE}, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")


def predire(chemin_image, sortie_json=False):
    sauvegarde = torch.load(MODEL_PATH, weights_only=True)
    class_names = sauvegarde["class_names"]
    model = creer_modele(len(class_names))
    model.load_state_dict(sauvegarde["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(TRANSFORM_EVAL(ouvrir_image(chemin_image)).unsqueeze(0))[0]
    probabilites = torch.softmax(logits, dim=0)

    confiances, indices = probabilites.topk(3)
    resultats = [
        {**class_names[index], "confidence": float(confiance)}
        for confiance, index in zip(confiances.tolist(), indices.tolist())
    ]

    if sortie_json:
        print(json.dumps(resultats))
    else:
        for rang, resultat in enumerate(resultats, start=1):
            print(f"{rang}. {resultat['name']} ({resultat['confidence']:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Who's That Pokemon? (PyTorch / EfficientNet-B0)")
    sous_commandes = parser.add_subparsers(dest="commande", required=True)

    parser_train = sous_commandes.add_parser("train", help="Entraîne le modèle sur pokemon_dataset/")
    parser_train.add_argument("--limit", type=int, default=None, help="Limiter à N Pokémon (pour un test rapide)")
    parser_train.add_argument("--epochs", type=int, default=30, help="Époques de la phase 1 (tête seule)")
    parser_train.add_argument("--epochs-fine", type=int, default=5, help="Époques de la phase 2 (ajustement fin)")

    parser_predict = sous_commandes.add_parser("predict", help="Identifie le Pokémon dans une image")
    parser_predict.add_argument("image", help="Chemin vers l'image à identifier")
    parser_predict.add_argument("--json", action="store_true", help="Sortie JSON (pour un appel programmatique)")

    args = parser.parse_args()

    if args.commande == "train":
        entrainer(limit=args.limit, epochs=args.epochs, epochs_fine=args.epochs_fine)
    elif args.commande == "predict":
        if not MODEL_PATH.exists():
            sys.exit("Aucun modèle entraîné. Lancez d'abord : python whoIsThatPokemon_torch.py train")
        predire(args.image, sortie_json=args.json)
