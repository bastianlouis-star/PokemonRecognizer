# Image for hosting the Pokédex (Node server + Python prediction), e.g. on
# Render's free plan (see render.yaml). Only prediction is supported, with the
# TFLite model: the dataset, TensorFlow and training dependencies are left out.
# Regenerate the model after retraining: python whoIsThatPokemon.py export-tflite
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHON=python3 \
    PORT=10000
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

COPY --chown=user server.js pokemon.html pokemon.css pokemon.js whoIsThatPokemon.py \
    pokemon_model.tflite pokemon_class_names.json ./

EXPOSE 10000
CMD ["node", "server.js"]
