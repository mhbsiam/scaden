"""
scaden Main functionality

Contains code to
- process a training datasets
- train a model
- perform predictions
"""

# Imports
import os
import logging
import pandas as pd
from anndata import read_h5ad
from scaden.model.architectures import architectures
from scaden.model.scaden import Scaden

logger = logging.getLogger(__name__)

"""
PARAMETERS
"""
# ==========================================#

# Extract architectures
M256_HIDDEN_UNITS = architectures["m256"][0]
M512_HIDDEN_UNITS = architectures["m512"][0]
M1024_HIDDEN_UNITS = architectures["m1024"][0]
M256_DO_RATES = architectures["m256"][1]
M512_DO_RATES = architectures["m512"][1]
M1024_DO_RATES = architectures["m1024"][1]

# ==========================================#


def _ensure_genes_file(model_dir: str, data_path: str):
    """
    Ensure <model_dir>/genes.txt exists. If absent and the training data is an .h5ad file,
    write genes.txt using adata.var_names from the training matrix.
    """
    genes_path = os.path.join(model_dir, "genes.txt")
    if os.path.isfile(genes_path):
        return  # nothing to do

    try:
        if isinstance(data_path, str) and data_path.lower().endswith(".h5ad") and os.path.isfile(data_path):
            adata = read_h5ad(data_path)
            pd.Series(adata.var_names).to_csv(genes_path, sep="\t")
            logger.warning(f"Wrote genes.txt to {model_dir}")
        else:
            logger.warning(
                f"genes.txt missing and training data not an .h5ad file (or not found): {data_path}. "
                f"Could not auto-generate genes.txt."
            )
    except Exception as ge:
        logger.warning(f"Could not write genes.txt to {model_dir}: {ge}")


def _train_with_keras3_fix(cdn: Scaden, data_path, train_datasets):
    """
    Run cdn.train(); if Keras 3 refuses to save to a bare directory
    ('Invalid filepath extension for saving'), save a `.keras` file inside the directory
    and also ensure that genes.txt exists there.
    """
    try:
        cdn.train(input_path=data_path, train_datasets=train_datasets)
        # If Scaden's internal save finished normally, genes.txt should already exist.
        # As a safety net, ensure it's there.
        _ensure_genes_file(cdn.model_dir, data_path)
    except ValueError as e:
        msg = str(e)
        if "Invalid filepath extension for saving" in msg:
            # Keras 3: model.save() requires .keras/.h5; save explicitly
            model_dir = cdn.model_dir
            os.makedirs(model_dir, exist_ok=True)
            outfile = os.path.join(model_dir, "model.keras")
            cdn.model.save(outfile)
            logger.warning(f"Scaden model saved to {outfile}")

            # Make sure genes.txt exists (derived from training data)
            _ensure_genes_file(model_dir, data_path)
        else:
            raise


def training(
    data_path, train_datasets, model_dir, batch_size, learning_rate, num_steps, seed=0
):
    """
    Perform training of three a scaden model ensemble consisting of three different models
    :param model_dir:
    :param batch_size:
    :param learning_rate:
    :param num_steps:
    :return:
    """
    # Convert training datasets
    if train_datasets == "":
        train_datasets = []
    else:
        train_datasets = train_datasets.split(",")
        logger.info(f"Training on: [cyan]{train_datasets}")

    # Training of M256 model
    logger.info("[cyan]Training M256 Model ... [/]")
    cdn256 = Scaden(
        model_dir=model_dir + "/m256",
        model_name="m256",
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
        hidden_units=M256_HIDDEN_UNITS,
        do_rates=M256_DO_RATES,
    )
    _train_with_keras3_fix(cdn256, data_path, train_datasets)
    del cdn256

    # Training of M512 model
    logger.info("[cyan]Training M512 Model ... [/]")
    cdn512 = Scaden(
        model_dir=model_dir + "/m512",
        model_name="m512",
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
        hidden_units=M512_HIDDEN_UNITS,
        do_rates=M512_DO_RATES,
    )
    _train_with_keras3_fix(cdn512, data_path, train_datasets)
    del cdn512

    # Training of M1024 model
    logger.info("[cyan]Training M1024 Model ... [/]")
    cdn1024 = Scaden(
        model_dir=model_dir + "/m1024",
        model_name="m1024",
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_steps=num_steps,
        seed=seed,
        hidden_units=M1024_HIDDEN_UNITS,
        do_rates=M1024_DO_RATES,
    )
    _train_with_keras3_fix(cdn1024, data_path, train_datasets)
    del cdn1024

    logger.info("[green]Training finished.")
