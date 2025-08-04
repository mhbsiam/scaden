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

# ==========================================#
# Extract architectures
M256_HIDDEN_UNITS = architectures["m256"][0]
M512_HIDDEN_UNITS = architectures["m512"][0]
M1024_HIDDEN_UNITS = architectures["m1024"][0]
M256_DO_RATES = architectures["m256"][1]
M512_DO_RATES = architectures["m512"][1]
M1024_DO_RATES = architectures["m1024"][1]
# ==========================================#

def _write_col_zero(items, out_path):
    """
    Write a TSV with a dummy index (first column) and a second column named '0'
    that contains the provided strings. This matches Scaden's read path:
        df = pd.read_table(..., index_col=0)
        list(df['0'])  # strings we wrote here
    """
    df = pd.DataFrame({"0": [str(x) for x in items]})
    df.to_csv(out_path, sep="\t", index=True)  # index becomes the dummy first column

def _ensure_genes_file(model_dir: str, data_path: str):
    genes_path = os.path.join(model_dir, "genes.txt")
    if os.path.isfile(genes_path):
        return
    if not (isinstance(data_path, str) and data_path.lower().endswith(".h5ad") and os.path.isfile(data_path)):
        raise RuntimeError("genes.txt missing and training data is not a readable .h5ad file.")
    adata = read_h5ad(data_path)
    genes = adata.var_names.astype(str).tolist()
    if not genes:
        raise RuntimeError("No genes found in adata.var_names; cannot write genes.txt.")
    _write_col_zero(genes, genes_path)
    logger.warning(f"Wrote genes.txt to {model_dir}")

def _ensure_celltypes_file(model_dir: str, data_path: str):
    path = os.path.join(model_dir, "celltypes.txt")
    if os.path.isfile(path):
        return
    if not (isinstance(data_path, str) and data_path.lower().endswith(".h5ad") and os.path.isfile(data_path)):
        raise RuntimeError("celltypes.txt missing and training data is not a readable .h5ad file.")
    adata = read_h5ad(data_path)
    labels = adata.uns.get("cell_types")
    if labels is None:
        raise RuntimeError("processed.h5ad must contain uns['cell_types'] to generate celltypes.txt.")
    labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    if not labels:
        raise RuntimeError("uns['cell_types'] is empty; cannot write celltypes.txt.")
    _write_col_zero(labels, path)
    logger.warning(f"Wrote celltypes.txt (from .uns['cell_types']) to {model_dir}")

def _post_train_ensure_metadata(model_dir: str, data_path: str):
    _ensure_genes_file(model_dir, data_path)
    _ensure_celltypes_file(model_dir, data_path)

def _train_with_keras3_fix(cdn: Scaden, data_path, train_datasets):
    try:
        cdn.train(input_path=data_path, train_datasets=train_datasets)
        _post_train_ensure_metadata(cdn.model_dir, data_path)
    except ValueError as e:
        if "Invalid filepath extension for saving" in str(e):
            model_dir = cdn.model_dir
            os.makedirs(model_dir, exist_ok=True)
            outfile = os.path.join(model_dir, "model.keras")
            cdn.model.save(outfile)
            logger.warning(f"Keras 3 workaround: saved model to {outfile}")
            _post_train_ensure_metadata(model_dir, data_path)
        else:
            raise

def training(
    data_path, train_datasets, model_dir, batch_size, learning_rate, num_steps, seed=0
):
    if train_datasets == "":
        train_datasets = []
    else:
        train_datasets = train_datasets.split(",")
        logger.info(f"Training on: [cyan]{train_datasets}")

    # M256
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

    # M512
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

    # M1024
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
