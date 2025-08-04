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


def _write_index_list_as_tsv_with_header_zero(index_list, out_path):
    """
    Write a TSV with:
      - index = items (genes or cell types)
      - single column named '0' filled with 1s
    This matches Scaden's read path:
        df = pd.read_table(..., index_col=0)
        list(df['0'])
    """
    idx = [str(x) for x in index_list]
    s = pd.Series([1] * len(idx), index=idx, name="0")
    s.to_csv(out_path, sep="\t", header=True)


def _ensure_genes_file(model_dir: str, data_path: str):
    """
    Ensure <model_dir>/genes.txt exists.
    Writes genes.txt with genes as index and a header column named '0'.
    """
    genes_path = os.path.join(model_dir, "genes.txt")
    if os.path.isfile(genes_path):
        return

    if not (isinstance(data_path, str) and data_path.lower().endswith(".h5ad") and os.path.isfile(data_path)):
        raise RuntimeError(
            "genes.txt missing and training data is not a readable .h5ad file. "
            "Provide a valid processed.h5ad."
        )

    adata = read_h5ad(data_path)
    genes = list(map(str, adata.var_names.tolist()))
    if not genes:
        raise RuntimeError("No genes found in adata.var_names; cannot write genes.txt.")
    _write_index_list_as_tsv_with_header_zero(genes, genes_path)
    logger.warning(f"Wrote genes.txt to {model_dir}")


def _ensure_celltypes_file(model_dir: str, data_path: str):
    """
    Ensure <model_dir>/celltypes.txt exists.
    Source: adata.uns['cell_types'] (accepts list/tuple/np.ndarray/Index).
    Writes a TSV with cell types as index and a header column named '0'.
    """
    path = os.path.join(model_dir, "celltypes.txt")
    if os.path.isfile(path):
        return

    if not (isinstance(data_path, str) and data_path.lower().endswith(".h5ad") and os.path.isfile(data_path)):
        raise RuntimeError(
            "celltypes.txt missing and training data is not a readable .h5ad file. "
            "Provide a valid processed.h5ad with uns['cell_types']."
        )

    adata = read_h5ad(data_path)
    labels = adata.uns.get("cell_types")
    if labels is None:
        raise RuntimeError(
            "processed.h5ad must contain uns['cell_types'] as a non-empty sequence "
            "to generate celltypes.txt."
        )

    # Robust conversion (handles numpy arrays, pandas Index, etc.)
    labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    if not labels_list:
        raise RuntimeError(
            "processed.h5ad must contain uns['cell_types'] as a non-empty sequence "
            "to generate celltypes.txt."
        )

    _write_index_list_as_tsv_with_header_zero([str(x) for x in labels_list], path)
    logger.warning(f"Wrote celltypes.txt (from .uns['cell_types']) to {model_dir}")


def _post_train_ensure_metadata(model_dir: str, data_path: str):
    """Ensure required sidecar files exist next to each model directory."""
    _ensure_genes_file(model_dir, data_path)
    _ensure_celltypes_file(model_dir, data_path)


def _train_with_keras3_fix(cdn: Scaden, data_path, train_datasets):
    """
    Run cdn.train(); if Keras 3 refuses to save to a bare directory
    ('Invalid filepath extension for saving'), save a `.keras` file inside the directory
    and also ensure that genes.txt and celltypes.txt exist there.
    """
    try:
        cdn.train(input_path=data_path, train_datasets=train_datasets)
        # If Scaden's internal save finished normally, sidecar files should exist.
        # As a safety net, ensure they're there (and fail loudly if prerequisites missing).
        _post_train_ensure_metadata(cdn.model_dir, data_path)
    except ValueError as e:
        msg = str(e)
        if "Invalid filepath extension for saving" in msg:
            # Keras 3: model.save() requires .keras/.h5; save explicitly
            model_dir = cdn.model_dir
            os.makedirs(model_dir, exist_ok=True)
            outfile = os.path.join(model_dir, "model.keras")
            cdn.model.save(outfile)
            logger.warning(f"Keras 3 workaround: saved model to {outfile}")

            # Ensure sidecar files (will raise if uns['cell_types'] or genes missing)
            _post_train_ensure_metadata(model_dir, data_path)
        else:
            raise


def training(
    data_path, train_datasets, model_dir, batch_size, learning_rate, num_steps, seed=0
):
    """
    Perform training of a scaden model ensemble consisting of three different models
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
