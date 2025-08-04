"""
scaden Main functionality

Contains code to
- process a training datasets
- train a model
- perform predictions
"""

# Imports
import os
import shutil
import logging
import tensorflow as tf
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


def _has_savedmodel(dir_path: str) -> bool:
    """Check if a TensorFlow SavedModel exists at dir_path."""
    return (
        os.path.isfile(os.path.join(dir_path, "saved_model.pb"))
        and os.path.isdir(os.path.join(dir_path, "variables"))
    )


def _prepare_savedmodel_dir(root_dir: str, subdir: str) -> str:
    """
    Ensure that <root_dir>/<subdir> can be consumed by Scaden.predict(), which expects:
      - a TF SavedModel directory at the provided path, and
      - a 'genes.txt' file in the same directory.

    If we only have a '.keras' or '.h5' file (from Keras 3 saving),
    we will:
      1) load that file with a custom mapping for 'softmax_v2',
      2) export a SavedModel into <root_dir>/<subdir>/_export,
      3) copy 'genes.txt' from <root_dir>/<subdir> into the _export dir.

    Returns the directory that Scaden should use as model_dir (either the original subdir
    if it already contains a SavedModel, or the new _export directory).
    """
    base = os.path.join(root_dir, subdir)
    export_dir = os.path.join(base, "_export")
    genes_src = os.path.join(base, "genes.txt")

    # Case 1: Already a SavedModel in the base directory
    if _has_savedmodel(base):
        if not os.path.isfile(genes_src):
            logger.warning(f"Expected genes.txt not found in {base}.")
        logger.info(f"Using existing SavedModel directory: {base}")
        return base

    # Case 2: Need to export a SavedModel from a file
    model_file = None
    for fname in ("model.keras", "model.h5"):
        candidate = os.path.join(base, fname)
        if os.path.isfile(candidate):
            model_file = candidate
            break

    if model_file is None:
        logger.warning(
            f"No SavedModel or model file (.keras/.h5) found in {base}. "
            f"Scaden may fail to load the model."
        )
        return base

    # Export only if not already exported
    if not _has_savedmodel(export_dir):
        os.makedirs(export_dir, exist_ok=True)
        logger.info(f"Loading model file for export: {model_file}")

        # --- IMPORTANT FIX: map 'softmax_v2' to a valid Keras 3 activation ---
        custom_objects = {"softmax_v2": tf.keras.activations.softmax}
        model = tf.keras.models.load_model(
            model_file, custom_objects=custom_objects, compile=False
        )

        logger.info(f"Exporting SavedModel to: {export_dir}")
        # Clear any previous contents in export_dir
        for item in os.listdir(export_dir):
            item_path = os.path.join(export_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        model.export(export_dir)

    # Ensure genes.txt is available alongside the SavedModel
    if os.path.isfile(genes_src):
        genes_dst = os.path.join(export_dir, "genes.txt")
        if not os.path.isfile(genes_dst):
            shutil.copy2(genes_src, genes_dst)
            logger.info(f"Copied genes.txt to: {genes_dst}")
    else:
        logger.warning(f"Expected genes.txt not found in {base}.")

    logger.info(f"Using exported SavedModel directory: {export_dir}")
    return export_dir


def prediction(model_dir, data_path, out_name, seed=0):
    """
    Perform prediction using a trained scaden ensemble
    :param model_dir: the directory containing the models
    :param data_path: the path to the gene expression file
    :param out_name: name of the output prediction file
    :return:
    """

    # Prepare model directories that Scaden can consume
    m256_dir = _prepare_savedmodel_dir(model_dir, "m256")
    m512_dir = _prepare_savedmodel_dir(model_dir, "m512")
    m1024_dir = _prepare_savedmodel_dir(model_dir, "m1024")

    # Small model predictions
    cdn256 = Scaden(
        model_dir=m256_dir,
        model_name="m256",
        seed=seed,
        hidden_units=M256_HIDDEN_UNITS,
        do_rates=M256_DO_RATES,
    )
    preds_256 = cdn256.predict(input_path=data_path)

    # Mid model predictions
    cdn512 = Scaden(
        model_dir=m512_dir,
        model_name="m512",
        seed=seed,
        hidden_units=M512_HIDDEN_UNITS,
        do_rates=M512_DO_RATES,
    )
    preds_512 = cdn512.predict(input_path=data_path)

    # Large model predictions
    cdn1024 = Scaden(
        model_dir=m1024_dir,
        model_name="m1024",
        seed=seed,
        hidden_units=M1024_HIDDEN_UNITS,
        do_rates=M1024_DO_RATES,
    )
    preds_1024 = cdn1024.predict(input_path=data_path)

    # Average predictions
    preds = (preds_256 + preds_512 + preds_1024) / 3
    preds.to_csv(out_name, sep="\t")
    logger.info(f"[bold]Created cell composition predictions: [green]{out_name}[/]")
