"""
scaden Main functionality

Contains code to
- process a training datasets
- train a model
- perform predictions
"""

import os
import shutil
import logging
import tensorflow as tf
from scaden.model.architectures import architectures
from scaden.model.scaden import Scaden

logger = logging.getLogger(__name__)

# Extract architectures
M256_HIDDEN_UNITS = architectures["m256"][0]
M512_HIDDEN_UNITS = architectures["m512"][0]
M1024_HIDDEN_UNITS = architectures["m1024"][0]
M256_DO_RATES = architectures["m256"][1]
M512_DO_RATES = architectures["m512"][1]
M1024_DO_RATES = architectures["m1024"][1]


def _has_savedmodel(dir_path: str) -> bool:
    return (
        os.path.isfile(os.path.join(dir_path, "saved_model.pb"))
        and os.path.isdir(os.path.join(dir_path, "variables"))
    )


def _copy_sidecars(base_dir: str, export_dir: str):
    """Ensure both genes.txt and celltypes.txt are present in export_dir."""
    for fname in ("genes.txt", "celltypes.txt"):
        src = os.path.join(base_dir, fname)
        dst = os.path.join(export_dir, fname)
        if os.path.isfile(src):
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
                logger.info(f"Copied {fname} to: {dst}")
        else:
            logger.warning(f"Expected {fname} not found in {base_dir}.")


def _prepare_savedmodel_dir(root_dir: str, subdir: str) -> str:
    """
    Make <root_dir>/<subdir> consumable by Scaden.predict():
      - Use an existing SavedModel if present.
      - Otherwise load model.keras/.h5 with Keras 3 custom mapping and export SavedModel to _export.
      - Copy genes.txt and celltypes.txt into _export.
    Returns the directory path to use as model_dir.
    """
    base = os.path.join(root_dir, subdir)
    export_dir = os.path.join(base, "_export")

    # Case 1: Already a SavedModel in the base directory
    if _has_savedmodel(base):
        _copy_sidecars(base, base)
        logger.info(f"Using existing SavedModel directory: {base}")
        return base

    # Case 2: Need to export SavedModel from a file
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

    if not _has_savedmodel(export_dir):
        os.makedirs(export_dir, exist_ok=True)
        logger.info(f"Loading model file for export: {model_file}")

        # Map 'softmax_v2' -> Keras 3 softmax; avoid compile dependency.
        custom_objects = {"softmax_v2": tf.keras.activations.softmax}
        model = tf.keras.models.load_model(
            model_file, custom_objects=custom_objects, compile=False
        )

        logger.info(f"Exporting SavedModel to: {export_dir}")
        # Clear export dir to avoid stale contents
        for item in os.listdir(export_dir):
            p = os.path.join(export_dir, item)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        model.export(export_dir)

    # Ensure both sidecar files are copied
    _copy_sidecars(base, export_dir)

    logger.info(f"Using exported SavedModel directory: {export_dir}")
    return export_dir


def _validate_prediction_input(data_path: str):
    """Fail fast if user passes an .h5ad to predict (Scaden expects a TSV)."""
    lower = data_path.lower()
    if lower.endswith(".h5ad") or lower.endswith(".h5"):
        raise ValueError(
            "Your prediction input appears to be an HDF5/H5AD file. "
            "Scaden 'predict' expects a tab-delimited text matrix (genes x samples). "
            "Please export your counts to a .txt/.tsv and try again."
        )


def prediction(model_dir, data_path, out_name, seed=0):
    """
    Perform prediction using a trained scaden ensemble
    :param model_dir: the directory containing the models
    :param data_path: the path to the *tab-delimited* gene expression file (genes x samples)
    :param out_name: name of the output prediction file
    :return:
    """

    # Guard against passing .h5ad by mistake
    _validate_prediction_input(data_path)

    # Prepare model directories that Scaden can consume
    m256_dir = _prepare_savedmodel_dir(model_dir, "m256")
    m512_dir = _prepare_savedmodel_dir(model_dir, "m512")
    m1024_dir = _prepare_savedmodel_dir(model_dir, "m1024")

    # Predict with each submodel
    cdn256 = Scaden(
        model_dir=m256_dir,
        model_name="m256",
        seed=seed,
        hidden_units=M256_HIDDEN_UNITS,
        do_rates=M256_DO_RATES,
    )
    preds_256 = cdn256.predict(input_path=data_path)

    cdn512 = Scaden(
        model_dir=m512_dir,
        model_name="m512",
        seed=seed,
        hidden_units=M512_HIDDEN_UNITS,
        do_rates=M512_DO_RATES,
    )
    preds_512 = cdn512.predict(input_path=data_path)

    cdn1024 = Scaden(
        model_dir=m1024_dir,
        model_name="m1024",
        seed=seed,
        hidden_units=M1024_HIDDEN_UNITS,
        do_rates=M1024_DO_RATES,
    )
    preds_1024 = cdn1024.predict(input_path=data_path)

    # Ensemble average
    preds = (preds_256 + preds_512 + preds_1024) / 3
    preds.to_csv(out_name, sep="\t")
    logger.info(f"[bold]Created cell composition predictions: [green]{out_name}[/]")
