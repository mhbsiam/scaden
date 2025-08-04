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


def _train_with_keras3_fix(cdn: Scaden, data_path, train_datasets):
    """
    Run cdn.train(); if Keras 3 rejects saving to a bare directory
    ('Invalid filepath extension for saving'), save a `.keras` file
    inside that directory instead.
    """
    try:
        cdn.train(input_path=data_path, train_datasets=train_datasets)
    except ValueError as e:
        if "Invalid filepath extension for saving" in str(e):
            model_dir = cdn.model_dir  # e.g., heart_model/m256
            os.makedirs(model_dir, exist_ok=True)
            outfile = os.path.join(model_dir, "model.keras")
            # Training completed; only the save step failed. Save explicitly:
            cdn.model.save(outfile)
            logger.warning(f"Keras 3 workaround: saved model to {outfile}")
        else:
            raise


def training(
    data_path, train_datasets, model_dir, batch_size, learning_rate, num_steps, seed=0
):
    """
    Perform training of a Scaden model ensemble consisting of three different models
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
