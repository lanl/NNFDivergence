"""
main.py: Run all experiments

This script runs experiments for ChemCam by default.
To switch to SuperCam, set DataType='PDS_SuperCam' (see main()).

For complete hyper-parameter settings, please refer to our paper.

Developed by Weizhi Li (CCS-6, Los Alamos National Laboratory).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# Prefer explicit imports over wildcard imports
from Options import parser  # assumes Options.py defines `parser = argparse.ArgumentParser(...)`
from DataUtils import download_and_split_data

from methods import (
    PLS,
    standard_regularization,
    standard_regularization_single_oxide,
    fdivergence_regularization,
    fdivergence_regularization_single_oxide,
)


def update_args(args: argparse.Namespace, **kwargs: Any) -> None:
    """Update argparse.Namespace in-place with provided keyword arguments."""
    for k, v in kwargs.items():
        setattr(args, k, v)


def main() -> None:
    # --------------------
    # Parse CLI + set paths
    # --------------------
    args = parser.parse_args()
    out_dir = Path(".")  # local directory (change if desired)

    # --------------------
    # Global defaults
    # --------------------
    update_args(
        args,
        DataType="PDS_ChemCam",  # switch to "PDS_SuperCam" if needed
        seed=1,
        epochs=1,
        batch_size=16,
        lr=1.0,
    )

    # --------------------
    # Data preparation
    # --------------------
    print(f"Downloading and splitting data for {args.DataType}...")
    download_and_split_data(args, str(out_dir))

    # --------------------
    # PLS (multi-oxide)
    # --------------------
    update_args(args, EmbeddingDim=10)
    print(f"Running PLS for {args.DataType} (EmbeddingDim={args.EmbeddingDim})...")
    PLS(args, str(out_dir))

    # --------------------
    # Standard 1D-CNN (multi-oxide)
    # --------------------
    print(f"Running 1D-CNN (no regularization) for {args.DataType}...")
    update_args(args, LossType="L2", l2delta=0.0)
    standard_regularization(args, str(out_dir))

    print(f"Running 1D-CNN (L1) for {args.DataType}...")
    update_args(args, LossType="L1", l1delta=1e-4)
    standard_regularization(args, str(out_dir))

    print(f"Running 1D-CNN (L2) for {args.DataType}...")
    update_args(args, LossType="L2", l2delta=1e-4)
    standard_regularization(args, str(out_dir))

    print(f"Running 1D-CNN (Dropout) for {args.DataType}...")
    update_args(args, LossType="Dropout", Dropout_prob=0.03)
    standard_regularization(args, str(out_dir))

    # --------------------
    # f-divergence regularization (multi-oxide)
    # --------------------
    print(f"Running 1D-CNN (f-divergence) for {args.DataType}...")
    update_args(args, LossType2="fdiv", Lambda=2.0, weight=5e-5, gamma=1e-3)
    fdivergence_regularization(args, str(out_dir))

    print(f"Running 1D-CNN (f-divergence + L1) for {args.DataType}...")
    update_args(args, LossType2="L1_fdiv", l1delta=7e-4, weight=7e-5, gamma=1.2e-3)
    fdivergence_regularization(args, str(out_dir))

    print(f"Running 1D-CNN (f-divergence + L2) for {args.DataType}...")
    update_args(args, LossType2="L2_fdiv", l2delta=7e-4, weight=7e-5, gamma=1.2e-3)
    fdivergence_regularization(args, str(out_dir))

    print(f"Running 1D-CNN (f-divergence + Dropout) for {args.DataType}...")
    update_args(args, LossType2="Dropout_fdiv", Dropout_prob=0.04, weight=7e-5, gamma=1.2e-3)
    fdivergence_regularization(args, str(out_dir))

    # --------------------
    # Single-oxide experiments
    # --------------------
    oxide_id = 2

    print(f"Running single-oxide 1D-CNN (Dropout) for {args.DataType}, oxide_id={oxide_id}...")
    update_args(args, oxide_id=oxide_id, LossType="Dropout", Dropout_prob=0.03)
    standard_regularization_single_oxide(args, str(out_dir))

    print(f"Running single-oxide 1D-CNN (f-divergence + Dropout) for {args.DataType}, oxide_id={oxide_id}...")
    update_args(args, oxide_id=oxide_id, LossType2="Dropout_fdiv", weight=7e-5, gamma=1.2e-3, Dropout_prob=0.03)
    fdivergence_regularization_single_oxide(args, str(out_dir))


if __name__ == "__main__":
    main()
