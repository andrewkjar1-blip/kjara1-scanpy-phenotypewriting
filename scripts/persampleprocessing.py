#!/usr/bin/env python3
"""
persampleprocessing.py

Process a single .h5ad file:
  - Load and validate adata.raw
  - Add sample metadata from annotation TSV
  - Filter cells by cell annotations (celltypelevel)
  - Rename obs_names with batch suffix
  - Write filtered .h5ad and metrics JSON

Corresponds to: process_single_file() in prepareSAIGEsinglecellphenotype_files.py
"""

import argparse
import os
import sys
import logging
import json
import re

import scanpy as sc
import pandas as pd
import anndata as ad

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: extract_batch_string
# ---------------------------------------------------------------------------
def extract_batch_string(filename: str) -> str:
    """
    Extract batch string from an h5ad filename.

    Rules
    -----
    - 'Pilot' in basename          → 'batch0'
    - regex r'batch\\s*(\\d+)'      → 'batch{N}'
    - Otherwise raise ValueError
    """
    basename = os.path.basename(filename)

    if "Pilot" in basename:
        log.info("Detected Pilot batch; returning 'batch0'.")
        return "batch0"

    match = re.search(r"batch\s*(\d+)", basename, re.IGNORECASE)
    if match:
        batch_num = match.group(1)
        batch_string = f"batch{batch_num}"
        log.info("Extracted batch string: %s", batch_string)
        return batch_string

    raise ValueError(
        f"Could not extract batch string from filename: {filename}"
    )


# ---------------------------------------------------------------------------
# Helper: add_metadata_to_adata
# ---------------------------------------------------------------------------
def add_metadata_to_adata(adata, annotation_path: str, sample_col: str):
    """
    Load annotation TSV, filter for Assay == 'Quantum', build TOS/Stanford
    maps, merge metadata into adata.obs, and store new_metadata_cols in
    adata.uns['new_metadata_cols'].

    Returns
    -------
    adata : AnnData  (filtered to matched samples, obs enriched with metadata)
    new_metadata_cols : list[str]
    matched_by : dict  {'TOS_ID': int, 'Stanford': int, 'Not found': int}
    """
    log.info("Loading annotation file: %s", annotation_path)
    annotation = pd.read_csv(annotation_path, sep="\t")

    # Filter for Quantum assay
    annotation_filtered = annotation[annotation["Assay"] == "Quantum"].copy()
    log.info(
        "Annotation rows after Assay=='Quantum' filter: %d",
        len(annotation_filtered),
    )

    # Drop rows where BOTH TOS_ID AND 'Sample ID - Stanford' are NaN
    annotation_clean = annotation_filtered.dropna(
        subset=["TOS_ID", "Sample ID - Stanford"], how="all"
    ).copy()
    log.info(
        "Annotation rows after NaN drop (how='all'): %d", len(annotation_clean)
    )

    # Build TOS map and Stanford map
    tos_map = annotation_clean[
        annotation_clean["TOS_ID"].notna()
    ].set_index("TOS_ID")
    stanford_map = annotation_clean[
        annotation_clean["Sample ID - Stanford"].notna()
    ].set_index("Sample ID - Stanford")
    log.info("TOS_ID map entries: %d", len(tos_map))
    log.info("Stanford map entries: %d", len(stanford_map))

    # Match each unique sample in adata.obs
    adata_samples = adata.obs[sample_col].unique()
    metadata_lookup: dict = {}
    matched_by = {"TOS_ID": 0, "Stanford": 0, "Not found": 0}

    for sample_id in adata_samples:
        if sample_id in tos_map.index:
            metadata_lookup[sample_id] = tos_map.loc[sample_id]
            matched_by["TOS_ID"] += 1
        elif sample_id in stanford_map.index:
            metadata_lookup[sample_id] = stanford_map.loc[sample_id]
            matched_by["Stanford"] += 1
        else:
            matched_by["Not found"] += 1

    log.info(
        "Match results — TOS_ID: %d  Stanford: %d  Not found: %d",
        matched_by["TOS_ID"],
        matched_by["Stanford"],
        matched_by["Not found"],
    )

    # Filter adata to matched samples only
    valid_samples = list(metadata_lookup.keys())
    adata = adata[adata.obs[sample_col].isin(valid_samples)].copy()
    log.info("adata shape after filtering to matched samples: %s", adata.shape)

    # Capture original obs columns BEFORE merge
    original_obs_cols = adata.obs.columns.tolist()

    # Build metadata DataFrame and merge into adata.obs
    metadata_df = pd.DataFrame.from_dict(metadata_lookup, orient="index")
    metadata_df.index.name = sample_col

    original_index = adata.obs.index
    adata.obs = adata.obs.merge(
        metadata_df,
        left_on=sample_col,
        right_index=True,
        how="left",
        suffixes=("_orig", ""),
    )
    adata.obs.index = original_index
    log.info("adata.obs shape after merge: %s", adata.obs.shape)

    # Derive new metadata columns (those not in original_obs_cols)
    new_metadata_cols = [
        col for col in adata.obs.columns if col not in original_obs_cols
    ]
    log.info("New metadata columns added (%d): %s", len(new_metadata_cols), new_metadata_cols)

    # Store in adata.uns BEFORE writing (allows concatandwrite.py to read back reliably)
    adata.uns["new_metadata_cols"] = new_metadata_cols

    return adata, new_metadata_cols, matched_by


# ---------------------------------------------------------------------------
# Helper: filter_adata_by_cell_annotations
# ---------------------------------------------------------------------------
def filter_adata_by_cell_annotations(
    adata,
    batch_string: str,
    cell_annotation_path: str,
    desired_celltype: list,
    celltype_level: str,
):
    """
    Load cellannotations.tsv, restrict to this batch's barcodes, filter to
    desired celltypes, and return adata filtered to matching cells.

    celltype_level should be 'l2_celltype'. Can also accept 'l1_celltype' for broader filtering.
    batch_prefix = f"{batch_string}_celltypist_"  (two underscores)
    """
    # Validate celltype_level
    if celltype_level not in ["l1_celltype", "l2_celltype"]:
        raise ValueError(
            f"celltype_level must be 'l1_celltype' or 'l2_celltype', got: {celltype_level!r}"
        )

    log.info("Loading cell annotation file: %s", cell_annotation_path)
    annotations = pd.read_csv(cell_annotation_path, sep="\t")
    log.info("Cell annotations loaded: %s", annotations.shape)

    # Build batch prefix — two underscores: one after batch_string, one after 'celltypist'
    batch_prefix = f"{batch_string}_celltypist_"
    log.info("Filtering annotations by batch prefix: %r", batch_prefix)

    batch_annotations = annotations[
        annotations["barcode"].str.startswith(batch_prefix, na=False)
    ].copy()
    log.info(
        "Annotations for %s: %d rows", batch_string, len(batch_annotations)
    )

    # Strip prefix to recover clean barcode
    batch_annotations["barcode_clean"] = batch_annotations["barcode"].str.replace(
        batch_prefix, "", regex=False
    )

    # Filter to desired celltypes
    celltype_annotations = batch_annotations[
        batch_annotations[celltype_level].isin(desired_celltype)
    ].copy()
    log.info(
        "Cells matching desired celltype(s) %s on level %s: %d",
        desired_celltype,
        celltype_level,
        len(celltype_annotations),
    )

    valid_barcodes = set(celltype_annotations["barcode_clean"].unique())
    celltype_counts = (
        celltype_annotations[celltype_level].value_counts().to_dict()
    )

    # Filter adata to valid barcodes
    adata_filtered = adata[adata.obs.index.isin(valid_barcodes)].copy()
    log.info("adata shape after cell annotation filter: %s", adata_filtered.shape)

    return adata_filtered, celltype_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-sample .h5ad processing for SAIGE single-cell pipeline."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to input .h5ad file.",
    )
    parser.add_argument(
        "--annotation_path",
        required=True,
        help="Path to sample annotation TSV (tab-separated).",
    )
    parser.add_argument(
        "--cell_annotation_path",
        required=True,
        help="Path to cellannotations.tsv (tab-separated).",
    )
    parser.add_argument(
        "--desired_celltype",
        nargs="+",
        required=True,
        help="One or more celltype strings to retain.",
    )
    parser.add_argument(
        "--celltype_level",
        required=True,
        help="Celltype column to use; must be 'l2_celltype'.",
    )
    parser.add_argument(
        "--sample_col",
        default="sample",
        help="Column name in adata.obs for sample ID (default: 'sample').",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Will be set once batch_string is known; used in the except block too
    batch_string = "unknown"

    try:
        # ------------------------------------------------------------------
        # 1. Validate input file exists
        # ------------------------------------------------------------------
        log.info("Input file: %s", args.input_file)
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(
                f"Input file not found: {args.input_file}"
            )

        # ------------------------------------------------------------------
        # 2. Extract batch_string
        # ------------------------------------------------------------------
        batch_string = extract_batch_string(args.input_file)
        log.info("Batch string: %s", batch_string)

        metrics_path = os.path.join(
            args.output_dir, f"{batch_string}.metrics.json"
        )
        output_h5ad = os.path.join(
            args.output_dir, f"{batch_string}.processed.h5ad"
        )

        # ------------------------------------------------------------------
        # 3. Load .h5ad
        # ------------------------------------------------------------------
        log.info("Loading .h5ad: %s", args.input_file)
        adata = sc.read_h5ad(args.input_file)
        log.info("Loaded adata shape: %s", adata.shape)

        # ------------------------------------------------------------------
        # 4. Validate adata.raw immediately after load
        # ------------------------------------------------------------------
        if adata.raw is None:
            raise ValueError(
                f"adata.raw is None in {args.input_file}. "
                "Expected raw counts to be stored before this step."
            )
        log.info("adata.raw validated: present (%d obs, %d vars)",
                 adata.raw.n_obs, adata.raw.shape[1])

        # ------------------------------------------------------------------
        # 5. Add metadata to adata
        # ------------------------------------------------------------------
        log.info("Adding metadata to adata...")
        adata, new_metadata_cols, matched_by = add_metadata_to_adata(
            adata,
            annotation_path=args.annotation_path,
            sample_col=args.sample_col,
        )

        # ------------------------------------------------------------------
        # 6. Filter by cell annotations
        # ------------------------------------------------------------------
        log.info(
            "Filtering by cell annotations (level=%s, celltypes=%s)...",
            args.celltype_level,
            args.desired_celltype,
        )
        adata, celltype_counts = filter_adata_by_cell_annotations(
            adata,
            batch_string=batch_string,
            cell_annotation_path=args.cell_annotation_path,
            desired_celltype=args.desired_celltype,
            celltype_level=args.celltype_level,
        )

        # ------------------------------------------------------------------
        # 7. Rename obs_names: "{cell}_{batch_string}"
        #    Underscore between cell barcode and batch_string is required.
        # ------------------------------------------------------------------
        log.info("Renaming obs_names with batch suffix...")
        adata.obs_names = [
            f"{cell}_{batch_string}" for cell in adata.obs_names
        ]
        adata.obs_names_make_unique()
        log.info("obs_names renamed and made unique.")

        # ------------------------------------------------------------------
        # 8. Handle empty result
        # ------------------------------------------------------------------
        if adata.n_obs == 0:
            log.warning(
                "No cells remain after filtering for batch %s. Writing EMPTY metrics.",
                batch_string,
            )
            metrics = {
                "batch_string": batch_string,
                "n_cells_after_filter": 0,
                "celltype_counts": {},
                "matched_by": matched_by,
                "status": "EMPTY",
            }
            with open(metrics_path, "w") as fh:
                json.dump(metrics, fh, indent=2)
            log.info("Metrics written to: %s", metrics_path)
            adata.write_h5ad(output_h5ad)
            sys.exit(0)

        # ------------------------------------------------------------------
        # 9. Ensure adata.uns['new_metadata_cols'] is set before writing
        #    (it was set inside add_metadata_to_adata, but re-assert here
        #     in case the filtered copy dropped uns)
        # ------------------------------------------------------------------
        adata.uns["new_metadata_cols"] = new_metadata_cols

        # ------------------------------------------------------------------
        # 10. Write filtered .h5ad
        # ------------------------------------------------------------------
        log.info("Writing processed .h5ad to: %s", output_h5ad)
        adata.write_h5ad(output_h5ad)
        log.info("Written: %s  shape=%s", output_h5ad, adata.shape)

        # ------------------------------------------------------------------
        # 11. Write metrics JSON
        # ------------------------------------------------------------------
        metrics = {
            "batch_string": batch_string,
            "n_cells_after_filter": int(adata.n_obs),
            "celltype_counts": {k: int(v) for k, v in celltype_counts.items()},
            "matched_by": {k: int(v) for k, v in matched_by.items()},
            "status": "SUCCESS",
        }
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        log.info("Metrics written to: %s", metrics_path)

        sys.exit(0)

    except Exception as exc:
        log.exception("Unhandled exception during processing: %s", exc)
        metrics_path = os.path.join(
            args.output_dir, f"{batch_string}.metrics.json"
        )
        metrics = {
            "batch_string": batch_string,
            "n_cells_after_filter": 0,
            "celltype_counts": {},
            "matched_by": {},
            "status": "FAILED",
            "reason": str(exc),
        }
        try:
            with open(metrics_path, "w") as fh:
                json.dump(metrics, fh, indent=2)
            log.info("Failure metrics written to: %s", metrics_path)
        except Exception as write_exc:
            log.error("Could not write failure metrics: %s", write_exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
