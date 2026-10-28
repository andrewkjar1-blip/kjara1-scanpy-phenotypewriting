#!/usr/bin/env python3
"""
concatandwrite.py

Collect all per-sample .processed.h5ad files produced by persampleprocessing.py.
Concatenate into a single AnnData object, run HVG selection, regression, scaling,
and PCA. Write one phenotype TSV.gz per chromosome using a parsed GTF.

All chromosomes are written in a single task (no scatter).
"""

import argparse
import gzip
import logging
import os
import sys
from math import ceil

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# ---------------------------------------------------------------------------
# Logging — all output goes to stderr
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate processed h5ad files and write per-chromosome phenotype TSVs."
    )
    parser.add_argument(
        "--processed_files",
        nargs="+",
        required=True,
        help="Space-separated list of .processed.h5ad paths.",
    )
    parser.add_argument(
        "--gtf_path",
        required=True,
        help="Path to gzipped GTF file (.gtf.gz).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write per-chromosome TSV.gz files.",
    )
    parser.add_argument(
        "--basename",
        required=True,
        help='Base name string for output files, e.g. "combined_l2_BIntermediate".',
    )
    parser.add_argument(
        "--additional_covariates",
        nargs="*",
        default=["percent.mt", "nCount_RNA"],
        help="Covariate column names to include in output (default: percent.mt nCount_RNA).",
    )
    parser.add_argument(
        "--n_pcs",
        type=int,
        default=10,
        help="Number of PCs to compute (default: 10).",
    )
    parser.add_argument(
        "--min_expression_pct",
        type=float,
        default=0.01,
        help="Min fraction of cells expressing a feature (default: 0.01).",
    )
    parser.add_argument(
        "--highly_variable_min_mean",
        type=float,
        default=0.0125,
        help="HVG min mean (default: 0.0125).",
    )
    parser.add_argument(
        "--highly_variable_max_mean",
        type=float,
        default=3.0,
        help="HVG max mean (default: 3.0).",
    )
    parser.add_argument(
        "--highly_variable_min_disp",
        type=float,
        default=0.5,
        help="HVG min dispersion (default: 0.5).",
    )
    parser.add_argument(
        "--scale_max_value",
        type=float,
        default=10.0,
        help="Max value for scaling (default: 10.0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

def parse_gtf(gtf_path):
    """
    Parse a gzipped GTF and return a DataFrame with columns:
        gene_name, gene_id, chr, start, end

    Only 'gene' rows are kept. Only standard chromosomes (1-22, X, Y) are kept.
    """
    log.info("Parsing GTF: %s", gtf_path)

    valid_chr = {
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19",
        "20", "21", "22", "X", "Y",
    }

    gene_data = []

    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or line.strip() == "":
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "gene":
                continue

            chrom = fields[0].replace("chr", "")
            if chrom not in valid_chr:
                continue

            attributes = fields[8]
            try:
                gene_id = (
                    attributes.split('gene_id "')[1].split('"')[0].rsplit(".", 1)[0]
                )
                gene_name = attributes.split('gene_name "')[1].split('"')[0]
            except IndexError:
                continue

            gene_data.append(
                {
                    "gene_name": gene_name,
                    "gene_id": gene_id,
                    "chr": chrom,
                    "start": fields[3],
                    "end": fields[4],
                }
            )

    gene_df = pd.DataFrame(gene_data)
    log.info("Parsed %d genes from GTF.", len(gene_df))
    return gene_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # STEP 1: Validate --processed_files is non-empty
    # ------------------------------------------------------------------
    if not args.processed_files:
        log.error("--processed_files is empty. Nothing to process.")
        raise ValueError("--processed_files list is empty.")

    log.info("Received %d candidate processed files.", len(args.processed_files))

    # ------------------------------------------------------------------
    # STEP 2: Validate each path; warn and skip missing files
    # ------------------------------------------------------------------
    valid_files = []
    for path in args.processed_files:
        if os.path.exists(path):
            valid_files.append(path)
        else:
            log.warning("File not found, skipping: %s", path)

    if not valid_files:
        log.error("No valid processed files remain after existence check.")
        sys.exit(1)

    log.info("%d valid files will be loaded.", len(valid_files))

    # ------------------------------------------------------------------
    # STEP 3-8: Load → concat → column groups → HVG → regress → scale → PCA
    # Wrapped in try/except; sys.exit(1) on failure.
    # ------------------------------------------------------------------
    try:
        # STEP 3: Load each .h5ad
        log.info("Loading .h5ad files...")
        adata_list = []
        for path in valid_files:
            log.info("  Loading: %s", path)
            adata_list.append(sc.read_h5ad(path))
        log.info("Loaded %d AnnData objects.", len(adata_list))

        # STEP 4: Concatenate
        log.info("Concatenating AnnData objects (axis=0, join='outer')...")
        adata = ad.concat(adata_list, axis=0, join="outer")
        log.info("Concatenated shape: %s", adata.shape)

        # Validate adata.raw after concat
        if adata.raw is None:
            raise ValueError(
                "adata.raw is None after ad.concat(). "
                "All input files must have adata.raw set."
            )
        log.info("adata.raw is present after concat. n_obs=%d, n_vars=%d",
                 adata.raw.n_obs, adata.raw.n_vars)

        # STEP 5: Read column groups from adata.uns
        log.info("Reading column groups from adata.uns...")

        # new_metadata_cols — written by persampleprocessing.py
        new_metadata_cols = adata.uns.get("new_metadata_cols", [])
        if not isinstance(new_metadata_cols, list):
            raise ValueError(
                "adata.uns['new_metadata_cols'] is not a list. "
                f"Got: {type(new_metadata_cols)}"
            )
        log.info("new_metadata_cols (%d): %s", len(new_metadata_cols), new_metadata_cols)

        # Cell-type-specific PC column names: PC_1, PC_2, ..., PC_{n_pcs}
        pc_col_names = [f"PC_{i + 1}" for i in range(args.n_pcs)]
        log.info("Cell-type PC column names: %s", pc_col_names)

        # original_obs_cols: everything NOT from metadata merge and NOT a PC column
        original_obs_cols = [
            col
            for col in adata.obs.columns
            if col not in new_metadata_cols and col not in pc_col_names
        ]
        log.info(
            "original_obs_cols (%d): %s", len(original_obs_cols), original_obs_cols
        )

        # STEP 6: Highly variable genes
        log.info("Computing highly variable genes...")
        sc.pp.highly_variable_genes(
            adata,
            min_mean=args.highly_variable_min_mean,
            max_mean=args.highly_variable_max_mean,
            min_disp=args.highly_variable_min_disp,
        )
        hvg_count = adata.var["highly_variable"].sum()
        log.info("Found %d highly variable genes.", hvg_count)

        adata = adata[:, adata.var["highly_variable"]]
        log.info("Subset to HVGs. Shape: %s", adata.shape)

        # STEP 7: Regression
        # Column names must match adata.obs exactly:
        #   nCount_RNA  (underscore after 'n', underscore after 'Count')
        #   percent.mt  (dot separator)
        log.info("Regressing out nCount_RNA and percent.mt...")
        sc.pp.regress_out(adata, ["nCount_RNA", "percent.mt"])
        log.info("Regression complete.")

        # STEP 8: Scaling
        log.info("Scaling data (max_value=%.2f)...", args.scale_max_value)
        sc.pp.scale(adata, max_value=args.scale_max_value)
        log.info("Scaling complete.")

        # STEP 9: PCA
        log.info("Computing PCA (%d components)...", args.n_pcs)
        sc.tl.pca(adata, svd_solver="arpack", n_comps=args.n_pcs)
        log.info("PCA complete.")

        # Build PCs DataFrame
        # Columns: PC_1, PC_2, ..., PC_{n_pcs}  (underscore before digit)
        pcs_df = pd.DataFrame(
            adata.obsm["X_pca"][:, : args.n_pcs],
            columns=[f"PC_{i + 1}" for i in range(args.n_pcs)],
            index=adata.obs_names,
        )
        log.info("PCs DataFrame shape: %s", pcs_df.shape)

        # STEP 10: Filter raw features by expression threshold
        log.info(
            "Filtering raw features by expression threshold (min_pct=%.4f)...",
            args.min_expression_pct,
        )
        n_cells = adata.raw.n_obs
        min_cells = ceil(n_cells * args.min_expression_pct)
        log.info(
            "n_cells=%d, min_expression_pct=%.4f → min_cells=%d",
            n_cells,
            args.min_expression_pct,
            min_cells,
        )

        # Subset raw to current obs BEFORE counting — required.
        # After HVG filtering, adata.obs_names may differ from adata.raw's full index.
        raw_subset = adata.raw[adata.obs_names]

        X_raw = raw_subset.X
        if hasattr(X_raw, "toarray"):
            X_raw = X_raw.toarray()
        feature_counts = np.asarray((X_raw > 0).sum(axis=0)).ravel()
        feature_mask = feature_counts > min_cells

        log.info(
            "Raw features before filter: %d | after filter: %d",
            adata.raw.shape[1],
            feature_mask.sum(),
        )

        adata_raw_filtered = adata.raw[:, feature_mask].copy()
        log.info("adata_raw_filtered shape: %s", adata_raw_filtered.shape)

    except Exception as exc:
        log.exception("Fatal error during concat/PCA/regression block: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 11: Parse GTF
    # ------------------------------------------------------------------
    try:
        gene_df = parse_gtf(args.gtf_path)
    except Exception as exc:
        log.exception("Fatal error parsing GTF: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 12: Write per-chromosome files
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    log.info("Output directory: %s", args.output_dir)

    additional_covariates = args.additional_covariates or []

    chromosomes = sorted(gene_df["chr"].unique())
    log.info("Chromosomes to process: %s", chromosomes)

    for chromosome in chromosomes:
        try:
            # Output filename — exact pattern:
            #   phenotypeFile_{basename}_chr{CHR}.tsv.gz
            output_file = os.path.join(
                args.output_dir,
                f"phenotypeFile_{args.basename}_chr{chromosome}.tsv.gz",
            )

            log.info("--- Chr %s ---", chromosome)

            # Genes on this chromosome
            genes_on_chr = gene_df[gene_df["chr"] == chromosome]["gene_name"].values

            # Intersect with genes present in filtered raw data
            genes_in_data = [
                g for g in genes_on_chr if g in adata_raw_filtered.var_names
            ]

            if len(genes_in_data) == 0:
                log.info("Chr %s: No genes found in data — skipping.", chromosome)
                continue

            log.info(
                "Chr %s: %d genes on chromosome, %d present in data.",
                chromosome,
                len(genes_on_chr),
                len(genes_in_data),
            )

            # Subset raw filtered data to this chromosome's genes
            adata_chr = adata_raw_filtered[:, genes_in_data]

            # Build expression DataFrame (sparse-safe)
            X_chr = adata_chr.X
            if hasattr(X_chr, "toarray"):
                X_chr = X_chr.toarray()
            expr_df = pd.DataFrame(
                X_chr,
                columns=adata_chr.var_names,
                index=adata_chr.obs_names,
            )

            # Append _raw_expr suffix to all gene columns
            expr_df = expr_df.add_suffix("_raw_expr")

            # Handle NaN values
            nan_count = expr_df.isna().sum().sum()
            if nan_count > 0:
                log.warning(
                    "Chr %s: Found %d NaN values in expression data — converting to 0.",
                    chromosome,
                    nan_count,
                )
                expr_df = expr_df.fillna(0)

            cells_in_chr = adata_chr.obs_names

            # Build components list for pd.concat
            components = [
                pd.DataFrame({"cell_id": cells_in_chr}),
                adata.obs.loc[cells_in_chr, original_obs_cols].reset_index(drop=True),
                adata.obs.loc[cells_in_chr, new_metadata_cols].reset_index(drop=True),
            ]

            # Additional covariates — skip entirely if empty
            if additional_covariates:
                components.append(
                    adata.obs.loc[cells_in_chr, additional_covariates].reset_index(
                        drop=True
                    )
                )
            else:
                components.append(pd.DataFrame())

            # Cell-type-specific PCs: PC_1 ... PC_{n_pcs}
            components.append(pcs_df.loc[cells_in_chr].reset_index(drop=True))

            # Expression columns
            components.append(expr_df.reset_index(drop=True))

            output_df = pd.concat(components, axis=1)

            # Rename 'sample' → 'Sample'
            output_df = output_df.rename(columns={"sample": "Sample"})

            # Write — pandas infers gzip from .gz extension
            output_df.to_csv(output_file, sep="\t", index=False)

            log.info(
                "Chr %s: wrote %s | shape=%s",
                chromosome,
                output_file,
                output_df.shape,
            )

        except Exception as exc:
            log.exception(
                "Error processing chromosome %s — skipping: %s", chromosome, exc
            )
            continue

    log.info("All chromosomes processed. Exiting successfully.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
