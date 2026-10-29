#!/usr/bin/env python3
"""
concatandwrite.py

Collect all per-sample .processed.h5ad files produced by persampleprocessing.py.
Concatenate into a single AnnData object and generate pseudobulk expression
phenotype BED files and paired covariate files per chromosome, suitable for
input to a QTL mapping tool (e.g., TensorQTL).

Pipeline summary
----------------
  1.  Load & concatenate .h5ad files.
  2.  Per-cell CP10K normalization (sparse).
  3.  Build donor-level pseudobulk sample index.
  4.  Filter donors by minimum cell count.
  5.  Average CP10K per donor  →  pseudobulk matrix.
  6.  Gene filter (expressed in ≥ min_sample_pct of donors).
  7.  log2(x + 1) transform.
  8.  Pseudobulk PCA (covariates).
  9.  Inverse-rank normalization (IRN) per gene.
  10. Assemble sample metadata (donor-level + cell-level mean covariates).
  11. Parse GTF.
  12. Load cis-window table.
  13. Write per-chromosome phenotype BED files.
  14. Write per-chromosome covariate files.
"""

import argparse
import gzip
import logging
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        description=(
            "Concatenate processed h5ad files and write per-chromosome "
            "pseudobulk phenotype BED files and covariate files."
        )
    )

    # ── kept from previous version ─────────────────────────────────────────
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
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--basename",
        required=True,
        help='Base name string for output files, e.g. "combined_l2_BIntermediate".',
    )

    # ── new arguments ───────────────────────────────────────────────────────
    parser.add_argument(
        "--donor_key",
        type=str,
        default="NWD_ID",
        help="Column in adata.obs identifying the donor/individual (default: NWD_ID).",
    )
    parser.add_argument(
        "--celltype_label",
        type=str,
        required=True,
        help="Cell type label string used for metadata and log output.",
    )
    parser.add_argument(
        "--cis_window_path",
        type=str,
        required=True,
        help="Path to cis-window TSV (gene_cis_windows_hg38_100kb.tsv).",
    )
    parser.add_argument(
        "--min_cells_per_sample",
        type=int,
        default=10,
        help="Minimum cells per donor to retain that donor's pseudobulk sample (default: 10).",
    )
    parser.add_argument(
        "--min_sample_pct",
        type=float,
        default=0.50,
        help=(
            "Minimum fraction of donors in which a gene must have avg CP10K > 0 "
            "to be retained (default: 0.50)."
        ),
    )
    parser.add_argument(
        "--n_pseudobulk_pcs",
        type=int,
        default=5,
        help="Number of pseudobulk PCs to compute and include as covariates (default: 5).",
    )
    parser.add_argument(
        "--covariate_cols",
        nargs="+",
        required=True,
        help=(
            "Ordered list of donor-level column names from adata.obs to write into "
            "the covariate file. E.g.: NWD_ID AgeAtBloodDraw INFERRED_SEX PC1 ... "
            "PC10 PC1_pseudobulk ... PC5_pseudobulk Batch_Biobank"
        ),
    )
    parser.add_argument(
        "--additional_covariates",
        nargs="*",
        default=[],
        help=(
            "Cell-level column names from adata.obs for which the per-donor mean "
            "is computed and written as 'mean_<col>' covariate rows. "
            "E.g.: percent.mt nCount_RNA  (default: none)"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------
def parse_gtf(gtf_path):
    """
    Parse a gzipped GTF and return a DataFrame with columns:
        gene_name, gene_id, chr, start, end
    Only 'gene' rows on standard chromosomes (1-22, X, Y) are kept.
    The 'chr' prefix is stripped; version suffix is stripped from gene_id.
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
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
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

    # -----------------------------------------------------------------------
    # Validate --processed_files is non-empty
    # -----------------------------------------------------------------------
    if not args.processed_files:
        log.error("--processed_files is empty. Nothing to process.")
        sys.exit(1)
    log.info("Received %d candidate processed files.", len(args.processed_files))

    # -----------------------------------------------------------------------
    # Validate each path; warn and skip missing files
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # STEP 1: Load and concatenate .h5ad files
    # -----------------------------------------------------------------------
    try:
        log.info("Loading .h5ad files...")
        adata_list = []
        for path in valid_files:
            log.info("  Loading: %s", path)
            adata_list.append(ad.read_h5ad(path))
        log.info("Loaded %d AnnData objects.", len(adata_list))

        log.info("Concatenating AnnData objects (axis=0, join='outer')...")
        adata = ad.concat(adata_list, axis=0, join="outer")
        adata.var_names_make_unique()
        log.info(
            "Concatenated shape: %s  (%d donors)",
            adata.shape,
            adata.obs[args.donor_key].nunique(),
        )

        # Verify required obs column exists
        if args.donor_key not in adata.obs.columns:
            raise KeyError(
                f"Column '{args.donor_key}' not found in adata.obs. "
                f"Available columns: {adata.obs.columns.tolist()}"
            )

        # ------------------------------------------------------------------
        # Resolve count matrix (prefer adata.raw)
        # ------------------------------------------------------------------
        if adata.raw is not None:
            log.info("Using adata.raw.X for count matrix.")
            count_matrix = adata.raw.X
            gene_names = adata.raw.var_names.tolist()
        else:
            log.warning(
                "adata.raw not found — falling back to adata.X. "
                "Ensure adata.X contains raw counts, not normalised values."
            )
            count_matrix = adata.X
            gene_names = adata.var_names.tolist()

        n_cells, n_genes = count_matrix.shape
        log.info("Count matrix: %d cells × %d genes", n_cells, n_genes)

        # Ensure CSR sparse format
        if not sp.issparse(count_matrix):
            count_matrix = sp.csr_matrix(count_matrix)
        else:
            count_matrix = count_matrix.tocsr()

        # ------------------------------------------------------------------
        # STEP 2: Per-cell CP10K normalization
        # ------------------------------------------------------------------
        cp10k_target = 1e4
        log.info("Normalizing each cell to CP%d...", int(cp10k_target))

        cell_totals = np.asarray(count_matrix.sum(axis=1)).ravel()
        cell_totals_safe = np.where(cell_totals == 0, 1, cell_totals)
        scaling_factors = cp10k_target / cell_totals_safe
        diag_scaler = sp.diags(scaling_factors)
        cp10k_matrix = diag_scaler.dot(count_matrix)   # sparse, (n_cells, n_genes)
        log.info("CP10K normalization complete.")

        # ------------------------------------------------------------------
        # STEP 3: Build donor-level pseudobulk sample index
        # ------------------------------------------------------------------
        obs = adata.obs[[args.donor_key]].copy()
        obs["_sample_id"] = obs[args.donor_key].astype(str)
        sample_ids = obs["_sample_id"].values
        unique_samples = obs["_sample_id"].unique()
        cell_counts = obs["_sample_id"].value_counts()

        log.info(
            "Unique donors before cell-count filter: %d", len(unique_samples)
        )

        # ------------------------------------------------------------------
        # STEP 4: Filter donors by minimum cell count
        # ------------------------------------------------------------------
        passing_mask = cell_counts >= args.min_cells_per_sample
        passing_samples = cell_counts[passing_mask].index.tolist()
        excluded_samples = cell_counts[~passing_mask]
        n_excluded = int((~passing_mask).sum())

        log.info(
            "Sample filtering (min %d cells): retained=%d  excluded=%d",
            args.min_cells_per_sample,
            len(passing_samples),
            n_excluded,
        )
        if n_excluded > 0:
            log.info(
                "Excluded donors: %s",
                excluded_samples.index.tolist(),
            )

        if len(passing_samples) == 0:
            raise ValueError(
                "No donors passed the minimum cell filter. "
                "Lower --min_cells_per_sample or check your data."
            )

        # ------------------------------------------------------------------
        # STEP 5: Average CP10K per donor → pseudobulk matrix
        # ------------------------------------------------------------------
        log.info(
            "Building pseudobulk matrix for %d donors...", len(passing_samples)
        )
        pseudobulk_rows = []
        sample_meta_list = []

        for donor in passing_samples:
            cell_idx = np.where(sample_ids == donor)[0]
            sample_cp10k = cp10k_matrix[cell_idx, :]          # sparse subset
            avg_cp10k = np.asarray(sample_cp10k.mean(axis=0)).ravel()
            pseudobulk_rows.append(avg_cp10k)
            sample_meta_list.append(
                {
                    "sample_id": donor,
                    args.donor_key: donor,
                    "cell_type": args.celltype_label,
                    "n_cells": int(cell_counts[donor]),
                }
            )

        pseudobulk_array = np.vstack(pseudobulk_rows)   # (n_donors, n_genes)
        log.info("Pseudobulk array shape: %s", pseudobulk_array.shape)

        # ------------------------------------------------------------------
        # STEP 6: Gene filter
        # ------------------------------------------------------------------
        n_samples_pb = pseudobulk_array.shape[0]
        expressed = (pseudobulk_array > 0).sum(axis=0)
        expr_frac = expressed / n_samples_pb
        gene_mask = expr_frac >= args.min_sample_pct

        log.info(
            "Gene filtering (expressed in ≥%.0f%% of donors): before=%d  after=%d",
            args.min_sample_pct * 100,
            n_genes,
            int(gene_mask.sum()),
        )

        pseudobulk_array = pseudobulk_array[:, gene_mask]
        retained_genes = [g for g, keep in zip(gene_names, gene_mask) if keep]

        # ------------------------------------------------------------------
        # STEP 7: log2(x + 1) transform
        # ------------------------------------------------------------------
        log.info("Applying log2(x + 1) transform...")
        pseudobulk_log = np.log2(pseudobulk_array + 1)
        log.info("log2 transform complete.")

        # ------------------------------------------------------------------
        # STEP 8: Pseudobulk PCA (covariates)
        # ------------------------------------------------------------------
        n_pcs = args.n_pseudobulk_pcs
        log.info("Computing top %d pseudobulk PCs...", n_pcs)

        scaler = StandardScaler()
        pb_scaled = scaler.fit_transform(pseudobulk_log)

        pca = PCA(n_components=n_pcs, random_state=42)
        pb_pcs = pca.fit_transform(pb_scaled)               # (n_donors, n_pcs)
        pc_cols = [f"PC{i + 1}_pseudobulk" for i in range(n_pcs)]
        pc_df = pd.DataFrame(pb_pcs, columns=pc_cols)
        var_exp = pca.explained_variance_ratio_ * 100

        for col, ve in zip(pc_cols, var_exp):
            log.info("  %s: %.2f%% variance explained", col, ve)
        log.info("Pseudobulk PCs computed.")

        # ------------------------------------------------------------------
        # STEP 9: Inverse-rank normalization (IRN) per gene
        # ------------------------------------------------------------------
        log.info("Applying inverse-rank normalization (per gene)...")
        n_s = pseudobulk_log.shape[0]

        pseudobulk_irn = np.apply_along_axis(
            lambda col: ndtri((rankdata(col) - 0.5) / n_s),
            axis=0,
            arr=pseudobulk_log,
        )
        log.info("Inverse-rank normalization complete.")

        # ------------------------------------------------------------------
        # STEP 10: Assemble sample metadata
        # ------------------------------------------------------------------
        sample_meta_df = pd.DataFrame(sample_meta_list).set_index("sample_id")

        # Attach pseudobulk PCs
        sample_meta_df = pd.concat(
            [sample_meta_df, pc_df.set_index(sample_meta_df.index)],
            axis=1,
        )

        # Donor-level covariates: take first-occurrence row per donor from adata.obs
        if args.covariate_cols:
            # Validate columns exist before attempting extraction
            missing_donor_cols = [
                c for c in args.covariate_cols
                if c not in adata.obs.columns
                # Pseudobulk PC cols don't exist in adata.obs — they come from pc_df
                and c not in pc_cols
            ]
            if missing_donor_cols:
                raise KeyError(
                    f"The following --covariate_cols are not in adata.obs or "
                    f"pseudobulk PC columns: {missing_donor_cols}\n"
                    f"adata.obs columns: {adata.obs.columns.tolist()}"
                )

            # Columns that live in adata.obs (exclude PC cols already added)
            obs_cov_cols = [c for c in args.covariate_cols if c not in pc_cols]

            if obs_cov_cols:
                # Filter obs to passing donors only
                passing_obs = adata.obs[
                    adata.obs[args.donor_key].astype(str).isin(passing_samples)
                ].copy()
                passing_obs["_sample_id"] = passing_obs[args.donor_key].astype(str)

                donor_first = (
                    passing_obs.groupby("_sample_id")[obs_cov_cols]
                    .first()
                )
                sample_meta_df = sample_meta_df.join(donor_first, how="left")

        # Cell-level mean covariates
        if args.additional_covariates:
            missing_add = [
                c for c in args.additional_covariates
                if c not in adata.obs.columns
            ]
            if missing_add:
                raise KeyError(
                    f"The following --additional_covariates are not in adata.obs: "
                    f"{missing_add}\n"
                    f"adata.obs columns: {adata.obs.columns.tolist()}"
                )

            passing_obs_add = adata.obs[
                adata.obs[args.donor_key].astype(str).isin(passing_samples)
            ].copy()
            passing_obs_add["_sample_id"] = (
                passing_obs_add[args.donor_key].astype(str)
            )

            for col in args.additional_covariates:
                mean_col = f"mean_{col}"
                donor_means = (
                    passing_obs_add.groupby("_sample_id")[col].mean().rename(mean_col)
                )
                sample_meta_df = sample_meta_df.join(donor_means, how="left")
                log.info("  Computed %s for %d donors.", mean_col, len(donor_means))

        log.info(
            "sample_metadata assembled: %d donors × %d columns",
            *sample_meta_df.shape,
        )

        # Final IRN pseudobulk DataFrame (donors × retained genes)
        pseudobulk_df = pd.DataFrame(
            pseudobulk_irn,
            index=sample_meta_df.index,
            columns=retained_genes,
        )

        log.info(
            "Final pseudobulk matrix: %d donors × %d genes",
            *pseudobulk_df.shape,
        )

    except Exception as exc:
        log.exception("Fatal error during pseudobulk processing: %s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 11: Parse GTF
    # -----------------------------------------------------------------------
    try:
        gene_df = parse_gtf(args.gtf_path)
    except Exception as exc:
        log.exception("Fatal error parsing GTF: %s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 12: Load cis-window table
    # -----------------------------------------------------------------------
    try:
        log.info("Loading cis-window table: %s", args.cis_window_path)
        cis_df = pd.read_csv(args.cis_window_path, sep="\t")
        cis_df["chr"] = (
            cis_df["chr"].astype(str).str.replace("chr", "", regex=False)
        )

        n_before = len(cis_df)
        cis_df = cis_df.drop_duplicates(subset="gene_name", keep="first")
        n_after = len(cis_df)

        if n_before != n_after:
            log.warning(
                "Dropped %d duplicate gene_name entries from cis-window table "
                "(%d → %d genes).",
                n_before - n_after,
                n_before,
                n_after,
            )

        cis_lookup = cis_df.set_index("gene_name")
        log.info("Loaded %d unique genes from cis-window table.", n_after)
    except Exception as exc:
        log.exception("Fatal error loading cis-window table: %s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STEP 13 & 14: Write per-chromosome phenotype BED files and
    #               covariate files
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    log.info("Output directory: %s", args.output_dir)

    sample_ids_ordered = pseudobulk_df.index.tolist()

    # Build ordered list of covariate rows for the covariate file
    # Order: --covariate_cols rows, then mean_<col> rows
    mean_cov_cols = [f"mean_{c}" for c in (args.additional_covariates or [])]
    all_covariate_rows = list(args.covariate_cols) + mean_cov_cols

    # Validate all covariate rows are present in sample_metadata
    missing_cov_rows = [
        c for c in all_covariate_rows if c not in sample_meta_df.columns
    ]
    if missing_cov_rows:
        log.error(
            "The following covariate rows are missing from sample_metadata: %s",
            missing_cov_rows,
        )
        sys.exit(1)

    chromosomes = sorted(
        gene_df["chr"].unique(),
        key=lambda x: (int(x) if x.isdigit() else float("inf"), x),
    )
    log.info("Chromosomes to process: %s", chromosomes)

    files_written = []
    cov_files_written = []

    for chromosome in chromosomes:
        # ── STEP 13: Phenotype BED file ────────────────────────────────────
        pheno_file = os.path.join(
            args.output_dir,
            f"phenotypeFile_{args.basename}_chr{chromosome}.bed.gz",
        )
        cov_file = os.path.join(
            args.output_dir,
            f"covariateFile_{args.basename}_chr{chromosome}.tsv.gz",
        )

        try:
            genes_on_chr = gene_df[gene_df["chr"] == chromosome]["gene_name"].values
            genes_in_data = [
                g for g in genes_on_chr
                if g in pseudobulk_df.columns and g in cis_lookup.index
            ]

            if not genes_in_data:
                log.info(
                    "Chr %s: No genes found in pseudobulk data — skipping.",
                    chromosome,
                )
                continue

            log.info(
                "Chr %s: %d genes on chromosome, %d present in pseudobulk data "
                "and cis-window table.",
                chromosome,
                len(genes_on_chr),
                len(genes_in_data),
            )

            # Build BED rows
            bed_rows = []
            for gene in genes_in_data:
                info = cis_lookup.loc[gene]
                row = {
                    "#chr": chromosome,
                    "start": int(info["gene_start"]),
                    "end": int(info["gene_end"]),
                    "phenotype_id": gene,
                }
                for sid in sample_ids_ordered:
                    row[sid] = pseudobulk_df.loc[sid, gene]
                bed_rows.append(row)

            bed_df = pd.DataFrame(bed_rows)
            bed_df = bed_df.sort_values("start").reset_index(drop=True)

            # Validate: no NaN in expression columns
            nan_count = bed_df[sample_ids_ordered].isna().sum().sum()
            if nan_count > 0:
                raise RuntimeError(
                    f"Chr {chromosome}: {nan_count} NaN values found in "
                    "expression columns of BED output."
                )

            bed_df.to_csv(pheno_file, sep="\t", index=False, compression="gzip")
            files_written.append(pheno_file)
            log.info(
                "Chr %s: wrote phenotype BED %s | shape=%s",
                chromosome,
                pheno_file,
                bed_df.shape,
            )

            # ── STEP 14: Covariate file ────────────────────────────────────
            if all_covariate_rows:
                missing_sids = [
                    s for s in sample_ids_ordered
                    if s not in sample_meta_df.index
                ]
                if missing_sids:
                    raise KeyError(
                        f"Chr {chromosome}: {len(missing_sids)} donors missing "
                        f"from sample_metadata. Examples: {missing_sids[:5]}"
                    )

                cov_meta = sample_meta_df.loc[
                    sample_ids_ordered, all_covariate_rows
                ]
                cov_T = cov_meta.T.copy()
                cov_T.columns = sample_ids_ordered
                cov_T.index.name = "covariate_id"

                cov_T.to_csv(
                    cov_file, sep="\t", index=True, compression="gzip"
                )
                cov_files_written.append(cov_file)
                log.info(
                    "Chr %s: wrote covariate file %s | shape=%s "
                    "(%d covariates × %d donors)",
                    chromosome,
                    cov_file,
                    cov_T.shape,
                    cov_T.shape[0],
                    cov_T.shape[1],
                )

        except Exception as exc:
            log.exception(
                "Error processing chromosome %s — skipping: %s", chromosome, exc
            )
            continue

    log.info("Wrote %d phenotype BED files.", len(files_written))
    log.info("Wrote %d covariate files.", len(cov_files_written))
    log.info("All chromosomes processed. Exiting successfully.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()