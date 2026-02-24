#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:23:06 2025

@author: wuw5
"""

import numpy as np
import pandas as pd


def classify_group(
    group,
    identity_col="identity",
    global_col="global",
    global_threshold=50,
    subset_cell_type="Bcell",
):
    """
    Classify rows for a given 1st_sequence_id group.
    Implements proper 'unique_within_fna' logic for Bcells.
    Uses group-level classification for cross_donor_fna and cross_epitope_fna,
    including proper handling when both epitope values are missing or empty.
    """

    g = group.copy()

    # Ensure required columns exist
    for col in ["unique_within_sample_type", "cross_donor_sample_type", "cross_epitope_sample_type"]:
        if col not in g.columns:
            g[col] = pd.Series("", index=g.index, dtype="string")
        else:
            g[col] = g[col].astype("string")

    # ---- Base masks -------------------------------------------------------
    mask_100 = g[identity_col] >= 99
    mask_global = g[global_col] > global_threshold
    mask_100_global = mask_100 & mask_global

    same_well = g["1st_sequence_or_well_id"] == g["2nd_sequence_or_well_id"]
    same_donor = g["1st_real_subjectid"] == g["2nd_real_subjectid"]

    # ---- Step 1️⃣ Remove lower‑read duplicates within same well -----------
    removed_rows = 0
    mask_within = mask_100_global & same_well
    if mask_within.any():
        to_drop = []
        for well, subset in g.loc[mask_within].groupby("1st_sequence_or_well_id", group_keys=False):
            if subset.shape[0] > 1:
                maxr = subset["1st_mixcr_read"].max()
                to_drop.extend(subset.index[subset["1st_mixcr_read"] < maxr])
        if to_drop:
            removed_rows = len(to_drop)
            g = g.drop(index=to_drop)
            mask_100_global = mask_100_global.loc[g.index]

    # ---- Step 2️⃣ Unique within FNA (Bcell‑specific) ---------------------
    mask_first_is_B = g["1st_cell_type"] == subset_cell_type
    mask_B_B = (
        (g["1st_cell_type"] == subset_cell_type)
        & (g["2nd_cell_type"] == subset_cell_type)
    )
    mask_identical_BB = mask_B_B & (g[identity_col] == 100) & (g[global_col] > global_threshold)
    has_BB_identical = mask_identical_BB.any()

    if mask_first_is_B.any():
        if has_BB_identical:
            g.loc[mask_first_is_B, "unique_within_fna"] = "no"
        else:
            g.loc[mask_first_is_B, "unique_within_fna"] = "yes"

    # ---- Step 3️⃣ Cross‑donor (group‑level logic) ------------------------
    mask_donor_compare = mask_100_global
    if mask_donor_compare.any():
        same_donor_mask = (
            g.loc[mask_donor_compare, "1st_real_subjectid"]
            == g.loc[mask_donor_compare, "2nd_real_subjectid"]
        )
        all_same_donor = same_donor_mask.all()
        any_same_donor = same_donor_mask.any()

        if all_same_donor:
            g.loc[mask_donor_compare, "cross_donor_fna"] = "no"
        elif any_same_donor:
            g.loc[mask_donor_compare, "cross_donor_fna"] = "yes and no"
        else:
            g.loc[mask_donor_compare, "cross_donor_fna"] = "yes"

    # ---- Step 4️⃣ Cross‑epitope (simplified, reliable logic) ----------------------

    num_both_empty_same = 0
    mask_epitope_compare = mask_100_global

    if mask_epitope_compare.any():
        epi1_series = (
            g.loc[mask_epitope_compare, "1st_flowindex_epitope"]
            .astype("string")
            .str.strip()
            .replace({"": "empty", pd.NA: "empty", None: "empty"})
        )
        epi2_series = (
            g.loc[mask_epitope_compare, "2nd_flowindex_epitope"]
            .astype("string")
            .str.strip()
            .replace({"": "empty", pd.NA: "empty", None: "empty"})
        )

        # There should be only one unique 1st epitope per 1st_sequence_id group
        unique_epi1 = epi1_series.unique()
        if len(unique_epi1) > 1:
            print(
                f"⚠️ Warning: Multiple distinct 1st_flowindex_epitope values found "
                f"for 1st_sequence_id={g['1st_sequence_id'].iloc[0]!r}: {unique_epi1}. "
                f"Using first value."
            )
        epi1_value = unique_epi1[0]

        # Compare each 2nd epitope to 1st epitope
        same_mask = epi2_series == epi1_value
        all_same = same_mask.all()
        any_same = same_mask.any()
        none_same = not any_same

        if all_same:
            g.loc[mask_epitope_compare, "cross_epitope_fna"] = "no"
            if epi1_value == "empty":
                num_both_empty_same = 1  # diagnostic count
        elif any_same:
            g.loc[mask_epitope_compare, "cross_epitope_fna"] = "yes and no"
        elif none_same:
            g.loc[mask_epitope_compare, "cross_epitope_fna"] = "yes"      
            
    return g, removed_rows, num_both_empty_same
# ==============================================================
def classify_all_sequences(
    df,
    groupby_col="1st_sequence_id",
    identity_col="identity",
    global_col="global",
    subset_cell_type="Bcell",
    global_threshold=50,
    cell_type_col_1="1st_cell_type",
):
    """
    Run classification group‑by‑group, rename all new columns with '_<subset_cell_type>' suffix.
    Adds diagnostic summaries including count of 'same epitope' due to both NA/empty values.
    """

    df = df.copy()
    total_removed = 0
    total_empty_epi = 0
    new_cols = ["unique_within_fna", "cross_donor_fna", "cross_epitope_fna"]

    print(f"🔬 Evaluating groups where 1st_cell_type='{subset_cell_type}'")
    mask = df[cell_type_col_1] == subset_cell_type
    work = df.copy()

    # --- process per 1st_sequence_id group -------------------------------
    def run(g):
        nonlocal total_removed, total_empty_epi
        up, rem, naepi = classify_group(
            g,
            identity_col=identity_col,
            global_col=global_col,
            global_threshold=global_threshold,
            subset_cell_type=subset_cell_type,
        )
        total_removed += rem
        total_empty_epi += naepi
        return up

    classified = work.groupby(groupby_col, group_keys=False).apply(run)
    classified = classified.loc[work.index]

    # --- Rename new columns with suffix ----------------------------------
    suffix = f"_{subset_cell_type}" if subset_cell_type else "_all"
    renamed_cols = {c: f"{c}{suffix}" for c in new_cols}
    classified = classified.rename(columns=renamed_cols)

    # --- Merge back into df (safe, avoid duplicates) --------------------
    for col in renamed_cols.values():
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
        df.loc[mask, col] = classified.loc[mask, col]

    # --- Summaries -------------------------------------------------------
    summary = df.loc[mask].groupby(groupby_col)[list(renamed_cols.values())].first()

    print("\n──────────── Summary (by 1st_sequence_id) ────────────")
    for col in summary.columns:
        vc = summary[col].value_counts()
        for label in ["yes", "no", "yes and no"]:
            if label in vc:
                print(f"{col:<28s} {label:<10s} → {vc[label]}")
        print()
    print("───────────────────────────────────────────────────────")

    print("\n────────── Step 1 Summary ──────────")
    print(
        f"Sequences removed (identity==100 & {global_col}>{global_threshold}, same well, lower read): {total_removed}"
    )
    print(f"Remaining rows: {df.shape[0]}")
    print(f"Processed subset rows (1st_cell_type=='{subset_cell_type}'): {mask.sum()}")
    print(f"1st_sequence_id groups classified as 'no' due to both epitopes empty: {total_empty_epi}")
    print("────────────────────────────────────\n")

    return df
