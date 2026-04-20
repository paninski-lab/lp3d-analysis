"""
Filter CollectedData by reprojection error — cheese-3d.

For each selected frame and each view:
  1. Load ensemble median predictions (x, y, likelihood) from source EKS CSVs
  2. Compare against per-view reprojected labels in CollectedData CSVs
  3. Compute mean reprojection error per frame over confident-view keypoints
  4. Remove the worst `remove_pct`% frames (default 25%)

Outputs to <output_dir>/clean_collected_data/:
  - CollectedData_{view}.csv           (filtered InD)
  - CollectedData_{view}_new.csv       (filtered OOD)
  - calibrations.csv                   (filtered InD mapping)
  - calibrations_new.csv               (filtered OOD mapping)

Reports per-session stats before and after filtering.

Performance: CSVs are cached per (session, view) to avoid repeated disk reads.
The comparison is done per-view: EKS prediction for view V vs reprojected
label in CollectedData_{V}.csv for the same frame.
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lp3d_analysis.io import load_cfgs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_session_view_frame(img_path: str, views: List[str]) -> Tuple[str, str, int]:
    """Extract session, view, frame from e.g.
    'labeled-data/20231031_B6_chew_bl_000_11-35-08_BC/img00001234.png'
    """
    parts = img_path.split('/')
    folder = parts[-2] if len(parts) >= 2 else parts[0]
    fname = parts[-1]
    frame = int(re.search(r'img(\d+)', fname).group(1)) if re.search(r'img(\d+)', fname) else -1

    sorted_views = sorted(views, key=len, reverse=True)
    view = next(
        (v for v in sorted_views if folder.endswith(f'_{v}') or f'_{v}_' in folder),
        folder.split('_')[-1],
    )
    session = folder
    for v in sorted_views:
        session = re.sub(rf'_{re.escape(v)}$', '', session)
    return session, view, frame


def _find_eks_csv(eks_dir: str, session: str, view: str) -> Optional[Path]:
    """Find the EKS CSV for a given session+view.

    CollectedData session names look like:  20231031_B20_chew_bl_000_08-38-25
    EKS CSV names look like:                20231031_B20_chew_bl_000_BC_08-38-25.csv
    (view is inserted before the time suffix HH-MM-SS)

    We try several patterns to be robust.
    """
    candidates = []

    # Primary pattern: strip time suffix (HH-MM-SS) and re-insert view before it
    time_match = re.search(r'_(\d{2}-\d{2}-\d{2})$', session)
    if time_match:
        prefix = session[:time_match.start()]
        time_sfx = time_match.group(1)
        candidates.append(f"{prefix}_{view}_{time_sfx}.csv")

    # Fallback: session_view.csv  (time appended after view, or no time)
    candidates.append(f"{session}_{view}.csv")
    # Fallback: split at last _ → prefix_view_suffix.csv
    parts = session.rsplit('_', 1)
    if len(parts) == 2:
        candidates.append(f"{parts[0]}_{view}_{parts[1]}.csv")

    for c in candidates:
        p = Path(eks_dir) / c
        if p.exists():
            return p
    return None


def _load_eks_csv_cached(
    eks_dir: str,
    session: str,
    view: str,
    cache: Dict,
) -> Optional[Tuple[pd.DataFrame, List[str], str, str]]:
    """Load and cache a single EKS CSV by (session, view).

    Returns (df, kp_names, xy_key_x, xy_key_y) or None if not found.
    The kp_names list and xy keys are pre-computed so per-frame extraction is fast.
    """
    key = (eks_dir, session, view)
    if key in cache:
        return cache[key]

    csv_path = _find_eks_csv(eks_dir, session, view)
    if csv_path is None:
        cache[key] = None
        return None

    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        scorer = df.columns.get_level_values(0)[0]
        kp_names = [c[1] for c in df.columns if c[2] == 'likelihood']

        has_median = any(
            (scorer, kp, 'x_ens_median') in df.columns for kp in kp_names[:1]
        )
        xy_key_x = 'x_ens_median' if has_median else 'x'
        xy_key_y = 'y_ens_median' if has_median else 'y'

        result = (df, kp_names, xy_key_x, xy_key_y, scorer)
        cache[key] = result
        return result
    except Exception as e:
        logger.debug(f"Failed to load {csv_path}: {e}")
        cache[key] = None
        return None


def _extract_frame(
    cached: Optional[Tuple],
    frame_idx: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Extract (kp_xy, likelihoods, kp_names) for one frame from cached EKS entry.

    Returns (None, None, None) if data unavailable.
    """
    if cached is None:
        return None, None, None
    df, kp_names, xy_key_x, xy_key_y, scorer = cached
    if frame_idx >= len(df):
        return None, None, None

    try:
        row = df.iloc[frame_idx]
        xs = np.array([row[(scorer, kp, xy_key_x)] for kp in kp_names], dtype=float)
        ys = np.array([row[(scorer, kp, xy_key_y)] for kp in kp_names], dtype=float)
        liks = np.array([row[(scorer, kp, 'likelihood')] for kp in kp_names], dtype=float)
        kp_xy = np.stack([xs, ys], axis=-1)  # (n_kp, 2)
        return kp_xy, liks, kp_names
    except Exception as e:
        logger.debug(f"Failed to extract frame {frame_idx}: {e}")
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Per-suffix filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_suffix(
    output_dir: Path,
    clean_dir: Path,
    eks_dir: str,
    views: List[str],
    suffix: str,
    remove_pct: float = 0.25,
    likelihood_threshold: float = 0.5,
) -> None:
    """Filter one set of CollectedData (InD or OOD) by reprojection error."""
    label = "OOD" if suffix else "InD"
    first_csv = output_dir / f"CollectedData_{views[0]}{suffix}.csv"
    if not first_csv.exists():
        logger.warning(f"[{label}] {first_csv} not found — skipping")
        return

    logger.info(f"[{label}] Loading CollectedData CSVs...")

    # Load all view CSVs into flat DataFrames (index = image path, cols = {kp}_{x/y/likelihood})
    view_dfs: Dict[str, pd.DataFrame] = {}
    for v in views:
        p = output_dir / f"CollectedData_{v}{suffix}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, header=[0, 1, 2], index_col=0)
        flat = df.copy()
        flat.columns = [f"{c[1]}_{c[2]}" for c in df.columns]
        view_dfs[v] = flat

    if not view_dfs:
        logger.warning(f"[{label}] No view CSVs found")
        return

    img_paths = list(next(iter(view_dfs.values())).index)
    n_frames = len(img_paths)
    logger.info(f"[{label}] {n_frames} frames, {len(view_dfs)} views")

    # Pre-load all unique (session, view) EKS CSVs
    logger.info(f"[{label}] Pre-loading EKS CSVs...")
    eks_cache: Dict = {}
    unique_sessions = set()
    for img_path in img_paths:
        session, _, _ = _parse_session_view_frame(img_path, views)
        unique_sessions.add(session)

    n_found = 0
    for session in sorted(unique_sessions):
        for v in views:
            result = _load_eks_csv_cached(eks_dir, session, v, eks_cache)
            if result is not None:
                df, kp_names, _, _, _ = result
                logger.info(f"[{label}]   {session}_{v}: {len(df)} rows, {len(kp_names)} kps")
                n_found += 1
            else:
                logger.warning(f"[{label}]   {session}_{v}: NOT FOUND")
    logger.info(f"[{label}] Loaded {n_found} EKS CSVs for {len(unique_sessions)} sessions × {len(views)} views")

    # Compute per-view reprojection error for each frame
    # For each (frame, view): compare EKS pred (2D detection) vs CollectedData reprojected label
    logger.info(f"[{label}] Computing reprojection errors (per-view comparison)...")
    reproj_errors = np.full(n_frames, np.nan)

    for fi, img_path in enumerate(img_paths):
        session, _, frame_idx = _parse_session_view_frame(img_path, views)

        errors = []
        for v in views:
            cached = _load_eks_csv_cached(eks_dir, session, v, eks_cache)
            kp_xy, liks, eks_kp_names = _extract_frame(cached, frame_idx)
            if kp_xy is None or liks is None or eks_kp_names is None:
                continue

            # Get per-view reprojected labels from CollectedData_{v}.csv
            if v not in view_dfs:
                continue
            flat_df = view_dfs[v]
            if img_path not in flat_df.index:
                continue
            view_row = flat_df.loc[img_path]

            for ki, kp in enumerate(eks_kp_names):
                lik = float(liks[ki])
                if lik <= likelihood_threshold:
                    continue
                # Reprojected label from CollectedData for this specific view
                rx = view_row.get(f'{kp}_x', np.nan)
                ry = view_row.get(f'{kp}_y', np.nan)
                if isinstance(rx, float) and np.isnan(rx):
                    continue
                if isinstance(ry, float) and np.isnan(ry):
                    continue
                try:
                    rx, ry = float(rx), float(ry)
                except (ValueError, TypeError):
                    continue
                if not (np.isfinite(rx) and np.isfinite(ry)):
                    continue
                ox, oy = kp_xy[ki]
                if not (np.isfinite(ox) and np.isfinite(oy)):
                    continue
                errors.append(np.sqrt((ox - rx) ** 2 + (oy - ry) ** 2))

        reproj_errors[fi] = float(np.mean(errors)) if errors else np.nan

        if (fi + 1) % 50 == 0:
            valid_so_far = (~np.isnan(reproj_errors[:fi+1])).sum()
            logger.info(f"[{label}]   {fi + 1}/{n_frames} processed, {valid_so_far} with valid error")

    # Stats before filtering
    valid_mask = ~np.isnan(reproj_errors)
    valid_errs = reproj_errors[valid_mask]
    n_valid = len(valid_errs)
    logger.info(f"\n[{label}] Reprojection error BEFORE filtering ({n_valid}/{n_frames} frames with valid error):")
    if n_valid > 0:
        logger.info(f"  mean:   {np.mean(valid_errs):.2f} px")
        logger.info(f"  median: {np.median(valid_errs):.2f} px")
        logger.info(f"  std:    {np.std(valid_errs):.2f} px")
        logger.info(f"  p75:    {np.percentile(valid_errs, 75):.2f} px")
        logger.info(f"  p90:    {np.percentile(valid_errs, 90):.2f} px")
        logger.info(f"  max:    {np.max(valid_errs):.2f} px")
    else:
        logger.warning(f"[{label}] No valid reprojection errors computed — check EKS dir and frame indices")

    # Session breakdown before
    session_frames: Dict[str, List[int]] = {}
    for fi, p in enumerate(img_paths):
        sess, _, _ = _parse_session_view_frame(p, views)
        session_frames.setdefault(sess, []).append(fi)
    logger.info(f"\n[{label}] Frames per session (before):")
    for sess in sorted(session_frames):
        indices = session_frames[sess]
        errs = reproj_errors[indices]
        n_sess_valid = (~np.isnan(errs)).sum()
        err_str = f"error mean={np.nanmean(errs):.2f}px" if n_sess_valid > 0 else "no valid errors"
        logger.info(f"  {sess}: {len(indices)} frames, {err_str}")

    # Determine which frames to remove
    n_remove = int(np.round(n_frames * remove_pct))
    if n_valid == 0:
        logger.warning(f"[{label}] No valid reprojection errors — keeping all frames")
        keep_mask = np.ones(n_frames, dtype=bool)
    else:
        # np.argsort with NaN: NaN goes last in ascending sort, so first in descending → remove first
        # We want to remove the WORST (highest error) frames
        # NaN frames have no error info → keep them (put at end of remove list)
        nan_safe = np.where(np.isnan(reproj_errors), -np.inf, reproj_errors)
        sort_order = np.argsort(nan_safe)[::-1]  # highest error first; NaN (-inf) last
        remove_set = set(sort_order[:n_remove].tolist())
        keep_mask = np.array([fi not in remove_set for fi in range(n_frames)])

    n_kept = int(keep_mask.sum())
    logger.info(f"\n[{label}] Removing {n_remove} worst frames ({remove_pct:.0%}), keeping {n_kept}")

    # Stats after
    kept_errs = reproj_errors[keep_mask]
    kept_valid = kept_errs[~np.isnan(kept_errs)]
    if len(kept_valid) > 0:
        logger.info(f"[{label}] Reprojection error AFTER filtering ({n_kept} frames):")
        logger.info(f"  mean:   {np.mean(kept_valid):.2f} px")
        logger.info(f"  median: {np.median(kept_valid):.2f} px")
        logger.info(f"  std:    {np.std(kept_valid):.2f} px")
        logger.info(f"  p90:    {np.percentile(kept_valid, 90):.2f} px")

    # Session breakdown after
    logger.info(f"\n[{label}] Frames per session (after removing worst {remove_pct:.0%}):")
    for sess in sorted(session_frames):
        indices = session_frames[sess]
        kept_in_sess = sum(1 for fi in indices if keep_mask[fi])
        logger.info(f"  {sess}: {kept_in_sess} / {len(indices)} kept")

    # Write filtered CSVs
    # All view CSVs are in the same row order (same logical frames), so use
    # keep_mask (boolean by row position) rather than path-string matching.
    keep_indices = np.where(keep_mask)[0]
    clean_dir.mkdir(parents=True, exist_ok=True)

    for v in views:
        p = output_dir / f"CollectedData_{v}{suffix}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, header=[0, 1, 2], index_col=0)
        if len(df) == n_frames:
            filtered = df.iloc[keep_indices]
        else:
            # Different length: fall back to path matching with view substitution
            logger.warning(f"[{label}] {v} CSV has {len(df)} rows (expected {n_frames}) — using path-match fallback")
            kept_paths = set(img_paths[fi] for fi in keep_indices)
            filtered = df.loc[[idx for idx in df.index if idx in kept_paths]]
        out_path = clean_dir / f"CollectedData_{v}{suffix}.csv"
        filtered.to_csv(out_path)
        logger.info(f"[{label}] Wrote {len(filtered)} rows → {out_path.name}")

    # Update calibrations CSV (keyed on img_paths which are BC-view paths)
    calib_in = output_dir / f"calibrations{suffix}.csv"
    if calib_in.exists():
        # Read preserving the original index (first column has no header name)
        calib_df = pd.read_csv(calib_in, index_col=0)
        if len(calib_df) == n_frames:
            calib_filt = calib_df.iloc[keep_indices]
        else:
            # Fall back to path-index matching
            kept_paths_set = set(img_paths[fi] for fi in keep_indices)
            calib_filt = calib_df.loc[calib_df.index.isin(kept_paths_set)]
        calib_out = clean_dir / f"calibrations{suffix}.csv"
        calib_filt.to_csv(calib_out)  # saves index back → preserves empty column name
        logger.info(f"[{label}] Calibrations: {len(calib_filt)}/{len(calib_df)} mappings → {calib_out.name}")
    else:
        logger.warning(f"[{label}] {calib_in} not found — skipping calibrations update")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter cheese-3d by reprojection error")
    parser.add_argument("--config", required=True)
    parser.add_argument("--remove-pct", type=float, default=0.25, help="Fraction to remove (default 0.25)")
    parser.add_argument("--likelihood-threshold", type=float, default=0.5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    cfg_pipe, cfg_lp = load_cfgs(args.config)
    views = cfg_lp['data']['view_names']
    dist = cfg_pipe.get('distillation', {})

    dataset_name = cfg_pipe.get('dataset_name', 'cheese-3d')
    base = cfg_pipe.get('pseudo_labeled_base_dir',
                        '/teamspace/studios/this_studio/pseudo_labeled_dataset_cheese-3d_10k_new')
    output_dir = Path(base) / dataset_name
    clean_dir = output_dir / "clean_collected_data"

    ind_eks = dist.get('ind', {}).get('eks_results_dir', '')
    ood_eks = dist.get('ood', {}).get('eks_results_dir', '')

    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Clean dir:   {clean_dir}")
    logger.info(f"Views:       {views}")
    logger.info(f"Remove pct:  {args.remove_pct:.0%}")
    logger.info(f"InD EKS dir: {ind_eks}")
    logger.info(f"OOD EKS dir: {ood_eks}")

    # InD
    filter_suffix(
        output_dir=output_dir,
        clean_dir=clean_dir,
        eks_dir=ind_eks,
        views=views,
        suffix="",
        remove_pct=args.remove_pct,
        likelihood_threshold=args.likelihood_threshold,
    )

    # OOD
    filter_suffix(
        output_dir=output_dir,
        clean_dir=clean_dir,
        eks_dir=ood_eks,
        views=views,
        suffix="_new",
        remove_pct=args.remove_pct,
        likelihood_threshold=args.likelihood_threshold,
    )

    logger.info("\nDone. Clean data at: " + str(clean_dir))


if __name__ == "__main__":
    main()
