"""
Cheese-3d Distillation Pipeline

Extends the base distillation pipeline with:
  - Ensemble median (x_ens_median / y_ens_median) for triangulation
  - Per-keypoint likelihood gating: keep only frames where every keypoint
    has >= min_confident_views views with likelihood > threshold
  - Per-keypoint triangulation + reprojection for label generation:
    triangulate each keypoint using only its confident views, then
    reproject to all views for geometrically consistent 2D labels
  - CSV / calibrations file suffix support (_new for OOD)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from lp3d_analysis.distillation import (
    DistillationConfig,
    DistillationPipeline,
    logger,
)
from lp3d_analysis.utils import (
    remap_keypoints_to_original_space,
    create_image_path,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Cheese3dDistillationConfig(DistillationConfig):
    """DistillationConfig with cheese-3d-specific extensions."""

    # CSV naming: "" → CollectedData_BC.csv,  "_new" → CollectedData_BC_new.csv
    csv_file_suffix: str = ""

    # Likelihood-based pre-filter
    likelihood_threshold: float = 0.5
    min_confident_views: int = 2


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Cheese3dDistillationPipeline(DistillationPipeline):
    """DistillationPipeline with cheese-3d-specific overrides."""

    # -- CSV processing: add likelihoods + ensemble median -----------------

    def _process_csv_vectorized(self, df, keypoint_names, bbox_data, uncropped_df):
        """Override: also extract likelihoods and ensemble median."""

        # Call the parent to get the base predictions
        predictions = super()._process_csv_vectorized(
            df, keypoint_names, bbox_data, uncropped_df
        )

        scorer = df.columns.levels[0][0]
        n_frames = len(df)
        n_keypoints = len(keypoint_names)

        # Extract likelihoods
        try:
            likelihoods = np.column_stack(
                [df[(scorer, kp, 'likelihood')].values for kp in keypoint_names]
            )
        except KeyError:
            likelihoods = np.ones((n_frames, n_keypoints), dtype=float)

        # Extract ensemble median
        try:
            median_x = np.column_stack(
                [df[(scorer, kp, 'x_ens_median')].values for kp in keypoint_names]
            )
            median_y = np.column_stack(
                [df[(scorer, kp, 'y_ens_median')].values for kp in keypoint_names]
            )
            kp_median = np.stack([median_x, median_y], axis=-1)
        except KeyError:
            # Fallback: use EKS-smoothed x/y
            keypoints_x = np.column_stack(
                [df[(scorer, kp, 'x')].values for kp in keypoint_names]
            )
            keypoints_y = np.column_stack(
                [df[(scorer, kp, 'y')].values for kp in keypoint_names]
            )
            kp_median = np.stack([keypoints_x, keypoints_y], axis=-1)

        # Patch each frame dict
        for i, pred in enumerate(predictions):
            pred['likelihoods'] = likelihoods[i]
            pred['keypoints_2d_median'] = kp_median[i]

        return predictions

    # -- Triangulation for frame selection: use ensemble median ------------

    def _perform_triangulation(self, session_data, camera_group):
        """Override: use ensemble median instead of EKS-smoothed x/y."""

        first_view = next(v for v in self.views if v in session_data)
        num_frames = len(session_data[first_view]['smoothed_predictions'])
        num_keypoints = len(session_data[first_view]['keypoint_names'])

        view_keypoints = {}
        for view_name in self.views:
            if view_name not in session_data or 'smoothed_predictions' not in session_data[view_name]:
                continue
            view_data = session_data[view_name]
            predictions = view_data['smoothed_predictions']
            has_bbox = view_data.get('has_bbox_data', False)
            has_uncropped = view_data.get('has_uncropped_data', False)

            kp_list = []
            for frame_data in predictions:
                bbox = frame_data.get('bbox_data')
                if (bbox is not None and has_bbox and has_uncropped
                        and frame_data.get('keypoints_2d_uncropped') is not None):
                    kp = frame_data['keypoints_2d_uncropped']
                else:
                    # ---- cheese-3d change: ensemble median ----
                    kp = frame_data.get('keypoints_2d_median', frame_data['keypoints_2d'])
                    if bbox is not None and has_bbox:
                        kp = remap_keypoints_to_original_space(kp, bbox)
                kp_list.append(kp)
            view_keypoints[view_name] = np.array(kp_list)

        nan_frame = np.full((num_keypoints, 3), np.nan)
        kp3d_all = []
        for fi in range(num_frames):
            valid = []
            for vn in self.views:
                if vn not in view_keypoints:
                    continue
                if fi >= len(view_keypoints[vn]):
                    continue
                kp = view_keypoints[vn][fi]
                if not (np.any(np.isnan(kp)) or np.any(np.isinf(kp))):
                    valid.append(kp)
            if len(valid) >= 2:
                try:
                    stacked = np.stack(valid)
                    kp3d = camera_group.triangulate_fast(stacked, undistort=True)
                    kp3d_all.append(kp3d)
                except Exception:
                    kp3d_all.append(nan_frame)
            else:
                kp3d_all.append(nan_frame)
        return kp3d_all

    # -- Likelihood filter -------------------------------------------------

    def filter_by_likelihood(self, eks_data: Dict) -> Dict:
        """Keep only frames where every keypoint has >= min_confident_views
        views with likelihood > likelihood_threshold."""

        threshold = self.config.likelihood_threshold
        min_views = self.config.min_confident_views
        total_before = total_after = 0
        filtered_data = {}

        for session_name, session_data in eks_data.items():
            available_views = [
                v for v in self.views
                if v in session_data and 'smoothed_predictions' in session_data[v]
                and len(session_data[v]['smoothed_predictions']) > 0
            ]
            if not available_views:
                filtered_data[session_name] = session_data
                continue

            n_frames = len(session_data[available_views[0]]['smoothed_predictions'])
            sample = session_data[available_views[0]]['smoothed_predictions'][0]
            if 'likelihoods' not in sample:
                logger.warning(f"No likelihood data for {session_name} — skipping filter")
                filtered_data[session_name] = session_data
                continue

            n_kp = len(sample['likelihoods'])
            counts = np.zeros((n_frames, n_kp), dtype=np.int32)
            all_liks = np.zeros((len(available_views), n_frames, n_kp), dtype=float)
            for vi, vn in enumerate(available_views):
                preds = session_data[vn]['smoothed_predictions']
                # Allow views with slightly different frame counts (e.g. ±1)
                # by only processing the overlapping prefix.
                n_use = min(len(preds), n_frames)
                if n_use == 0:
                    continue
                liks = np.stack([p['likelihoods'] for p in preds[:n_use]])
                all_liks[vi, :n_use] = liks
                counts[:n_use] += (liks > threshold).astype(np.int32)

            # A keypoint should only contribute to the filter if at least
            # min_views contributing views can ever predict it confidently.
            # Keypoints visible in < min_views contributing views (due to
            # occlusion or mismatched frame counts) are excluded.
            n_contributing = (all_liks.max(axis=1) >= threshold).sum(axis=0)  # (n_kp,): # views ever confident
            active_kp = n_contributing >= min_views
            if not np.any(active_kp):
                active_kp = np.ones(n_kp, dtype=bool)
            n_inactive = int(np.sum(~active_kp))
            if n_inactive:
                logger.info(
                    f"  {session_name}: {n_inactive} keypoints have < {min_views} "
                    f"confident contributing views → excluded from likelihood filter"
                )

            passing = np.all(counts[:, active_kp] >= min_views, axis=1)
            indices = np.where(passing)[0].tolist()
            total_before += n_frames
            total_after += len(indices)

            filtered_data[session_name] = self._filter_session_flat(
                session_data, indices
            )

        removed = total_before - total_after
        pct = 100 * removed / total_before if total_before else 0
        logger.info(
            f"Likelihood filter (thr={threshold}, min_views={min_views}): "
            f"{total_before} → {total_after} frames ({removed} removed, {pct:.1f}%)"
        )
        return filtered_data

    def _filter_session_flat(self, session_data: Dict, indices: List[int]) -> Dict:
        """Filter session keeping keypoints_3d as a plain list (not wrapped
        in dicts), so downstream _filter_session_by_indices wraps once."""

        filtered = {}
        for vn in self.views:
            if vn in session_data and 'smoothed_predictions' in session_data[vn]:
                vd = session_data[vn].copy()
                preds = vd['smoothed_predictions']
                vd['smoothed_predictions'] = [preds[i] for i in indices if i < len(preds)]
                vd['frame_count'] = len(vd['smoothed_predictions'])
                filtered[vn] = vd

        if 'keypoints_3d' in session_data:
            kp3d = session_data['keypoints_3d']
            filtered['keypoints_3d'] = [kp3d[i] for i in indices if i < len(kp3d)]

        for key in ('camera_group', 'camera_params', 'reconstruction_method'):
            if key in session_data:
                filtered[key] = session_data[key]
        return filtered

    # -- Per-keypoint triangulation + reprojection -------------------------

    def _triangulate_and_reproject_frame(
        self, session_data: Dict, camera_group, frame_idx: int
    ) -> dict:
        """For one frame: triangulate each keypoint from its confident views
        using ensemble median, then reproject to all views.

        Returns dict with:
            'reprojected': dict mapping view_name → (n_keypoints, 2) coords
            'reproj_errors': list of per-keypoint reprojection errors (pixels)
                             across the confident views used for triangulation
        NaN where fewer than min_confident_views had likelihood > threshold.
        """
        threshold = self.config.likelihood_threshold
        min_needed = self.config.min_confident_views

        cam_names = camera_group.get_names()
        v2ci = {name: idx for idx, name in enumerate(cam_names)}
        n_cams = len(cam_names)

        first_view = next(
            v for v in self.views
            if v in session_data and 'smoothed_predictions' in session_data[v]
        )
        n_kp = len(session_data[first_view]['keypoint_names'])
        kp3d = np.full((n_kp, 3), np.nan)

        # Track which cameras + 2D points were used per keypoint (for reproj error)
        kp_confident_info: List[Optional[tuple]] = [None] * n_kp

        for ki in range(n_kp):
            conf_ci, conf_2d = [], []
            for vn in self.views:
                if vn not in session_data or 'smoothed_predictions' not in session_data[vn]:
                    continue
                ci = v2ci.get(vn)
                if ci is None:
                    continue
                preds = session_data[vn]['smoothed_predictions']
                if frame_idx >= len(preds):
                    continue
                fd = preds[frame_idx]
                lik = float(fd['likelihoods'][ki])
                if lik > threshold:
                    kp = fd.get('keypoints_2d_median', fd['keypoints_2d'])
                    bd = fd.get('bbox_data')
                    if bd is not None and session_data[vn].get('has_bbox_data', False):
                        kp = remap_keypoints_to_original_space(kp, bd)
                    x, y = float(kp[ki, 0]), float(kp[ki, 1])
                    if np.isfinite(x) and np.isfinite(y):
                        conf_ci.append(ci)
                        conf_2d.append([x, y])

            if len(conf_ci) >= min_needed:
                try:
                    sub = camera_group.subset_cameras(conf_ci)
                    pts = np.array(conf_2d, dtype=float).reshape(len(conf_ci), 1, 2)
                    p3 = sub.triangulate(pts, undistort=True, fast=True)  # (1, 3)
                    kp3d[ki] = p3[0]
                    kp_confident_info[ki] = (conf_ci, np.array(conf_2d))
                except Exception as e:
                    logger.debug(f"Per-kp tri failed kp={ki}: {e}")

        # Reproject valid 3D → all views
        valid = ~np.any(np.isnan(kp3d), axis=1)
        reproj = np.full((n_cams, n_kp, 2), np.nan)
        if np.any(valid):
            try:
                proj = camera_group.project(kp3d[valid])  # (C, n_valid, 2)
                reproj[:, valid, :] = proj
            except Exception as e:
                logger.debug(f"Reprojection failed: {e}")

        # Compute reprojection error: distance between original 2D (ensemble
        # median) and reprojected 2D in the confident views that were used
        reproj_errors = []
        for ki in range(n_kp):
            if kp_confident_info[ki] is None:
                continue
            ci_list, orig_2d = kp_confident_info[ki]  # cam indices, (n_conf, 2)
            for j, ci in enumerate(ci_list):
                rp = reproj[ci, ki]  # (2,)
                if np.any(np.isnan(rp)):
                    continue
                err = np.sqrt((orig_2d[j, 0] - rp[0])**2 + (orig_2d[j, 1] - rp[1])**2)
                reproj_errors.append(err)

        reprojected = {
            vn: reproj[v2ci[vn]] if vn in v2ci else np.full((n_kp, 2), np.nan)
            for vn in self.views
        }

        return {
            'reprojected': reprojected,
            'reproj_errors': reproj_errors,
        }

    def _compute_reprojected_labels(
        self, filtered_data: Dict, pseudo_data: Dict
    ) -> None:
        """Pre-compute per-keypoint reprojected 2D labels for all selected frames.

        Stores in pseudo_data['selected_frames'][idx]['reprojected']
        as dict(view_name → (n_kp, 2)).
        Also logs reprojection error statistics.
        """
        total = len(pseudo_data['selected_frames'])
        logger.info(f"Computing reprojected labels for {total} selected frames...")

        all_reproj_errors: List[float] = []
        done = 0
        for fi in pseudo_data['selected_frames'].values():
            sn = fi.get('original_session_name', fi['session_name'])
            sd = filtered_data.get(sn, {})
            cg = sd.get('camera_group')
            if cg is None:
                fi['reprojected'] = None
            else:
                result = self._triangulate_and_reproject_frame(
                    sd, cg, fi['filtered_frame_idx']
                )
                fi['reprojected'] = result['reprojected']
                all_reproj_errors.extend(result['reproj_errors'])
            done += 1
            if done % 100 == 0:
                logger.info(f"  Reprojected {done}/{total} frames")

        # Log reprojection error summary
        if all_reproj_errors:
            errs = np.array(all_reproj_errors)
            logger.info(
                f"Reprojection error (across {len(errs)} confident-view measurements):"
            )
            logger.info(f"  mean:   {np.mean(errs):.2f} px")
            logger.info(f"  median: {np.median(errs):.2f} px")
            logger.info(f"  std:    {np.std(errs):.2f} px")
            logger.info(f"  p95:    {np.percentile(errs, 95):.2f} px")
            logger.info(f"  max:    {np.max(errs):.2f} px")
        else:
            logger.warning("No reprojection errors computed (no valid triangulations?)")

        logger.info(f"Reprojection complete: {total} frames processed")

    # -- Label generation: use reprojected coords --------------------------

    def _create_pseudo_rows(self, view_name, pseudo_data, filtered_data):
        """Override: use reprojected labels when available."""

        pseudo_rows = []
        for cluster_idx, frame_info in pseudo_data['selected_frames'].items():
            session_name = frame_info['session_name']
            original_session_name = frame_info.get('original_session_name', session_name)
            filtered_idx = frame_info['filtered_frame_idx']
            original_frame = frame_info.get('original_frame_number',
                                            frame_info.get('original_frame_idx'))
            if original_frame is None:
                continue
            if view_name not in filtered_data.get(original_session_name, {}):
                continue

            view_data = filtered_data[original_session_name][view_name]
            frame_data = view_data['smoothed_predictions'][filtered_idx]

            # Prefer reprojected labels → ensemble median → EKS smoothed
            reprojected = frame_info.get('reprojected')
            if reprojected is not None and view_name in reprojected:
                keypoints_2d = reprojected[view_name]
            else:
                keypoints_2d = frame_data.get(
                    'keypoints_2d_median', frame_data['keypoints_2d']
                )

            img_path = create_image_path(
                session_name, view_name, original_frame, self.views,
                labeled_data_prefix=self.labeled_data_prefix,
                image_ext=self.image_ext,
            )

            row_data = {}
            for i, kp in enumerate(view_data['keypoint_names']):
                row_data[f'{kp}_x'] = keypoints_2d[i][0]
                row_data[f'{kp}_y'] = keypoints_2d[i][1]
            pseudo_rows.append(pd.Series(row_data, name=img_path))

        return pseudo_rows

    # -- CSV output with suffix -------------------------------------------

    def _generate_csv_files(self, filtered_data, pseudo_data, existing_data, output_dir):
        """Override: use csv_file_suffix for output filenames."""
        extraction_info = self._create_extraction_info(existing_data, pseudo_data)
        suffix = self.config.csv_file_suffix

        for view_name in self.views:
            logger.info(f"Processing {view_name}...")
            combined_df = self._create_view_dataframe(
                view_name, existing_data, pseudo_data, filtered_data
            )
            if not combined_df.empty:
                csv_path = output_dir / f"CollectedData_{view_name}{suffix}.csv"
                combined_df.to_csv(csv_path)
                logger.info(f"Saved {len(combined_df)} frames to {csv_path.name}")

        return extraction_info

    # -- Calibrations with suffix -----------------------------------------

    def _generate_calibrations_file(self, pseudo_data, existing_data, output_dir):
        """Override: read/write calibrations files with csv_file_suffix."""
        suffix = self.config.csv_file_suffix
        logger.info(f"Generating calibrations{suffix}.csv...")

        calib_dir = output_dir / "calibrations"
        if not calib_dir.exists() or not list(calib_dir.glob("*.toml")):
            logger.info("No calibration files found — skipping")
            return

        calib_files = {
            f.replace('.toml', ''): f
            for f in os.listdir(calib_dir) if f.endswith('.toml')
        }

        collected_path = output_dir / f"CollectedData_{self.views[0]}{suffix}.csv"
        if not collected_path.exists():
            logger.warning(f"CollectedData CSV not found at {collected_path}")
            return

        try:
            collected_df = pd.read_csv(collected_path, header=[0, 1, 2], index_col=0)
        except Exception as e:
            logger.error(f"Failed to read CollectedData CSV: {e}")
            return

        mappings = []
        for img_path in collected_df.index:
            session_dir = img_path.split('/')[-2]
            session_name = session_dir
            for view in self.views:
                session_name = (session_name
                    .replace(f'_{view}.', '.')
                    .replace(f'_{view}', '')
                    .replace(f'{view}_', ''))
            if session_name in calib_files:
                mappings.append({
                    '': img_path,
                    'file': f"calibrations/{calib_files[session_name]}"
                })

        if mappings:
            df = pd.DataFrame(mappings).drop_duplicates(subset=[''], keep='first')
            out_path = output_dir / f"calibrations{suffix}.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Generated {out_path.name} with {len(df)} mappings")
        else:
            logger.warning("No calibration mappings found")

    # -- Per-session balanced frame selection ------------------------------

    def create_pseudo_labeled_dataset(self, filtered_data: Dict) -> Dict:
        """Per-session balanced K-means: allocate n_clusters / n_sessions per session."""
        from sklearn.cluster import KMeans

        n_clusters = self.config.n_clusters
        valid_sessions = sorted([
            sn for sn, sd in filtered_data.items()
            if 'keypoints_3d' in sd and len(sd['keypoints_3d']) > 0
        ])
        n_sessions = len(valid_sessions)

        if n_sessions == 0:
            logger.warning("No sessions with 3D data — falling back to pooled clustering")
            return super().create_pseudo_labeled_dataset(filtered_data)

        base_per = n_clusters // n_sessions
        remainder = n_clusters % n_sessions
        logger.info(
            f"Per-session balanced K-means: {n_clusters} total across "
            f"{n_sessions} sessions (~{base_per} per session)"
        )

        all_selected: Dict = {}
        global_idx = 0
        reconstruction_methods = {}

        for i, session_name in enumerate(valid_sessions):
            n_target = base_per + (1 if i < remainder else 0)
            sd = filtered_data[session_name]
            reconstruction_methods[session_name] = sd.get('reconstruction_method', 'unknown')

            session_kps, session_fi = self._collect_3d_keypoints({session_name: sd})
            if len(session_kps) == 0:
                logger.warning(f"  {session_name}: no valid 3D poses — skipped")
                continue

            n_use = min(n_target, len(session_kps))
            logger.info(f"  {session_name}: {len(session_kps)} poses → {n_use} frames")

            if n_use >= len(session_kps):
                # Too few poses: take all
                for j, (sn, fi, of) in enumerate(session_fi):
                    all_selected[global_idx] = {
                        'session_name': sn.replace('.short', ''),
                        'original_session_name': sn,
                        'filtered_frame_idx': fi,
                        'original_frame_number': of,
                        'cluster_center': session_kps[j],
                        'num_frames_in_cluster': 1,
                    }
                    global_idx += 1
            else:
                km = KMeans(n_clusters=n_use, random_state=self.config.random_seed, n_init=10)
                labels = km.fit_predict(session_kps)
                for c in range(n_use):
                    cidx = np.where(labels == c)[0]
                    if len(cidx) == 0:
                        continue
                    center = km.cluster_centers_[c]
                    dists = np.linalg.norm(session_kps[cidx] - center, axis=1)
                    best = cidx[np.argmin(dists)]
                    sn, fi, of = session_fi[best]
                    all_selected[global_idx] = {
                        'session_name': sn.replace('.short', ''),
                        'original_session_name': sn,
                        'filtered_frame_idx': fi,
                        'original_frame_number': of,
                        'cluster_center': center,
                        'num_frames_in_cluster': len(cidx),
                    }
                    global_idx += 1

        logger.info(f"Per-session selection complete: {global_idx} frames total")
        all_kps, all_fi = self._collect_3d_keypoints(filtered_data)
        centers = (
            np.array([v['cluster_center'] for v in all_selected.values()])
            if all_selected else np.array([])
        )
        return {
            'selected_frames': all_selected,
            'cluster_centers': centers,
            'all_3d_keypoints': all_kps,
            'cluster_labels': None,
            'frame_info': all_fi,
            'reconstruction_methods': reconstruction_methods,
        }

    def create_random_pseudo_labeled_dataset(self, filtered_data: Dict) -> Dict:
        """Per-session balanced random selection: allocate n_random / n_sessions per session."""
        import random as _random

        n_total = self.config.n_random_frames
        valid_sessions = sorted([
            sn for sn, sd in filtered_data.items()
            if 'keypoints_3d' in sd and len(sd['keypoints_3d']) > 0
        ])
        n_sessions = len(valid_sessions)

        if n_sessions == 0:
            logger.warning("No sessions with 3D data — falling back to pooled random")
            return super().create_random_pseudo_labeled_dataset(filtered_data)

        base_per = n_total // n_sessions
        remainder = n_total % n_sessions
        logger.info(
            f"Per-session balanced random: {n_total} total across "
            f"{n_sessions} sessions (~{base_per} per session)"
        )

        all_selected: Dict = {}
        global_idx = 0
        reconstruction_methods = {}

        for i, session_name in enumerate(valid_sessions):
            n_target = base_per + (1 if i < remainder else 0)
            sd = filtered_data[session_name]
            reconstruction_methods[session_name] = sd.get('reconstruction_method', 'unknown')

            session_kps, session_fi = self._collect_3d_keypoints({session_name: sd})
            if len(session_kps) == 0:
                continue

            n_use = min(n_target, len(session_kps))
            _random.seed(self.config.random_seed + i)
            chosen = sorted(_random.sample(range(len(session_kps)), n_use))
            for j in chosen:
                sn, fi, of = session_fi[j]
                all_selected[global_idx] = {
                    'session_name': sn.replace('.short', ''),
                    'original_session_name': sn,
                    'filtered_frame_idx': fi,
                    'original_frame_number': of,
                    'cluster_center': session_kps[j],
                    'num_frames_in_cluster': 1,
                }
                global_idx += 1

        all_kps, all_fi = self._collect_3d_keypoints(filtered_data)
        return {
            'selected_frames': all_selected,
            'cluster_centers': np.array([v['cluster_center'] for v in all_selected.values()]) if all_selected else np.array([]),
            'all_3d_keypoints': all_kps,
            'cluster_labels': None,
            'frame_info': all_fi,
            'reconstruction_methods': reconstruction_methods,
        }

    # -- Fix None guard in _create_view_dataframe --------------------------

    def _create_view_dataframe(self, view_name, existing_data, pseudo_data, filtered_data):
        """Override: guard against existing_data=None."""
        frames_list = []

        if existing_data:
            existing_df = existing_data.get(view_name, {}).get('data', pd.DataFrame())
            if not existing_df.empty:
                flat = existing_df.copy()
                flat.columns = [
                    f"{c[1]}_{c[2]}" if isinstance(c, tuple) else str(c)
                    for c in existing_df.columns
                ]
                flat.index = [p.replace('.short', '') for p in flat.index]
                frames_list.append(flat)

        pseudo_rows = self._create_pseudo_rows(view_name, pseudo_data, filtered_data)
        if pseudo_rows:
            frames_list.append(pd.DataFrame(pseudo_rows))

        if not frames_list:
            return pd.DataFrame()

        combined_df = pd.concat(frames_list, axis=0)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Guard: existing_data can be None
        first_view_data = None
        if existing_data:
            first_view_data = next(
                (existing_data[v]['data'] for v in self.views
                 if v in existing_data and not existing_data[v]['data'].empty),
                None,
            )

        if first_view_data is not None:
            bodyparts = []
            seen = set()
            for col in first_view_data.columns:
                if isinstance(col, tuple) and len(col) == 3:
                    bp = col[1]
                    if bp not in seen and col[2] in ('x', 'y'):
                        bodyparts.append(bp)
                        seen.add(bp)
        else:
            # Preserve insertion order (EKS CSV column order = config keypoint order)
            bodyparts = list(dict.fromkeys(
                col.rsplit('_', 1)[0]
                for col in combined_df.columns
                if col.endswith(('_x', '_y'))
            ))

        columns = pd.MultiIndex.from_tuples(
            [('anipose', bp, coord) for bp in bodyparts for coord in ['x', 'y']],
            names=['scorer', 'bodyparts', 'coords'],
        )
        df_cols = [f'{bp}_{coord}' for bp in bodyparts for coord in ['x', 'y']]
        combined_df = combined_df.reindex(columns=df_cols, fill_value=np.nan)
        combined_df.columns = columns
        return combined_df


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_cheese3d_distillation_pipeline(config: Cheese3dDistillationConfig) -> Dict:
    """Run the cheese-3d distillation pipeline.

    Same steps as the generic pipeline, with these additions:
      - Step 3.5: Likelihood filter
      - Step 5.5: Per-keypoint reprojection for label generation
    """
    logger.info("Starting cheese-3d distillation pipeline...")

    pipeline = Cheese3dDistillationPipeline(config)

    # Step 0
    logger.info("Step 0: Copying existing data and calibrations...")
    copied_existing = pipeline.copy_existing_data()
    copied_calib = pipeline.copy_calibrations()

    # Step 1
    logger.info("Step 1: Loading data...")
    eks_data = pipeline.load_eks_data()

    existing_data_dict = pipeline.load_existing_data()
    if existing_data_dict:
        existing_data = existing_data_dict['filtered']
        original_existing_data = existing_data_dict['original']
    else:
        existing_data = None
        original_existing_data = None

    # Step 2
    logger.info("Step 2: Adding camera data and performing 3D reconstruction...")
    eks_data_3d = pipeline.add_camera_data_and_triangulate(eks_data)

    # Step 3
    logger.info("Step 3: Filtering out labeled frames...")
    filtered_data = pipeline.filter_out_labeled_frames(eks_data_3d, original_existing_data)

    # Step 3.5 — cheese-3d specific
    logger.info(
        "Step 3.5: Filtering by likelihood (>= %d views with likelihood > %.2f per keypoint)...",
        config.min_confident_views, config.likelihood_threshold,
    )
    filtered_data = pipeline.filter_by_likelihood(filtered_data)

    # Step 4
    if config.use_random_selection:
        logger.info("Step 4: Filtering by random selection...")
        filtered_frames = pipeline.filter_by_random(filtered_data)
    else:
        logger.info("Step 4: Filtering by variance...")
        filtered_frames = pipeline.filter_by_variance(filtered_data)

    # Step 5
    logger.info("Step 5: Creating pseudo-labeled dataset...")
    if config.use_random_selection:
        pseudo_data = pipeline.create_random_pseudo_labeled_dataset(filtered_frames)
    else:
        pseudo_data = pipeline.create_pseudo_labeled_dataset(filtered_frames)

    # Step 5.5 — cheese-3d specific: per-keypoint triangulation + reprojection
    logger.info("Step 5.5: Computing reprojected labels (per-keypoint triangulation)...")
    pipeline._compute_reprojected_labels(filtered_frames, pseudo_data)

    # Step 6
    logger.info("Step 6: Generating outputs...")
    outputs = pipeline.generate_outputs(filtered_frames, pseudo_data, existing_data)

    # Step 7
    logger.info("Step 7: Extracting frames...")
    extraction_results = pipeline.extract_frames(outputs['extraction_info'])

    logger.info("Cheese-3d distillation pipeline completed successfully!")

    return {
        'eks_data': eks_data_3d,
        'filtered_data': filtered_frames,
        'pseudo_data': pseudo_data,
        'outputs': outputs,
        'extraction_results': extraction_results,
        'copied_existing_data': copied_existing,
        'copied_calibrations': copied_calib,
    }
