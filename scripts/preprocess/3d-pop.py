"""Convert 3D-POP (multi-animal, 4-view pigeon data) to Lightning Pose format, "3d-pop".

Raw data: /media/mattw/multiview-data/_raw/3d-pop (Edmond download, doi:10.17617/3.HPBBC7;
code: https://github.com/alexhang212/Dataset-3DPOP). Only the 2- and 5-pigeon sessions are used;
the 10-pigeon sessions are ignored. We use the full-sequence files (`Videos/`, `Annotation/`,
`CalibrationInfo/`), NOT the raw `TrainingSplit/` folders: those are re-encoded chunks whose
videos are one frame shorter than their annotations, whereas `Annotation/` rows line up exactly
with `Videos/` frames (row index == frame index, verified for all sequences/cameras).

Splits: sequences are split 50/50 (seed 0, stratified by bird count) into train sequences (InD)
and test sequences (OOD, `_new` suffix). Test sequences never overlap train sequences.

An *instance* is one bird at one time point. We select whole time points (every bird visible at
that time) so each time point contributes 2 or 5 instances: 125 two-bird + 50 five-bird time
points per split = 250 + 250 = 500 instances per split. Each CollectedData_<cam>[_new].csv and
bboxes_<cam>[_new].csv therefore has 500 rows, with several consecutive rows (one per bird, in a
fixed order) pointing at the same full-frame image. The images are full frames (not crops); a
derived cropped dataset is built separately.

Candidate time points (all must hold for every bird):
    - all 9 anatomical keypoints present in the *filtered* 2D annotations in all 4 views
    - bbox present with positive size in all 4 views; complete 3D pose
    - the 4 per-camera Keypoint3D files agree to within 3 mm (they differ by ~1 frame of motion
      in some sequences, which would make 2D labels inconsistent across views)
    - frames t-1 and t+1 also pass all of the above (neighbouring frames with NaN labels indicate
      fast motion / blur, where head keypoints are least accurate)
Sequences 39 and 50 (2-bird) are excluded because of very high NaN rates.

Frame selection (see lp3d_analysis.preprocess.pop3d.select_time_points): stratify by inter-bird
separation (5 quantile bins; centroid distance for 2 birds, mean nearest-neighbour distance for
5 birds), then greedily cover ego-aligned pose clusters (k-means, K=30, square-root-flattened
target). Intermediate artifacts are written to <raw>/frame_selection/.

Keypoints are the 9 anatomical keypoints (motion-capture markers are not used). 2D labels are the
GESD-filtered 2D annotations. World units are mm.

Bounding boxes (x, y, h, w in original pixels) are NOT the raw 3D-POP boxes: they are square boxes
derived from each bird's 9 keypoints in each view (center of the tight keypoint extent, side =
2.0 x the larger extent; see pop3d.keypoint_bbox), so that they are identical to the boxes used
to build the cropped dataset (scripts/preprocess/3d-pop-crop.py) and to what a cropzoom detector
would produce. The raw 3D-POP boxes remain in the raw Annotation/ files.

Steps (set flags below, or use the CLI): select -> extract -> labels -> calibrations -> check
    python scripts/preprocess/3d-pop.py --dry-run   # first sequence of each pool only, separate dir
    python scripts/preprocess/3d-pop.py --update-bboxes
        # rewrite bboxes_*.csv of an existing 3d-pop dir from its label CSVs, and redo the check
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import toml

from lp3d_analysis import dataset_info
from lp3d_analysis.preprocess import pop3d as P
from lp3d_analysis.video import export_frames

# ----------------------------------------------------------------------------------------
# path info / config
# ----------------------------------------------------------------------------------------
raw_dir = Path('/media/mattw/multiview-data/_raw/3d-pop')
selection_dir = raw_dir / 'frame_selection'
dst_root = Path('/media/mattw/multiview-data')

data_info = dataset_info['3d-pop']
camera_names = data_info['cam_names']  # Cam1..Cam4, same names as the raw videos
keypoints = data_info['keypoints']  # same order as the raw hd_/bp_ columns in P.KEYPOINTS
assert len(keypoints) == len(P.KEYPOINTS)
video_size = (3840, 2160)  # (width, height)
scorer = 'pop3d'

exclude_sequences = ['Sequence39_n02_29062022', 'Sequence50_n02_28062022']
n_time_points = {2: 125, 5: 50}  # per split
stride = 5  # candidate time points: every 5th valid frame
n_clusters = 30

run_select = False  # recompute frame selection (never overwrites unless --force-select)
run_extract_frames = True
run_labels = True
run_calibrations = True
run_check = True


# ----------------------------------------------------------------------------------------
# frame selection
# ----------------------------------------------------------------------------------------
def run_frame_selection(out_dir: Path) -> pd.DataFrame:
    """Split sequences, build candidate pools, cluster poses, select time points.

    Writes selection.csv, sequence_split.json, candidate_pool_stride5.pkl and
    reproj_err_candidates.pkl to `out_dir`, and returns the selection table.
    """
    seqs = P.list_sequences(raw_dir, exclude_sequences)
    rng = np.random.default_rng(0)
    split = {}
    for nb in (2, 5):
        s = [q for q in seqs if P.n_birds(q) == nb]
        perm = rng.permutation(len(s))
        for i, j in enumerate(perm):
            split[s[j]] = 'train' if i < len(s) // 2 else 'test'

    cache, reproj = {}, {}
    for q in seqs:
        d = P.load_sequence(raw_dir, q)
        frames = np.where(P.eligible_mask(d))[0]
        frames = frames[frames % stride == 0]
        k3 = d['kp3d'][0][frames]
        ego, _ = P.ego_pose(k3)
        cache[q] = dict(
            birds=d['birds'],
            frames=frames,
            ego=ego.reshape(len(frames), len(d['birds']), -1),
            sep=np.array([P.separation(x) for x in k3]),
        )
        reproj[q] = (
            frames,
            P.reprojection_error(d['kp2d'][:, frames], P.load_calibration(raw_dir, q)),
        )
        print(f'{q}: {split[q]}, {len(frames)} candidate time points', flush=True)

    all_ego = np.concatenate([c['ego'].reshape(-1, 27) for c in cache.values()])
    mean, proj = P.pca_project(all_ego, 10)
    sub = np.random.default_rng(0).choice(len(all_ego), min(30000, len(all_ego)), replace=False)
    _, centers = P.kmeans((all_ego[sub] - mean) @ proj, n_clusters, seed=0)
    for c in cache.values():
        z = (c['ego'].reshape(-1, 27) - mean) @ proj
        lab = np.argmin(((z[:, None] - centers[None]) ** 2).sum(-1), 1).reshape(
            len(c['frames']), -1
        )
        c['cluster_counts'] = np.stack([np.bincount(row, minlength=n_clusters) for row in lab])

    rows = []
    for sp in ('train', 'test'):
        for nb in (2, 5):
            qs = [q for q in seqs if split[q] == sp and P.n_birds(q) == nb]
            seq_of = np.concatenate([[q] * len(cache[q]['frames']) for q in qs])
            idx = P.select_time_points(
                seq_of,
                np.concatenate([cache[q]['frames'] for q in qs]),
                np.concatenate([cache[q]['sep'] for q in qs]),
                np.concatenate([cache[q]['cluster_counts'] for q in qs]),
                n_time_points[nb],
                seed=0,
            )
            for i in idx:
                q = seq_of[i]
                j = i - sum(len(cache[r]['frames']) for r in qs[: qs.index(q)])  # index within q
                rows.append(
                    dict(
                        split=sp,
                        n_birds=nb,
                        sequence=str(q),
                        frame=int(cache[q]['frames'][j]),
                        birds=';'.join(cache[q]['birds']),
                        sep_mm=round(float(cache[q]['sep'][j]), 1),
                        max_reproj_px=round(float(reproj[q][1][:, j].max()), 3),
                    )
                )
    df = pd.DataFrame(rows).sort_values(['split', 'n_birds', 'sequence', 'frame'])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'selection.csv', index=False)
    json.dump(split, open(out_dir / 'sequence_split.json', 'w'), indent=1)
    pickle.dump(
        dict(cache=cache, kmeans=dict(mean=mean, proj=proj, centers=centers)),
        open(out_dir / 'candidate_pool_stride5.pkl', 'wb'),
    )
    pickle.dump(reproj, open(out_dir / 'reproj_err_candidates.pkl', 'wb'))
    return df


def load_selection(dry_run: bool) -> pd.DataFrame:
    sel = pd.read_csv(selection_dir / 'selection.csv')
    if dry_run:  # first sequence of each (split, n_birds) pool, first 4 time points
        sel = sel.groupby(['split', 'n_birds'], group_keys=False).apply(
            lambda g: g[g.sequence == g.sequence.iloc[0]].head(4)
        )
    return sel.sort_values(['split', 'n_birds', 'sequence', 'frame']).reset_index(drop=True)


def img_path(seq: str, cam: str, frame: int) -> str:
    return f'labeled-data/{seq}_{cam}/img{frame:08d}.png'


# ----------------------------------------------------------------------------------------
# extract frames (full frames, no context frames)
# ----------------------------------------------------------------------------------------
def extract_frames(sel: pd.DataFrame, dst_dir: Path) -> None:
    for seq, g in sel.groupby('sequence'):
        idxs = np.sort(g.frame.unique())
        for cam in camera_names:
            video = P.seq_dir(raw_dir, seq) / 'Videos' / f'{seq}-{cam}.mp4'
            export_frames(
                video_file=video,
                save_dir=dst_dir / 'labeled-data' / f'{seq}_{cam}',
                frame_idxs=idxs,
                context_frames=0,
                grayscale=False,
            )
        print(f'extracted {len(idxs)} frames x {len(camera_names)} views: {seq}', flush=True)


# ----------------------------------------------------------------------------------------
# labels + bboxes
# ----------------------------------------------------------------------------------------
def write_labels(sel: pd.DataFrame, dst_dir: Path) -> None:
    """One row per instance (bird) per time point; identical row order in every view."""
    index = pd.MultiIndex.from_product(
        [[scorer], keypoints, ['x', 'y']], names=['scorer', 'bodyparts', 'coords']
    )
    data = {q: P.load_sequence(raw_dir, q) for q in sel.sequence.unique()}
    for split, suffix in (('train', ''), ('test', '_new')):
        s = sel[sel.split == split]
        for ci, cam in enumerate(camera_names):
            label_rows, bbox_rows = {}, []
            paths = []
            for _, r in s.iterrows():
                d = data[r.sequence]
                assert d['birds'] == r.birds.split(';')
                for bi in range(len(d['birds'])):
                    paths.append(img_path(r.sequence, cam, r.frame))
                    label_rows[len(paths) - 1] = d['kp2d'][ci, r.frame, bi].reshape(-1)
                    bbox_rows.append(P.keypoint_bbox(d['kp2d'][ci, r.frame, bi]))  # x, y, h, w
            df = pd.DataFrame(np.stack(list(label_rows.values())), columns=index, index=paths)
            df.to_csv(dst_dir / f'CollectedData_{cam}{suffix}.csv')
            pd.DataFrame(bbox_rows, index=paths, columns=['x', 'y', 'h', 'w']).to_csv(
                dst_dir / f'bboxes_{cam}{suffix}.csv'
            )
            print(f'{cam}{suffix}: {len(df)} instances', flush=True)


def update_bboxes(dst_dir: Path) -> None:
    """Rewrite bboxes_<cam>[_new].csv from the label CSVs (keypoint-extent square boxes)."""
    for cam in camera_names:
        for suffix in ('', '_new'):
            lab = pd.read_csv(
                dst_dir / f'CollectedData_{cam}{suffix}.csv', header=[0, 1, 2], index_col=0
            )
            boxes = [P.keypoint_bbox(row.reshape(-1, 2)) for row in lab.to_numpy()]
            pd.DataFrame(boxes, index=lab.index, columns=['x', 'y', 'h', 'w']).to_csv(
                dst_dir / f'bboxes_{cam}{suffix}.csv'
            )
            print(f'{cam}{suffix}: rewrote {len(boxes)} bboxes', flush=True)


# ----------------------------------------------------------------------------------------
# calibrations (OpenCV pickles -> anipose toml)
# ----------------------------------------------------------------------------------------
def write_calibrations(sequences, dst_dir: Path) -> None:
    for seq in sequences:
        cal = P.load_calibration(raw_dir, seq)
        data = {
            f'cam_{i}': {
                'name': cam,
                'size': list(video_size),
                'matrix': c['K'].tolist(),
                'distortions': c['dist'].tolist(),
                'rotation': c['rvec'].tolist(),
                'translation': c['tvec'].tolist(),
            }
            for i, (cam, c) in enumerate(zip(camera_names, cal))
        }
        data['metadata'] = {}
        save_path = dst_dir / 'calibrations' / f'{seq}.toml'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            toml.dump(data, f)


def load_toml_calibration(path: Path) -> list:
    t = toml.load(path)
    return [
        dict(
            rvec=np.array(t[f'cam_{i}']['rotation']),
            tvec=np.array(t[f'cam_{i}']['translation']),
            K=np.array(t[f'cam_{i}']['matrix']),
            dist=np.array(t[f'cam_{i}']['distortions']),
        )
        for i in range(len(camera_names))
    ]


# ----------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------
def check_dataset(dst_dir: Path, sel: pd.DataFrame, max_reproj_px: float = 2.0) -> None:
    """Programmatic checks, read back from disk, plus overlay plots in labeled-data-check/."""
    n_expected = {sp: int(sum(P.n_birds(q) for q in g.sequence)) for sp, g in sel.groupby('split')}
    for suffix, sp in (('', 'train'), ('_new', 'test')):
        labels, bboxes = {}, {}
        for cam in camera_names:
            labels[cam] = pd.read_csv(
                dst_dir / f'CollectedData_{cam}{suffix}.csv', header=[0, 1, 2], index_col=0
            )
            bboxes[cam] = pd.read_csv(dst_dir / f'bboxes_{cam}{suffix}.csv', index_col=0)
            assert len(labels[cam]) == n_expected[sp], (cam, suffix, len(labels[cam]))
            assert labels[cam].index.tolist() == bboxes[cam].index.tolist()
            assert not labels[cam].isna().any().any(), 'NaN labels'
            assert all((dst_dir / p).exists() for p in labels[cam].index), 'missing image'
            xy = labels[cam].to_numpy().reshape(len(labels[cam]), -1, 2)
            assert (xy[..., 0] >= 0).all() and (xy[..., 0] < video_size[0]).all()
            assert (xy[..., 1] >= 0).all() and (xy[..., 1] < video_size[1]).all()
            assert (bboxes[cam][['h', 'w']] > 0).all().all()
            assert [k for k in labels[cam].columns.get_level_values(1).unique()] == keypoints
        base = camera_names[0]
        for cam in camera_names[1:]:
            assert [p.replace(f'_{cam}/', f'_{base}/') for p in labels[cam].index] == labels[
                base
            ].index.tolist(), 'rows differ across views'
        # reproject triangulated 2D labels using the written tomls
        worst = 0.0
        seq_of = np.array([Path(p).parent.name.rsplit('_', 1)[0] for p in labels[base].index])
        for seq in np.unique(seq_of):
            m = seq_of == seq
            kp = np.stack(
                [labels[c].to_numpy()[m].reshape(m.sum(), 1, -1, 2) for c in camera_names]
            )
            err = P.reprojection_error(
                kp, load_toml_calibration(dst_dir / 'calibrations' / f'{seq}.toml')
            )
            worst = max(worst, float(err.max()))
        assert worst <= max_reproj_px, f'reprojection error {worst:.2f}px > {max_reproj_px}'
        print(
            f'{sp}: {len(labels[base])} rows/view OK; worst triangulate->reproject error '
            f'{worst:.2f}px',
            flush=True,
        )

    # overlay plots: all birds' keypoints + bboxes, 4 views in a 2x2 grid
    colors = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 255, 80), (255, 80, 255)]
    scale = 0.25
    for _, r in sel.iterrows():
        tiles = []
        suffix = '' if r.split == 'train' else '_new'
        for cam in camera_names:
            lab = pd.read_csv(
                dst_dir / f'CollectedData_{cam}{suffix}.csv', header=[0, 1, 2], index_col=0
            )
            bb = pd.read_csv(dst_dir / f'bboxes_{cam}{suffix}.csv', index_col=0)
            p = img_path(r.sequence, cam, r.frame)
            img = cv2.imread(str(dst_dir / p))
            rows = np.where(lab.index == p)[0]
            for bi, ri in enumerate(rows):
                x, y, h, w = bb.iloc[ri].to_numpy()
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), colors[bi], 6)
                for kx, ky in lab.iloc[ri].to_numpy().reshape(-1, 2):
                    cv2.circle(img, (int(kx), int(ky)), 8, colors[bi], -1)
            tiles.append(cv2.resize(img, None, fx=scale, fy=scale))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        out = dst_dir / 'labeled-data-check' / r.sequence / f'img{r.frame:08d}.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), grid)


# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='first sequence of each pool only; writes to <name>_dryrun',
    )
    ap.add_argument('--force-select', action='store_true', help='allow overwriting selection.csv')
    ap.add_argument(
        '--update-bboxes',
        action='store_true',
        help='rewrite bboxes (and the labeled-data-check overlays) of an existing 3d-pop',
    )
    args = ap.parse_args()

    if args.update_bboxes:
        dst_dir = dst_root / '3d-pop'
        update_bboxes(dst_dir)
        check_dataset(dst_dir, load_selection(False))
        raise SystemExit(0)

    dst_dir = dst_root / ('3d-pop_dryrun' if args.dry_run else '3d-pop')
    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise SystemExit(f'{dst_dir} already exists and is not empty; refusing to overwrite')
    dst_dir.mkdir(parents=True, exist_ok=True)

    if run_select:
        if (selection_dir / 'selection.csv').exists() and not args.force_select:
            raise SystemExit('selection.csv exists; pass --force-select to overwrite')
        run_frame_selection(selection_dir)

    sel = load_selection(args.dry_run)
    print(sel.groupby(['split', 'n_birds']).size().rename('time_points'))
    if run_extract_frames:
        extract_frames(sel, dst_dir)
    if run_labels:
        write_labels(sel, dst_dir)
    if run_calibrations:
        write_calibrations(sel.sequence.unique(), dst_dir)
    if run_check:
        check_dataset(dst_dir, sel)
