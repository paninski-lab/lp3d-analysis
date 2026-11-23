"""Crop the 3d-pop dataset into a single-animal dataset, 3d-pop-crop.

Each row (instance = one bird at one time point) of 3d-pop's label/bbox CSVs becomes its own
cropped image in 3d-pop-crop/labeled-data/<sequence>_id<tag>_<view>/img<frame>.png, where <tag> is
the bird's tag number (raw ids look like '483_1307' = tag + date). Sequences are therefore
effectively split into one "session" per bird, <sequence>_id<tag>, and the label/bbox CSVs have
unique rows (unlike 3d-pop, where several rows share one full-frame image). The row order, and
the boxes, are identical to 3d-pop's.

Crops use the boxes already stored in 3d-pop/bboxes_<view>[_new].csv (x, y, h, w in original
pixels; square, centered on the tight extent of the bird's 9 keypoints in that view, side = 2.0 x
the larger extent; see pop3d.keypoint_bbox), resized to dataset_info['3d-pop']['crop_size'].
Out-of-frame regions are padded with black. Labels are converted to crop coordinates.

Outputs (train rows in CollectedData_<view>.csv / bboxes_<view>.csv; test rows in the `_new` files):
    labeled-data/<sequence>_id<tag>_<view>/img%08d.png
    CollectedData_<view>[_new].csv   (keypoints in crop pixels)
    bboxes_<view>[_new].csv          (columns x, y, h, w; original-image pixel coords)
    calibrations/<sequence>_id<tag>.toml   (identical copies of 3d-pop's per-sequence toml)
    labeled-data-check/    flat directory, one image per selected time point: for each bird (rows)
        and view (columns), the crop with the skeleton obtained by triangulating the labels with
        the copied calibration and reprojecting into that view (checks labels, boxes and
        calibration at once)
"""

import argparse
import shutil
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import toml

from lp3d_analysis import dataset_info
from lp3d_analysis.preprocess import pop3d as P

# ----------------------------------------------------------------------------------------
# path info / config
# ----------------------------------------------------------------------------------------
raw_dir = Path('/media/mattw/multiview-data/_raw/3d-pop')
selection_csv = raw_dir / 'frame_selection' / 'selection.csv'
dst_root = Path('/media/mattw/multiview-data')
src_dir = dst_root / '3d-pop'

data_info = dataset_info['3d-pop']
camera_names = data_info['cam_names']
keypoints = data_info['keypoints']
skeleton = data_info['skeleton']
skeleton_colors = data_info['skeleton_colors']
crop_size = data_info['crop_size']  # (width, height)
scorer = 'pop3d'

run_crop = True
run_copy_calibrations = True
run_check = True


def img_path(seq: str, cam: str, frame: int) -> str:
    """Relative path of the full-frame image in 3d-pop."""
    return f'labeled-data/{seq}_{cam}/img{frame:08d}.png'


def bird_tag(bird: str) -> str:
    """'483_1307' -> '483'."""
    return bird.split('_')[0]


def load_instances(sel: pd.DataFrame, split: str) -> list:
    """(sequence, frame, bird, time_point_index) per row of the 3d-pop CSVs for this split."""
    s = sel[sel.split == split].sort_values(['n_birds', 'sequence', 'frame'])
    out = []
    for i, r in enumerate(s.itertuples()):
        for bird in r.birds.split(';'):
            out.append((r.sequence, int(r.frame), bird, i))
    return out


# ----------------------------------------------------------------------------------------
# crop
# ----------------------------------------------------------------------------------------
def crop_split(sel: pd.DataFrame, dst_dir: Path, split: str, suffix: str) -> None:
    instances = load_instances(sel, split)
    for seq in {q for q, _, _, _ in instances}:  # tags must be unique within a sequence
        birds = sorted({b for q, _, b, _ in instances if q == seq})
        assert len({bird_tag(b) for b in birds}) == len(birds), seq

    for cam in camera_names:
        lab = pd.read_csv(
            src_dir / f'CollectedData_{cam}{suffix}.csv', header=[0, 1, 2], index_col=0
        )
        bb = pd.read_csv(src_dir / f'bboxes_{cam}{suffix}.csv', index_col=0)
        assert len(instances) == len(lab) == len(bb)
        assert lab.index.tolist() == bb.index.tolist()

        label_rows, bbox_rows, paths = [], [], []
        cur_path, frame_img = None, None
        for k, (seq, frame, bird, _) in enumerate(instances):
            src_path = img_path(seq, cam, frame)
            assert lab.index[k] == src_path, (k, lab.index[k], src_path)  # row order matches
            if src_path != cur_path:
                frame_img = cv2.imread(str(src_dir / src_path))
                assert frame_img is not None, src_path
                cur_path = src_path

            x, y, h, w = (int(v) for v in bb.iloc[k].to_numpy())
            cropped = P.crop_array(frame_img, x, y, h, w)
            resized = cv2.resize(cropped, tuple(crop_size), interpolation=cv2.INTER_LINEAR)

            name = f'{seq}_id{bird_tag(bird)}_{cam}'
            save_dir = dst_dir / 'labeled-data' / name
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f'img{frame:08d}.png'), resized)

            xy = lab.iloc[k].to_numpy().reshape(-1, 2).astype(float)
            xy_crop = (xy - np.array([x, y])) / np.array([w, h]) * np.array(crop_size)
            paths.append(f'labeled-data/{name}/img{frame:08d}.png')
            label_rows.append(xy_crop.reshape(-1))
            bbox_rows.append([x, y, h, w])

        assert len(set(paths)) == len(paths), 'duplicate rows'
        cols = pd.MultiIndex.from_product(
            [[scorer], keypoints, ['x', 'y']], names=['scorer', 'bodyparts', 'coords']
        )
        pd.DataFrame(np.stack(label_rows), index=paths, columns=cols).to_csv(
            dst_dir / f'CollectedData_{cam}{suffix}.csv'
        )
        pd.DataFrame(bbox_rows, index=paths, columns=['x', 'y', 'h', 'w']).to_csv(
            dst_dir / f'bboxes_{cam}{suffix}.csv'
        )
        print(f'{cam}{suffix}: cropped {len(paths)} instances', flush=True)


def copy_calibrations(sel: pd.DataFrame, dst_dir: Path) -> None:
    """One copy of each sequence's toml per bird in that sequence."""
    save_dir = dst_dir / 'calibrations'
    save_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for seq, g in sel.groupby('sequence'):
        for bird in g.birds.iloc[0].split(';'):
            shutil.copy(
                src_dir / 'calibrations' / f'{seq}.toml',
                save_dir / f'{seq}_id{bird_tag(bird)}.toml',
            )
            n += 1
    print(f'copied calibrations: {sel.sequence.nunique()} sequences -> {n} per-bird tomls')


# ----------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------
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


def check_dataset(sel: pd.DataFrame, dst_dir: Path, max_reproj_px: float = 2.0) -> None:
    """Programmatic checks + flat-directory overlay images (labeled-data-check/)."""
    w_crop, h_crop = crop_size
    colors = [tuple(int(255 * c) for c in mcolors.to_rgb(name)[::-1]) for name in skeleton_colors]
    out_dir = dst_dir / 'labeled-data-check'
    out_dir.mkdir(parents=True, exist_ok=True)
    cell = 400
    n_tp, worst_all = 0, 0.0
    cals = {}

    for split, suffix in (('train', ''), ('test', '_new')):
        instances = load_instances(sel, split)
        lab, bb = {}, {}
        for cam in camera_names:
            lab[cam] = pd.read_csv(
                dst_dir / f'CollectedData_{cam}{suffix}.csv', header=[0, 1, 2], index_col=0
            )
            bb[cam] = pd.read_csv(dst_dir / f'bboxes_{cam}{suffix}.csv', index_col=0)
            src_bb = pd.read_csv(src_dir / f'bboxes_{cam}{suffix}.csv', index_col=0)
            assert len(lab[cam]) == len(instances) == len(src_bb)
            assert not lab[cam].index.duplicated().any()
            assert lab[cam].index.tolist() == bb[cam].index.tolist()
            assert np.array_equal(bb[cam].to_numpy(), src_bb.to_numpy()), (
                'bboxes differ from 3d-pop'
            )
            assert all((dst_dir / p).exists() for p in lab[cam].index)
            xy = lab[cam].to_numpy()
            assert (
                np.isfinite(xy).all() and (xy[:, 0::2] >= 0).all() and (xy[:, 0::2] <= w_crop).all()
            )
            assert (xy[:, 1::2] >= 0).all() and (xy[:, 1::2] <= h_crop).all(), (
                'keypoint outside crop'
            )
        for cam in camera_names[1:]:
            assert [p.replace(f'_{cam}/', f'_{camera_names[0]}/') for p in lab[cam].index] == lab[
                camera_names[0]
            ].index.tolist(), 'rows differ across views'

        # one figure per time point
        tps = sorted({i for *_, i in instances})
        for tp in tps:
            ks = [k for k, inst in enumerate(instances) if inst[3] == tp]
            seq, frame = instances[ks[0]][0], instances[ks[0]][1]
            rows = []
            for k in ks:
                _, _, bird, _ = instances[k]
                name = f'{seq}_id{bird_tag(bird)}'
                if name not in cals:
                    cals[name] = load_toml_calibration(dst_dir / 'calibrations' / f'{name}.toml')
                kp_orig = []
                for cam in camera_names:
                    x, y, h, w = bb[cam].iloc[k].to_numpy()
                    kp_orig.append(
                        np.array([x, y])
                        + lab[cam].iloc[k].to_numpy().reshape(-1, 2)
                        * np.array([w, h])
                        / np.array(crop_size)
                    )
                kp_orig = np.stack(kp_orig)  # (4, 9, 2) in original-frame pixels
                pts3d = P.triangulate(kp_orig, cals[name])
                proj = P.reproject(pts3d, cals[name])  # (4, 9, 2)
                err = np.linalg.norm(proj - kp_orig, axis=-1)  # (4, 9)
                worst_all = max(worst_all, float(err.max()))
                tiles = []
                for ci, cam in enumerate(camera_names):
                    x, y, h, w = bb[cam].iloc[k].to_numpy()
                    img = cv2.imread(str(dst_dir / lab[cam].index[k]))
                    img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_LINEAR)
                    s = cell / np.array(crop_size)
                    reproj_crop = (
                        (proj[ci] - np.array([x, y])) / np.array([w, h]) * np.array(crop_size) * s
                    )
                    label_crop = lab[cam].iloc[k].to_numpy().reshape(-1, 2) * s
                    for (i0, i1), c in zip(skeleton, colors):
                        cv2.line(
                            img,
                            tuple(np.round(reproj_crop[i0]).astype(int)),
                            tuple(np.round(reproj_crop[i1]).astype(int)),
                            c,
                            2,
                            cv2.LINE_AA,
                        )
                    for pt in reproj_crop:
                        cv2.circle(
                            img,
                            tuple(np.round(pt).astype(int)),
                            4,
                            (255, 255, 255),
                            -1,
                            cv2.LINE_AA,
                        )
                    for pt in label_crop:
                        cv2.circle(
                            img, tuple(np.round(pt).astype(int)), 2, (0, 0, 0), -1, cv2.LINE_AA
                        )
                    txt = f'{cam} id{bird_tag(bird)}  reproj max {err[ci].max():.2f}px'
                    cv2.putText(
                        img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
                    )
                    cv2.putText(
                        img,
                        txt,
                        (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    tiles.append(img)
                rows.append(np.hstack(tiles))
            grid = np.vstack(rows)
            header = np.zeros((34, grid.shape[1], 3), np.uint8)
            n_tp += 1
            title = f'{n_tp:03d}  {split}  {seq}  frame {frame}  ({len(ks)} birds)'
            cv2.putText(
                header,
                title,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(
                str(out_dir / f'{n_tp:03d}_{split}_{seq}_img{frame:08d}.png'),
                np.vstack([header, grid]),
            )
        print(f'{split}: {len(instances)} instances / {len(tps)} time points OK', flush=True)

    assert worst_all <= max_reproj_px, f'worst triangulate->reproject error {worst_all:.2f}px'
    print(
        f'worst triangulate->reproject error over all instances: {worst_all:.2f}px; '
        f'{n_tp} check images in {out_dir}'
    )


# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--dry-run', action='store_true', help='4 time points per pool, <name>_dryrun dir'
    )
    args = ap.parse_args()

    sel = pd.read_csv(selection_csv)
    if args.dry_run:
        # needs a matching dry-run 3d-pop dir; keep the first 4 time points of the first sequence
        # of each pool, the same subset scripts/preprocess/3d-pop.py --dry-run uses
        src_dir = dst_root / '3d-pop_dryrun'
        sel = sel.groupby(['split', 'n_birds'], group_keys=False).apply(
            lambda g: g[g.sequence == g.sequence.iloc[0]].head(4)
        )
    sel = sel.sort_values(['split', 'n_birds', 'sequence', 'frame']).reset_index(drop=True)

    dst_dir = dst_root / ('3d-pop-crop_dryrun' if args.dry_run else '3d-pop-crop')
    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise SystemExit(f'{dst_dir} already exists and is not empty; refusing to overwrite')
    dst_dir.mkdir(parents=True, exist_ok=True)

    if run_crop:
        crop_split(sel, dst_dir, 'train', '')
        crop_split(sel, dst_dir, 'test', '_new')
    if run_copy_calibrations:
        copy_calibrations(sel, dst_dir)
    if run_check:
        check_dataset(sel, dst_dir)
