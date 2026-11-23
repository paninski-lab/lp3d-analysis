#!/usr/bin/env python3
"""Render a model's out-of-distribution predictions as videos, one frame per labelled frame.

Three videos, all tiling the five camera views into one image per labelled frame:

``ood_uncropped.mp4``
    The full 1328x1048 frames with both animals' predictions, taken from
    ``preds_original/`` -- the full-frame coordinates every result is reported in.
``ood_cropped_an1.mp4`` / ``ood_cropped_an2.mp4``
    The 320x320 crops the model was actually given (from the model's own
    ``data_dir``, so for a SAM model these are the recoloured crops), with that
    animal's raw crop-space predictions. This is what the network saw and said.

Predictions are filled dots joined by the skeleton; ground truth is a hollow
ring at the same keypoint, so a gap between a dot and its ring is the error.
Each frame's caption gives the session, image, and mean error in **full-frame
pixels** for every video -- including the cropped ones, where the crop-pixel
error would not be comparable between frames.

The labelled frames are not consecutive video: they are samples from four
held-out recordings, so motion between frames is not meaningful.

Usage::

    python scripts/postprocess/make_ood_prediction_videos.py \\
        outputs/r24m_s_sam/crop_1.4/mvt_3d_loss_496_0
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/teamspace/studios/this_studio")
VIEWS = ["Camera1", "Camera2", "Camera3", "Camera5", "Camera6"]
ANIMALS = ["an1", "an2"]
# BGR. Two hues far apart, both legible on tan bedding and on black/white fur.
COLORS = {"an1": (255, 170, 30), "an2": (60, 60, 255)}
GT_COLOR = (255, 255, 255)
PANEL_BG = (32, 32, 32)


def load_xy(path: Path, keypoints: list[str]) -> tuple[pd.Index, np.ndarray]:
    """(index, (n, n_kp, 2)) with columns in ``keypoints`` order, `set` ignored."""
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df = df.drop(columns=[c for c in df.columns if str(c[0]) == "set"])
    sc = df.columns.get_level_values(0)[0]
    xy = np.stack([np.stack([df[(sc, k, "x")].to_numpy(float),
                             df[(sc, k, "y")].to_numpy(float)], -1) for k in keypoints], 1)
    return df.index, xy


def draw(img: np.ndarray, pred: np.ndarray, gt: np.ndarray, keypoints: list[str],
         skeleton: list[tuple[str, str]], color, scale: float, r: int) -> None:
    """Draw one animal. ``pred``/``gt`` are (n_kp, 2) in the image's original pixels."""
    k = {name: i for i, name in enumerate(keypoints)}
    p = pred * scale
    g = gt * scale
    for a, b in skeleton:
        pa, pb = p[k[a]], p[k[b]]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(img, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)),
                     color, max(1, r // 2), cv2.LINE_AA)
    for q in g:
        if np.isfinite(q).all():
            cv2.circle(img, tuple(np.round(q).astype(int)), r + 2, GT_COLOR, 1, cv2.LINE_AA)
    for q in p:
        if np.isfinite(q).all():
            c = tuple(np.round(q).astype(int))
            cv2.circle(img, c, r, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, r - 1, color, -1, cv2.LINE_AA)


def label(img: np.ndarray, text: str, org, size: float = 0.6, color=(255, 255, 255),
          box: bool = True) -> None:
    """Text on a solid backing box.

    Not an outline: a thick black stroke drawn under the text at these sizes bleeds
    past the glyphs and reads as ghosted trailing characters ("7.6 pxx").
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, size, 1)
    x, y = org
    if box and text:
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + base + 4), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), font, size, color, 1, cv2.LINE_AA)


def tile(tiles: list[np.ndarray], info: np.ndarray) -> np.ndarray:
    """3 x 2 grid: five views and an info panel in the last slot."""
    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:] + [info])
    return np.vstack([top, bottom])


def info_panel(w: int, h: int, lines: list[tuple[str, tuple]]) -> np.ndarray:
    panel = np.full((h, w, 3), PANEL_BG, np.uint8)
    y = 36
    for text, color in lines:
        label(panel, text, (16, y), 0.55, color, box=False)
        y += 30
    return panel


class Writer:
    """H.264 via ffmpeg, yuv420p so the result plays in a browser and in VS Code."""

    def __init__(self, path: Path, w: int, h: int, fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def session_of(rel: str) -> tuple[str, str]:
    p = PurePosixPath(rel)
    s = "_".join(t for t in p.parent.name.split("_") if t not in VIEWS and t not in ANIMALS)
    return s, p.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--full-data", type=Path, default=ROOT / "data" / "r24m")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <run_dir>/videos")
    ap.add_argument("--limit", type=int, default=None, help="first N frames only")
    args = ap.parse_args()

    run = args.run_dir
    cfg = yaml.safe_load(open(run / "config.yaml"))["data"]
    crop_data = Path(cfg["data_dir"])
    stems = list(cfg["keypoint_names"])
    skeleton = [tuple(e) for e in cfg["skeleton"]]
    full_kps = [f"{s}_{a}" for a in ANIMALS for s in stems]
    out = args.out_dir or run / "videos"
    model = f"{run.parts[-3]}/{run.parts[-2]}/{run.parts[-1]}"

    # ---- full-frame -----------------------------------------------------------
    idx, pred_full, gt_full, err_full = {}, {}, {}, {}
    for v in VIEWS:
        idx[v], pred_full[v] = load_xy(run / "preds_original" / f"predictions_{v}_new.csv", full_kps)
        gi, gt_full[v] = load_xy(args.full_data / f"CollectedData_{v}_new.csv", full_kps)
        if list(gi) != list(idx[v]):
            raise AssertionError(f"{v}: predictions and ground truth rows differ")
        e = pd.read_csv(run / "preds_original" / f"predictions_{v}_new_pixel_error.csv", index_col=0)
        err_full[v] = e[full_kps].to_numpy(float)
    frames = [session_of(i) for i in idx[VIEWS[0]]]
    for v in VIEWS:
        if [session_of(i) for i in idx[v]] != frames:
            raise AssertionError(f"{v}: frame order differs from {VIEWS[0]}")
    n = len(frames) if args.limit is None else min(args.limit, len(frames))

    tw, th = 664, 524                                   # half of 1328x1048
    scale = tw / 1328.0
    w = Writer(out / "ood_uncropped.mp4", tw * 3, th * 2, args.fps)
    for i in range(n):
        session, name = frames[i]
        tiles = []
        for v in VIEWS:
            rel = idx[v][i]
            img = cv2.resize(cv2.imread(str(args.full_data / rel)), (tw, th),
                             interpolation=cv2.INTER_AREA)
            for j, a in enumerate(ANIMALS):
                sl = slice(j * len(stems), (j + 1) * len(stems))
                draw(img, pred_full[v][i, sl], gt_full[v][i, sl], stems, skeleton,
                     COLORS[a], scale, 4)
            label(img, f"{v}   err {np.nanmean(err_full[v][i]):.1f} px", (10, 26))
            tiles.append(img)
        err = np.nanmean([np.nanmean(err_full[v][i]) for v in VIEWS])
        per_an = [np.nanmean([np.nanmean(err_full[v][i, j * len(stems):(j + 1) * len(stems)])
                              for v in VIEWS]) for j in range(len(ANIMALS))]
        tiles_img = tile(tiles, info_panel(tw, th, [
            ("OOD predictions - uncropped (full frame)", (255, 255, 255)),
            (model, (180, 180, 180)),
            ("", PANEL_BG),
            (f"session  {session.replace('_Recording', '').replace('_social_wvid', '')}", (255, 255, 255)),
            (f"frame    {name}   ({i + 1}/{n})", (255, 255, 255)),
            ("", PANEL_BG),
            (f"mean error {err:.1f} px (full-frame)", (255, 255, 255)),
            (f"  an1  {per_an[0]:.1f} px", COLORS["an1"]),
            (f"  an2  {per_an[1]:.1f} px", COLORS["an2"]),
            ("", PANEL_BG),
            ("filled dot = prediction", (220, 220, 220)),
            ("white ring = ground truth", (220, 220, 220)),
        ]))
        w.write(tiles_img)
    w.close()
    print(f"wrote {out / 'ood_uncropped.mp4'}  ({n} frames)")

    # ---- cropped, one video per animal ----------------------------------------
    cw = 480
    cscale = cw / 320.0
    crop_idx, crop_pred, crop_gt = {}, {}, {}
    for v in VIEWS:
        crop_idx[v], crop_pred[v] = load_xy(run / f"predictions_{v}_new.csv", stems)
        gi, crop_gt[v] = load_xy(crop_data / f"CollectedData_{v}_new.csv", stems)
        if list(gi) != list(crop_idx[v]):
            raise AssertionError(f"{v}: crop predictions and ground truth rows differ")
    full_row = {session_of(r): k for k, r in enumerate(idx[VIEWS[0]])}

    for a in ANIMALS:
        rows = {v: [k for k, r in enumerate(crop_idx[v])
                    if a in PurePosixPath(r).parent.name.split("_")] for v in VIEWS}
        order = [session_of(crop_idx[VIEWS[0]][k]) for k in rows[VIEWS[0]]]
        for v in VIEWS:
            if [session_of(crop_idx[v][k]) for k in rows[v]] != order:
                raise AssertionError(f"{v}/{a}: frame order differs across views")
        j = ANIMALS.index(a)
        sl = slice(j * len(stems), (j + 1) * len(stems))
        m = len(order) if args.limit is None else min(args.limit, len(order))
        w = Writer(out / f"ood_cropped_{a}.mp4", cw * 3, cw * 2, args.fps)
        for i in range(m):
            session, name = order[i]
            fr = full_row[(session, name)]
            tiles = []
            for v in VIEWS:
                k = rows[v][i]
                img = cv2.resize(cv2.imread(str(crop_data / crop_idx[v][k])), (cw, cw),
                                 interpolation=cv2.INTER_LINEAR)
                draw(img, crop_pred[v][k], crop_gt[v][k], stems, skeleton, COLORS[a], cscale, 5)
                label(img, f"{v}   err {np.nanmean(err_full[v][fr, sl]):.1f} px", (10, 26))
                tiles.append(img)
            err = np.nanmean([np.nanmean(err_full[v][fr, sl]) for v in VIEWS])
            w.write(tile(tiles, info_panel(cw, cw, [
                (f"OOD predictions - cropped, {a}", COLORS[a]),
                (model, (180, 180, 180)),
                ("", PANEL_BG),
                (f"session {session.replace('_Recording', '').replace('_social_wvid', '')}", (255, 255, 255)),
                (f"frame   {name}", (255, 255, 255)),
                (f"        ({i + 1}/{m})", (255, 255, 255)),
                ("", PANEL_BG),
                (f"mean error {err:.1f} px", (255, 255, 255)),
                ("(full-frame px, not crop px)", (180, 180, 180)),
                ("", PANEL_BG),
                ("crops are the model's input", (220, 220, 220)),
                ("filled dot = prediction", (220, 220, 220)),
                ("white ring = ground truth", (220, 220, 220)),
            ])))
        w.close()
        print(f"wrote {out / f'ood_cropped_{a}.mp4'}  ({m} frames)")


if __name__ == "__main__":
    main()
