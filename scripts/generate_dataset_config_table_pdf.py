#!/usr/bin/env python3
"""
Build a publication-style PDF table: rows = hyperparameters, columns = datasets.

Reads lp3d-analysis YAML configs (Hydra-style), extracts nested fields with
sensible fallbacks (e.g. imgaug_3d vs. typo imagug_3d), and writes vector PDF.

Example:
  python lp3d-analysis/scripts/generate_dataset_config_table_pdf.py \\
    --out figures/dataset_training_hyperparameters.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------------------------------------
# Path resolution (script may be run from repo root or lp3d-analysis/)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LP3D_ROOT = SCRIPT_DIR.parent
REPO_ROOT = LP3D_ROOT.parent


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.2e}"
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v), separators=(", ", ": "))
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, separators=(", ", ": "))
    return str(v)


def num_views(cfg: dict) -> str:
    data = cfg.get("data") or {}
    cf = data.get("csv_file")
    if isinstance(cf, list) and cf:
        return str(len(cf))
    vn = data.get("view_names")
    if isinstance(vn, list) and vn:
        return str(len(vn))
    return "—"


def image_resize_dims(cfg: dict) -> str:
    dims = _get(cfg, "data", "image_resize_dims")
    if not isinstance(dims, dict):
        return "—"
    h, w = dims.get("height"), dims.get("width")
    if h is None or w is None:
        return "—"
    return f"{int(h)}×{int(w)}"


def imgaug_3d(cfg: dict) -> str:
    tr = cfg.get("training") or {}
    if "imgaug_3d" in tr:
        return _fmt(tr["imgaug_3d"])
    if "imagug_3d" in tr:  # typo in some configs
        return _fmt(tr["imagug_3d"])
    return "—"


def patch_mask_field(cfg: dict, key: str) -> str:
    pm = _get(cfg, "training", "patch_mask")
    if not isinstance(pm, dict):
        return "—"
    return _fmt(pm.get(key))


def supervised_repr_heatmap_mse(cfg: dict) -> str:
    losses = cfg.get("losses") or {}
    block = losses.get("supervised_reprojection_heatmap_mse")
    if not isinstance(block, dict):
        return "—"
    # Compact: log_weight is the main knob
    lw = block.get("log_weight")
    if lw is None:
        return _fmt(block)
    return _fmt(lw)


def freeze_until_epoch(cfg: dict) -> str:
    v = _get(cfg, "callbacks", "anneal_weight", "freeze_until_epoch")
    return _fmt(v)


def milestone_steps(cfg: dict) -> str:
    ms = _get(cfg, "training", "lr_scheduler_params", "multisteplr", "milestones")
    return _fmt(ms)


def lr_gamma(cfg: dict) -> str:
    g = _get(cfg, "training", "lr_scheduler_params", "multisteplr", "gamma")
    return _fmt(g)


Extractor = Callable[[dict], str]

ROWS: List[Tuple[str, Extractor]] = [
    # --- User-requested (config keys mapped to YAML paths) ---
    ("Image resize (H×W)", image_resize_dims),
    ("Number of views", num_views),
    ("Number of keypoints", lambda c: _fmt(_get(c, "data", "num_keypoints"))),
    ("imgaug", lambda c: _fmt(_get(c, "training", "imgaug"))),
    ("imgaug_3d", imgaug_3d),
    ("Train batch size", lambda c: _fmt(_get(c, "training", "train_batch_size"))),
    ("Val batch size", lambda c: _fmt(_get(c, "training", "val_batch_size"))),
    (
        "Max epochs",
        lambda c: _fmt(_get(c, "training", "max_epochs")),
    ),  # "max_steps" in YAML is max_epochs for this project
    (
        "Unfreezing epoch",
        lambda c: _fmt(_get(c, "training", "unfreezing_epoch")),
    ),
    ("LR milestone epochs", milestone_steps),
    ("Learning rate", lambda c: _fmt(_get(c, "training", "optimizer_params", "learning_rate"))),
    ("Patch mask init step", lambda c: patch_mask_field(c, "init_step")),
    ("Patch mask final step", lambda c: patch_mask_field(c, "final_step")),
    ("Patch mask init ratio", lambda c: patch_mask_field(c, "init_ratio")),
    ("Patch mask final ratio", lambda c: patch_mask_field(c, "final_ratio")),
    ("Backbone", lambda c: _fmt(_get(c, "model", "backbone"))),
    ("Model type", lambda c: _fmt(_get(c, "model", "model_type"))),
    ("Head", lambda c: _fmt(_get(c, "model", "head"))),
    (
        "supervised_reprojection_heatmap_mse (log_weight)",
        supervised_repr_heatmap_mse,
    ),
    ("freeze_until_epoch (anneal_weight)", freeze_until_epoch),
    # --- Extra reproducibility / training context ---
    ("Optimizer", lambda c: _fmt(_get(c, "training", "optimizer"))),
    ("LR scheduler γ", lr_gamma),
    ("Downsample factor", lambda c: _fmt(_get(c, "data", "downsample_factor"))),
    ("train_prob / val_prob", lambda c: f"{_fmt(_get(c, 'training', 'train_prob'))} / {_fmt(_get(c, 'training', 'val_prob'))}"),
    ("num_gpus", lambda c: _fmt(_get(c, "training", "num_gpus"))),
    ("num_workers (dataloaders)", lambda c: _fmt(_get(c, "training", "num_workers"))),
    ("rng_seed_data_pt / rng_seed_model_pt", lambda c: f"{_fmt(_get(c, 'training', 'rng_seed_data_pt'))} / {_fmt(_get(c, 'training', 'rng_seed_model_pt'))}"),
    ("heatmap_loss_type", lambda c: _fmt(_get(c, "model", "heatmap_loss_type"))),
    ("early_stop_patience", lambda c: _fmt(_get(c, "training", "early_stop_patience"))),
    ("check_val_every_n_epoch", lambda c: _fmt(_get(c, "training", "check_val_every_n_epoch"))),
]


def dataset_label_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("config_"):
        return stem[len("config_") :]
    return stem


def default_config_paths() -> List[Path]:
    """Primary benchmark configs (edit list if you add datasets)."""
    cfg_dir = LP3D_ROOT / "configs"
    names = [
        "config_chickadee-crop.yaml",
        "config_fly-anipose.yaml",
        "config_ibl-mouse.yaml",
        "config_rat7m-crop.yaml",
        "config_human36m-crop.yaml",
        "config_two-mouse.yaml",
        "config_mirror-mouse-separate.yaml",
    ]
    return [cfg_dir / n for n in names if (cfg_dir / n).is_file()]


def build_table_matrix(
    paths: List[Path],
) -> Tuple[List[str], List[str], List[List[str]]]:
    col_labels = [dataset_label_from_path(p) for p in paths]
    row_labels = [label for label, _ in ROWS]
    matrix: List[List[str]] = []
    for label, fn in ROWS:
        row = []
        for p in paths:
            cfg = load_yaml(p)
            try:
                row.append(fn(cfg))
            except Exception as e:  # noqa: BLE001
                row.append(f"error: {e}")
        matrix.append(row)
    return row_labels, col_labels, matrix


def render_pdf(
    out_path: Path,
    row_labels: List[str],
    col_labels: List[str],
    matrix: List[List[str]],
    *,
    title: str,
    reproduction_notes: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman", "serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            # TrueType fonts in PDF (editable in Illustrator / journal-friendly)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Transpose for matplotlib table: each row is one table row; first col = parameter name
    cell_text: List[List[str]] = []
    for i, rlabel in enumerate(row_labels):
        cell_text.append([rlabel] + matrix[i])

    headers = ["Parameter"] + col_labels
    nrows = len(cell_text)
    ncols = len(headers)

    col_widths = [0.22] + [0.78 / (ncols - 1)] * (ncols - 1)

    with PdfPages(out_path) as pdf:
        fig_w = min(17.0, 4.0 + 1.15 * (ncols - 1))
        fig_h = min(11.0, 1.2 + 0.22 * nrows)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        ax.axis("off")
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

        table = ax.table(
            cellText=cell_text,
            colLabels=headers,
            loc="center",
            cellLoc="left",
            colLoc="center",
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.35)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#333333")
            cell.set_linewidth(0.4)
            if row == 0:
                cell.set_facecolor("#e8e8e8")
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_fontsize(7)
            else:
                face = "#fafafa" if row % 2 == 0 else "#ffffff"
                cell.set_facecolor(face)
            if col == 0 and row > 0:
                cell.get_text().set_fontweight("600")

        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Second page: reproduction / hardware notes
        fig2, ax2 = plt.subplots(figsize=(8.5, 11), dpi=150)
        ax2.axis("off")
        fig2.suptitle("Reproducibility notes", fontsize=12, fontweight="bold", y=0.97)
        body = "\n\n".join(reproduction_notes)
        ax2.text(
            0.06,
            0.93,
            body,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="serif",
            wrap=True,
        )
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--configs",
        nargs="*",
        type=Path,
        default=None,
        help="YAML config paths (default: built-in benchmark list)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "figures" / "dataset_training_hyperparameters.pdf",
        help="Output PDF path",
    )
    p.add_argument("--title", default="Multiview training configuration by dataset")
    p.add_argument(
        "--gpu",
        default="NVIDIA L4",
        help='GPU label for reproducibility page (e.g. "NVIDIA L4")',
    )
    p.add_argument(
        "--extra-note",
        action="append",
        default=[],
        help="Additional bullet for reproducibility page (repeatable)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    paths = args.configs if args.configs else default_config_paths()
    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"Missing config: {p}", file=sys.stderr)
        return 1

    row_labels, col_labels, matrix = build_table_matrix(paths)

    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    repro_lines = [
        f"Generated: {gen_time}",
        f"Source configs: {len(paths)} files under `lp3d-analysis/configs/` (see table column headers).",
        f"Hardware (reported): {args.gpu}, single-GPU training (`training.num_gpus: 1` unless overridden in a config).",
        "Framework versions: record `python -V`, PyTorch, CUDA driver, and Lightning Pose commit in the paper supplement.",
        "This table is parsed directly from the checked-in YAML; any typo keys (e.g. `imagug_3d`) are read with a fallback so the PDF reflects the file on disk.",
    ]
    repro_lines.extend(args.extra_note)

    render_pdf(
        args.out.resolve(),
        row_labels,
        col_labels,
        matrix,
        title=args.title,
        reproduction_notes=repro_lines,
    )
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
