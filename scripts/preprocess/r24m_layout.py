#!/usr/bin/env python3
"""Path and naming conventions for the two-rat r24m dataset.

r24m and two-mouse are the same *kind* of dataset -- two animals, five views,
one crop per animal -- but they spell the session directory differently::

    two-mouse_s   labeled-data/<session>_<view>_<animal>/img.jpg
    r24m_s        labeled-data/<session>_<animal>_<view>/img.png

Nothing downstream should have to care where in the name the animal sits, so the
parsing lives here rather than being re-derived as a ``rsplit`` in each script.
The animal and view are matched as whole underscore-separated tokens against the
known vocabularies, which is what makes position irrelevant: no session name in
this dataset contains a bare ``an1``/``an2`` or ``CameraN`` token.
"""

from __future__ import annotations

from pathlib import PurePosixPath

# Camera order is load-bearing: it must match the `name` order inside the
# calibration tomls, which lightning_pose asserts against cfg.data.view_names.
VIEWS = ["Camera1", "Camera2", "Camera3", "Camera5", "Camera6"]
ANIMALS = ["an1", "an2"]
KEYPOINTS = [
    "HeadF", "HeadB", "HeadL",
    "SpineF", "SpineM", "SpineL",
    "Offset1", "Offset2",
    "HipL", "HipR",
    "ShoulderL", "ShoulderR",
]
SPLITS = {"ind": "", "ood": "_new"}
HEADER_ROWS = [0, 1, 2]


def parse_crop_dir(dir_name: str) -> tuple[str, str, str]:
    """``<session>_<animal>_<view>`` -> ``(session, animal, view)``."""
    tokens = dir_name.split("_")
    animal_at = [i for i, t in enumerate(tokens) if t in ANIMALS]
    view_at = [i for i, t in enumerate(tokens) if t in VIEWS]
    if len(animal_at) != 1 or len(view_at) != 1:
        raise ValueError(
            f"cannot parse {dir_name!r}: expected exactly one animal token from "
            f"{ANIMALS} and one view token from {VIEWS}"
        )
    a, v = animal_at[0], view_at[0]
    session = "_".join(t for i, t in enumerate(tokens) if i not in (a, v))
    return session, tokens[a], tokens[v]


def parse_full_dir(dir_name: str) -> tuple[str, str]:
    """``<session>_<view>`` -> ``(session, view)`` for the uncropped frames."""
    tokens = dir_name.split("_")
    view_at = [i for i, t in enumerate(tokens) if t in VIEWS]
    if len(view_at) != 1:
        raise ValueError(f"cannot parse {dir_name!r}: expected one view token")
    v = view_at[0]
    return "_".join(t for i, t in enumerate(tokens) if i != v), tokens[v]


def session_of(dir_name: str) -> str:
    """Session with the animal and view tokens removed, whichever are present."""
    tokens = dir_name.split("_")
    return "_".join(t for t in tokens if t not in ANIMALS and t not in VIEWS)


def animal_of(path: str) -> str | None:
    """The animal a cropped row belongs to, or None if the path carries none."""
    tokens = PurePosixPath(path).parent.name.split("_")
    found = [t for t in tokens if t in ANIMALS]
    return found[0] if len(found) == 1 else None


def instance_path(img_path: str, animal: str) -> str:
    """``labeled-data/<session>_<view>/img.png`` -> ``..._<animal>_<view>/img.png``.

    The inverse of :func:`parse_crop_dir`, and the bridge from an uncropped frame
    to the cropped row it produces. The animal goes before the view so the result
    matches the directories as delivered.
    """
    p = PurePosixPath(img_path)
    session, view = parse_full_dir(p.parent.name)
    return str(p.parent.with_name(f"{session}_{animal}_{view}") / p.name)


def full_path(crop_rel: str) -> str:
    """``labeled-data/<session>_<animal>_<view>/img.png`` -> the uncropped frame."""
    p = PurePosixPath(crop_rel)
    session, _animal, view = parse_crop_dir(p.parent.name)
    return str(p.parent.with_name(f"{session}_{view}") / p.name)
