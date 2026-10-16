"""
Load one Allen Brain Observatory natural-scenes experiment, keeping every trial.

Returns images as float32 (n_images, 224, 224, 3) in [0, 1] and responses as
float32 (n_images, n_trials, n_neurons), with NaN where an image had fewer
presentations than the maximum. The trial dimension is what the noise ceiling
in encoding.py needs; the old loader averaged it away on load.

Response per presentation: mean dF/F over the stimulus frames shifted by
`lag_frames` to allow for the calcium indicator's delay (the Allen natural
scenes stimulus lasts 250 ms, about 7 frames at 30 Hz).
"""

import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
except ImportError:  # the rest of the repo can still be tested without the SDK
    BrainObservatoryCache = None


def pick_experiment(boc, targeted_structures: List[str], cre_lines: Optional[List[str]] = None,
                    stimuli: List[str] = ["natural_scenes"]) -> Dict:
    """First experiment matching the filters, after sorting by cell count (most first)."""
    kwargs = dict(targeted_structures=targeted_structures, stimuli=stimuli)
    if cre_lines:
        kwargs["cre_lines"] = cre_lines
    exps = boc.get_ophys_experiments(**kwargs)
    if not exps:
        raise ValueError(f"No experiments for {kwargs}")
    exps = sorted(exps, key=lambda e: -len(boc.get_ophys_experiment_data(e["id"]).get_cell_specimen_ids()))
    return exps[0]


def load_experiment(cache_dir: str = "./allen_cache", targeted_structures: List[str] = ["VISp"],
                    cre_lines: Optional[List[str]] = None, experiment_id: Optional[int] = None,
                    lag_frames: int = 2, image_size: int = 224) -> Dict:
    if BrainObservatoryCache is None:
        raise ImportError("allensdk is required: pip install allensdk")
    os.makedirs(cache_dir, exist_ok=True)
    boc = BrainObservatoryCache(manifest_file=os.path.join(cache_dir, "manifest.json"))

    if experiment_id is None:
        exp = pick_experiment(boc, targeted_structures, cre_lines)
        experiment_id = exp["id"]
    data = boc.get_ophys_experiment_data(experiment_id)
    meta = data.get_metadata()

    template = data.get_stimulus_template("natural_scenes")  # (118, 918, 1174) uint8
    table = data.get_stimulus_table("natural_scenes")
    dff = data.get_dff_traces()[1]  # (n_neurons, n_frames)

    scenes = sorted(int(s) for s in table["frame"].unique() if s >= 0)
    max_trials = int(table[table["frame"] >= 0].groupby("frame").size().max())
    n_neurons = dff.shape[0]

    images = np.zeros((len(scenes), image_size, image_size, 3), dtype=np.float32)
    responses = np.full((len(scenes), max_trials, n_neurons), np.nan, dtype=np.float32)
    for i, scene in enumerate(scenes):
        img = Image.fromarray(template[scene].astype(np.uint8)).convert("RGB").resize((image_size, image_size))
        images[i] = np.asarray(img, dtype=np.float32) / 255.0
        rows = table[table["frame"] == scene]
        for t, (_, row) in enumerate(rows.iterrows()):
            start, end = int(row["start"]) + lag_frames, int(row["end"]) + lag_frames
            responses[i, t] = dff[:, start:end].mean(axis=1)

    return {
        "images": images,
        "responses_trials": responses,
        "responses": np.nanmean(responses, axis=1).astype(np.float32),
        "scene_ids": np.array(scenes),
        "metadata": {
            "experiment_id": int(experiment_id),
            "targeted_structure": meta.get("targeted_structure"),
            "cre_line": meta.get("cre_line"),
            "imaging_depth_um": meta.get("imaging_depth_um"),
            "n_neurons": int(n_neurons),
            "n_images": len(scenes),
            "max_trials": max_trials,
            "lag_frames": lag_frames,
            "note": "stimulus images are grayscale and were resized from 918x1174 to a square; aspect ratio is not preserved",
        },
    }
