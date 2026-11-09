"""Builds the Kaggle dataset `ritisingh123/engs108-submission-h5-artifacts` and the shared-pass dev reference.

  ENGS108_DATA_DIR=~/data/nthl ARMH_OUT_DIR=~/data/armh_ds python submission/h5/make_artifacts_dataset.py OUT_DIR

ARMH_OUT_DIR holds `pooled_hidden_armh.npy` from the engs108-exp03-armh-artifacts dataset (sha-checked against
exp03/qlora_armh/artifacts/kernel_output_SHA256SUMS). CPU only; nothing is fitted.

artifacts/h5_dev_reference.csv = H5 recomputed from the Arm H kernel's recorded *shared* pooled pass (head logits
from armh_pooled_pass_logits.parquet + probe on pooled_hidden_armh.npy of the same pass) + stylo joblib. It differs
from blend_H5_dev.csv only in the head component, which analyze_armh.py took from the separate length-sorted pass.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "exp03" / "stylo"))
sys.path.insert(0, str(REPO / "exp03" / "probe"))
import h5_core as C  # noqa: E402
import protocol as P  # noqa: E402
from probe import ProbeModel  # noqa: E402

ARMH_ART = REPO / "exp03" / "qlora_armh" / "artifacts"
STYLO = REPO / "exp03" / "stylo" / "artifacts" / "stylo_exp03_model.joblib"
CFG = json.loads((HERE / "config.json").read_text())
LOGIT_COLS = [f"logit_{n}" for n in P.CLASS_NAMES]


def reference_frame(dev_df, comps, blend):
    out = dev_df[P.META_COLS].reset_index(drop=True).copy()
    parts = [out]
    for k in C.H5_COMPONENTS:
        parts.append(pd.DataFrame(comps[k], columns=[f"{k}:{c}" for c in P.PROBA_COLS]))
    parts.append(pd.DataFrame(blend, columns=[f"h5:{c}" for c in P.PROBA_COLS]))
    return pd.concat(parts, axis=1)


def main(out_dir):
    out_dir = Path(out_dir)
    armh_out = Path(os.environ.get("ARMH_OUT_DIR", Path.home() / "data" / "armh_ds")).expanduser()
    sums = dict(l.split(maxsplit=1)[::-1] for l in (ARMH_ART / "kernel_output_SHA256SUMS").read_text().splitlines())
    assert P.sha256_file(armh_out / "pooled_hidden_armh.npy") == sums["pooled_hidden_armh.npy"]
    for fn, path in [("probe_armh.npz", ARMH_ART / "probe_armh.npz"), ("stylo_exp03_model.joblib", STYLO),
                     ("blend_H5_dev.csv", ARMH_ART / "blend_H5_dev.csv")]:
        assert P.sha256_file(path) == CFG["sha256"][fn], fn

    _, dev_df = P.load_data()
    pm = pd.read_parquet(ARMH_ART / "armh_pooled_pass_logits.parquet")
    pos = {k: i for i, k in enumerate(pm["id"])}
    rows = [pos[k] for k in dev_df["id"]]
    H = np.load(armh_out / "pooled_hidden_armh.npy")[rows]  # float16, as stored by the kernel
    logits = pm[LOGIT_COLS].to_numpy(np.float32)[rows]
    probe = ProbeModel(ARMH_ART / "probe_armh.npz")
    stylo = joblib.load(STYLO)
    comps = C.component_probas(H, logits, dev_df["text"].tolist(), probe, stylo)
    blend = C.h5_blend(comps)
    y = dev_df["labels"].to_numpy()
    ref_h5 = P.read_predictions(ARMH_ART / "blend_H5_dev.csv", dev_df)
    rep, fails = C.dev_reproduction(
        y, blend, comps, ref_h5, blend,
        {"probe_armh": P.read_predictions(ARMH_ART / "probe_armh_dev.csv", dev_df),
         "armh_head": P.read_predictions(ARMH_ART / "armh_head_dev.csv", dev_df),
         "stylo_exp03": P.read_predictions(REPO / "exp03" / "stylo" / "artifacts" / "stylo_exp03_dev.csv", dev_df)},
        CFG["dev_check"]["expected_balanced_accuracy"], CFG["dev_check"]["tolerance"], 0.0)
    print(json.dumps(rep, indent=1))
    assert not fails, fails

    ref_path = HERE / "artifacts" / "h5_dev_reference.csv"
    reference_frame(dev_df, comps, blend).to_csv(ref_path, index=False, float_format="%.10g")
    print("wrote", ref_path, P.sha256_file(ref_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in [ARMH_ART / "probe_armh.npz", STYLO, ARMH_ART / "blend_H5_dev.csv", ref_path]:
        shutil.copy2(path, out_dir / path.name)
    files = sorted(p.name for p in out_dir.iterdir() if p.name not in ("SHA256SUMS", "dataset-metadata.json"))
    (out_dir / "SHA256SUMS").write_text("".join(f"{P.sha256_file(out_dir / f)}  {f}\n" for f in files))
    (out_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "engs108-submission-h5-artifacts", "id": CFG["kaggle"]["artifacts_dataset"],
        "licenses": [{"name": "CC0-1.0"}]}, indent=2) + "\n")
    print((out_dir / "SHA256SUMS").read_text())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "kaggle_h5_artifacts")
