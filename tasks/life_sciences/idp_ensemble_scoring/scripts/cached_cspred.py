"""Validated batch inference using the supplied CSpred feature and model APIs."""

from pathlib import Path

import numpy as np
import pandas as pd


class CachedUCBShiftX:
    """Reuse unchanged R0/R1 models; never invoke transfer prediction or fit."""

    def __init__(self, cspred, model_dir=None, prediction_jobs=4):
        self.cspred = cspred
        self.model_dir = Path(model_dir or cspred.ML_MODEL_PATH)
        self.models = {}
        self.prediction_jobs = prediction_jobs

    def load_models(self):
        for atom in self.cspred.toolbox.ATOMS:
            for stage in ("R0", "R1"):
                key = (atom, stage)
                if key not in self.models:
                    self.models[key] = self.cspred.joblib.load(
                        str(self.model_dir / "{}_{}.sav".format(atom, stage))
                    )
                    self.models[key].n_jobs = self.prediction_jobs

    def predict_features(self, frames):
        """Return one stock-format prediction frame per independent conformer."""
        if not frames or any(frame.empty for frame in frames):
            raise ValueError("Every conformer must have nonempty features")
        self.load_models()
        lengths = [len(frame) for frame in frames]
        features = pd.concat(frames, ignore_index=True)
        features.rename(index=str, columns=self.cspred.sparta_rename_map, inplace=True)
        resnames = features["RESNAME"]
        resnums = features["RES_NUM"]
        coils = features[self.cspred.rcoil_cols]
        processed = self.cspred.data_preprocessing(features)
        result = {"RESNUM": resnums, "RESNAME": resnames}
        for atom in self.cspred.toolbox.ATOMS:
            atom_features = self.cspred.prepare_data_for_atom(processed, atom)
            initial = self.models[(atom, "R0")].predict(atom_features.values)
            refined_features = atom_features.copy()
            refined_features["R0_PRED"] = initial
            refined = self.models[(atom, "R1")].predict(refined_features.values)
            result[atom + "_X"] = refined + coils["RCOIL_" + atom]
        predictions = pd.DataFrame(result)
        split = []
        offset = 0
        for length in lengths:
            split.append(predictions.iloc[offset : offset + length].reset_index(drop=True))
            offset += length
        return split


def experimental_shifts(predictions, experiment):
    """Select shifts by residue identity and atom in experimental row order."""
    if predictions["RESNUM"].duplicated().any():
        raise ValueError("Duplicate predicted residue identifiers")
    indexed = predictions.set_index("RESNUM")
    values = np.array(
        [indexed.loc[int(row.resnum), row.atomname + "_X"] for row in experiment.itertuples()],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite experimentally requested shifts")
    return values
