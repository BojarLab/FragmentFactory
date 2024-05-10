import copy
import pickle
from typing import Literal, Optional

import numpy as np
import pandas as pd
from datasail.sail import datasail
from glycowork.motif.tokenization import structure_to_basic
from sklearn import tree
from sklearn.metrics import accuracy_score


class TopoData:
    def __init__(self, spec_df, weighting: bool = True, GPID_SIM: float = 0.8):
        # Extract the most important data from the spectra dataframe to a new dataframe
        self.df = pd.DataFrame({
            "ID": [f"G{i:05d}" for i in range(len(spec_df))],
            "X": list(np.stack(spec_df["binned_intensities_norm"].values)),
            "y": spec_df["glycan"],
            "filename": spec_df["filename"],
            "GP_ID": spec_df["GlycoPost_ID"],
        })

        # remove everything that cannot be usefully split
        _vc = dict(self.df["y"].value_counts())
        self.df = self.df[[_vc[x] >= 3 and "?" not in x for x in self.df["y"]]]

        # Calculate the topology
        _topo_map = {g: structure_to_basic(g) for g in self.df["y"].unique()}
        self.df["topo_y"] = self.df["y"].apply(lambda x: _topo_map[x])

        # Calculate similarity matrix
        _sim = np.zeros((len(self.df), len(self.df)))
        _tmp = [list(x) for x in self.df[["filename", "GP_ID"]].values]
        for i in range(len(_tmp)):
            for j in range(i + 1, len(_tmp)):
                _sim[i, j] = 1 if _tmp[i][0] == _tmp[j][1] else (GPID_SIM if _tmp[i][1] == _tmp[j][1] else 0)
                _sim[j, i] = _sim[i, j]

        # Individual splits per topology as we grow one tree per topology
        _splits = {}
        for _topo in self.df["topo_y"].unique():
            _mask = self.df["topo_y"] == _topo
            try:
                _e_splits, _, _ = datasail(
                    techniques=["C1e"],
                    splits=[0.7, 0.2, 0.1],
                    names=["train", "val", "test"],
                    e_type="O",
                    e_data=dict(self.df[_mask][["ID", "X"]].values.tolist()),
                    e_strat=dict(self.df[_mask][["ID", "topo_y"]].values.tolist()),
                    e_sim=(self.df[_mask]["ID"].values.tolist(), _sim[_mask, :][:, _mask]),
                )
                _splits.update(_e_splits["C1e"][0])
            except Exception as e:
                # report failure
                print(f"Error for {_topo}: {e}")
                if _mask is not None:
                    print(f"Mask: {_mask.sum()} ({_mask.sum()}/{len(_mask)})")
        self.df["split"] = self.df["ID"].apply(lambda x: _splits.get(x, None))

        # remove everything that cannot be usefully split and update the similarity matrix
        _mask = self.df["split"].isna()
        print("Masking out", _mask.sum(), "entries")
        self.df = self.df[~_mask]
        _sim = _sim[~_mask, :][:, ~_mask]

        # split all topologies
        _e_splits, _, _ = datasail(
            techniques=["C1e"],
            splits=[0.7, 0.2, 0.1],
            names=["train", "val", "test"],
            e_type="O",
            e_data=dict(self.df[["ID", "X"]].values.tolist()),
            e_strat=dict(self.df[["ID", "y"]].values.tolist()),
            e_sim=(self.df["ID"].values.tolist(), _sim),
        )
        self.df["topo_split"] = self.df["ID"].apply(lambda x: _e_splits["C1e"][0][x])

        # Calculate the weights
        _vc = dict(self.df["y"].value_counts())
        _topo_vc = dict(self.df["topo_y"].value_counts())
        self.weights = {}

        if weighting:
            # TODO: weights have to be calculated for each topology separately
            for _topo in _topo_vc.keys():
                _vc = dict(self.df[self.df["topo_y"] == _topo]["y"].value_counts())
                count = len(self.df[(self.df["split"] == "train") & (self.df["topo_y"] == _topo)])
                self.weights[_topo] = {k: count / v for k, v in _vc.items()}
            self.topo_weights = {k: len(self.df[self.df["split"] == "train"]) / v for k, v in _topo_vc.items()}
        else:
            self.weights = {_topo: {k: 1 for k in dict(self.df[self.df["topo_y"] == _topo]["y"].value_counts()).keys()}
                            for _topo in _topo_vc.keys()}
            self.topo_weights = {k: 1 for k in _topo_vc.keys()}

        self.df["w"] = self.df[["topo_y", "y"]].apply(lambda row: self.weights[row["topo_y"]][row["y"]], axis=1)
        self.df["topo_w"] = self.df["topo_y"].apply(lambda x: self.topo_weights[x])

    def __call__(self, split: Literal["train", "val", "test"], feat: Literal["X", "y", "w"], topo: str, **kwargs):
        """

        :param split:
        :param feat:
        :param topo: either "topo" or some actual topology from topo_y
        :param kwargs:
        :return:
        """
        if topo == "topo":
            out = self.df[self.df["topo_split"] == split][("" if feat == "X" else "topo_") + feat].values
        else:
            out = self.df[(self.df["split"] == split) & (self.df["topo_y"] == topo)][feat].values
        if feat == "X":
            return np.stack(out)
        return out


class TopoTree:
    def __init__(self):
        self.topo_tree = None
        self.isomer_trees = {}
        self.topo_classes_ = []
        self.classes_ = []

    def fit(self, data: TopoData):
        print("Training topological tree...")
        self.topo_tree = self._fit_tree(
            data("train", "X", "topo"),
            data("train", "y", "topo"),
            data.topo_weights,
            data("val", "X", "topo"),
            data("val", "y", "topo"),
            data("val", "w", "topo"),
        )
        self.topo_classes_ = self.topo_tree.classes_
        self.classes_ = []
        for topo in self.topo_classes_:
            print("Training isomer tree for", topo)
            self.isomer_trees[topo] = self._fit_tree(
                data("train", "X", topo),
                data("train", "y", topo),
                data.weights[topo],
                data("val", "X", topo),
                data("val", "y", topo),
                data("val", "w", topo),
            )
            print(topo, self.isomer_trees[topo].classes_)
            self.classes_ += list(self.isomer_trees[topo].classes_)
        self.classes_ = np.array(self.classes_)
        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        topo_probs = self.topo_tree.predict_proba(X)
        return np.stack([
            p for topo, p in zip(self.topo_classes_, topo_probs) for p in self.isomer_trees[topo].predict_proba(X)
        ])

    def get_params(self, deep=True):
        topo_dics = {f"topo_{k}": v for k, v in self.topo_tree.get_params(deep).items()}
        isomer_dics = {f"{k}_{kk}": vv for k, v in self.isomer_trees.items() for kk, vv in v.get_params(deep).items()}
        return {**topo_dics, **isomer_dics}

    def set_params(self, **params):
        topo_params = {k[5:]: v for k, v in params.items() if k.startswith("topo_")}
        isomer_params = {k.split("_")[0]: {kk: vv for kk, vv in v.items()} for k, v in params.items() if
                         k.split("_")[0] in self.isomer_trees.keys()}
        self.topo_tree.set_params(**topo_params)
        for k, v in isomer_params.items():
            self.isomer_trees[k].set_params(**v)
        return self

    @staticmethod
    def _fit_tree(train_X, train_y, train_w: dict, val_X, val_y, val_w):
        max_depth = int(np.ceil(np.log2(len(train_X)) * 1.5))
        best_acc, best_tree = 0, None
        for criterion in ["gini", "entropy", "log_loss"]:
            for max_depth in list(range(1, max_depth)):
                classifier = tree.DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    class_weight=train_w,
                ).fit(train_X, train_y)
                val_p = classifier.predict(val_X)
                acc = accuracy_score(val_y, val_p, sample_weight=val_w)
                if acc > best_acc:
                    best_acc = acc
                    best_tree = copy.deepcopy(classifier)
        return best_tree


def train_topo_tree(in_filename, out_filename: Optional[str] = None, weighting: bool = True, GPID_SIM: float = 0.8):
    if in_filename.endswith(".pkl"):
        with open(in_filename, "rb") as f:
            spec_df = pickle.load(f)
    elif in_filename.endswith(".csv"):
        spec_df = pd.read_csv("data/spectra.csv")
    elif in_filename.endswith(".tsv"):
        spec_df = pd.read_csv("data/spectra.tsv", sep="\t")
    else:
        raise ValueError("Unknown file format")

    # Train the tree
    data = TopoData(spec_df, weighting=weighting, GPID_SIM=GPID_SIM)
    tree = TopoTree().fit(data)

    # Save the tree
    if out_filename is not None:
        with open("data/topo_tree.pkl", "wb") as f:
            pickle.dump(tree, f)
    return tree
