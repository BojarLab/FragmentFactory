import argparse
import copy
import os
import pickle
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd
from datasail.sail import datasail
from glycowork.motif.tokenization import structure_to_basic
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from visualize import DOTTreeExporter


def split(spec_df, GPID_SIM=0.8):
    df = pd.DataFrame({
        "ID": [f"G{i:05d}" for i in range(len(spec_df))],
        "X": list(np.stack(spec_df["binned_intensities_norm"].values)),
        "y": spec_df["glycan"],
        "filename": spec_df["filename"],
        "GP_ID": spec_df["GlycoPost_ID"],
    })

    # remove everything that cannot be usefully split
    _vc = dict(df["y"].value_counts())
    df = df[[_vc[x] >= 3 and "?" not in x for x in df["y"]]]

    # Calculate the topology
    _topo_map = {g: structure_to_basic(g) for g in df["y"].unique()}
    df["topo_y"] = df["y"].apply(lambda x: _topo_map[x])

    _sim = np.zeros((len(df), len(df)))
    _tmp = [list(x) for x in df[["filename", "GP_ID"]].values]
    for i in range(len(_tmp)):
        for j in range(i + 1, len(_tmp)):
            _sim[i, j] = 1 if _tmp[i][0] == _tmp[j][1] else (GPID_SIM if _tmp[i][1] == _tmp[j][1] else 0)
            _sim[j, i] = _sim[i, j]

    splits = {}
    results = {}
    for _topo in df["topo_y"].unique():
        _mask = df["topo_y"] == _topo
        print("Splitting for ", _topo, "\n", df[_mask]["y"].unique())
        _e_splits, _, _ = datasail(
            techniques=["C1e"],
            splits=[0.7, 0.2, 0.1],
            names=["train", "val", "test"],
            e_type="O",
            e_data=dict(df[_mask][["ID", "X"]].values.tolist()),
            e_strat=dict(df[_mask][["ID", "y"]].values.tolist()),
            e_sim=(df[_mask]["ID"].values.tolist(), _sim[_mask, :][:, _mask]),
        )
        results[_topo] = copy.deepcopy(_e_splits["C1e"][0])
        splits.update(_e_splits["C1e"][0])
    df["split"] = df["ID"].apply(lambda x: splits.get(x, None))

    _mask = df["split"].isna()
    df = df[~_mask]

    form = np.ascontiguousarray if _sim.flags['C_CONTIGUOUS'] else np.asfortranarray
    _sim = _sim[~_mask, :][:, ~_mask]
    _sim = form(_sim)

    e_splits, _, _ = datasail(
        techniques=["C1e"],
        splits=[0.7, 0.2, 0.1],
        names=["train", "val", "test"],
        e_type="O",
        e_data=dict(df[["ID", "X"]].values.tolist()),
        e_strat=dict(df[["ID", "topo_y"]].values.tolist()),
        e_sim=(df["ID"].values.tolist(), _sim),
    )
    df["topo_split"] = df["ID"].apply(lambda x: e_splits["C1e"][0][x])

    return df


class TopoData:
    def __init__(self, spec_df, weighting: bool = True, GPID_SIM: float = 0.8):
        # Extract the most important data from the spectra dataframe to a new dataframe
        self.df = split(spec_df, GPID_SIM=GPID_SIM)

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
        max_depth = max(int(np.ceil(np.log2(len(train_w)) * 1.5)), 1)
        best_acc, best_tree = 0, None
        # for criterion in ["gini", "entropy", "log_loss"]:
        for max_depth in list(range(1, max_depth + 1)):
            classifier = DecisionTreeClassifier(
                criterion="entropy",
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
    return data, tree


def dot2svg(output_path_prefix):
    # pydot.graph_from_dot_file(topo_tree_name + ".dot")[0].write_svg(topo_tree_name + ".svg")
    os.system(f"dot -Tsvg{':cairo' if sys.platform.lower().startswith('linux') else ''} {output_path_prefix}.dot -o {output_path_prefix}.svg")


def spec2svg(in_filename, output_path_prefix: str, weighting: bool = True, GPID_SIM: float = 0.8):
    data, tree = train_topo_tree(in_filename, weighting=weighting, GPID_SIM=GPID_SIM)

    topo_tree_name = f"{output_path_prefix}_topo_tree".encode('ascii', 'replace').decode('ascii')

    # Export the topology tree
    isomer_map = data.df.groupby('topo_y')['y'].apply(set).to_dict()
    DOTTreeExporter(out_file=topo_tree_name + ".dot").export(
        tree.topo_tree,
        class_weights=data.topo_weights,
        total=dict(np.asarray(np.unique(data("train", "y", "topo"), return_counts=True)).T),
        topo=None,
        isomer_map=isomer_map,
        val_X=data("val", "X", "topo"),
        val_y=data("val", "y", "topo"),
        test_X=data("test", "X", "topo"),
        test_y=data("test", "y", "topo"),
    )
    dot2svg(topo_tree_name)

    for i, topo in enumerate(tree.topo_classes_):
        if len(tree.isomer_trees[topo].classes_) == 1:
            continue
        iso_tree_name = f"{output_path_prefix}_isomer_tree_{i}".encode('ascii', 'replace').decode('ascii')
        DOTTreeExporter(out_file=iso_tree_name + ".dot").export(
            tree.isomer_trees[topo],
            class_weights=data.weights[topo],
            total=dict(np.asarray(np.unique(data("train", "y", topo), return_counts=True)).T),
            topo=topo,
            val_X=data("val", "X", topo),
            val_y=data("val", "y", topo),
            test_X=data("test", "X", topo),
            test_y=data("test", "y", topo),
        )
        dot2svg(iso_tree_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_filename", type=str, help="Path to the pickled dataframe containing the spectra.")
    parser.add_argument("output_path_prefix", type=str, help="Prefix for the output files.")
    parser.add_argument("--weighting", action="store_true", default=False, help="Whether to use class weights.")
    parser.add_argument("--GPID-SIM", dest="GPID_SIM", type=float, default=0.8,
                        help="Similarity threshold for spectra with the same GlycoPostID.")
    args = parser.parse_args()
    spec2svg(args.in_filename, args.output_path_prefix, weighting=args.weighting, GPID_SIM=args.GPID_SIM)
    # spec2svg("spectra_for_roman/733_isomer_spectra.pkl", "733", weighting=True, GPID_SIM=0.8)
