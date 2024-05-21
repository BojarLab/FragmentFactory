import re
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
from CandyCrunch.analysis import CandyCrumbs, domon_costello_to_fragIUPAC
from glycowork.motif.draw import GlycoDraw
from sklearn.tree import _tree
from sklearn.tree._export import _BaseTreeExporter

min_mz_ = 39.714
max_mz_ = 3000
bin_num = 2048
step = (max_mz_ - min_mz_) / (bin_num - 1)
frames = np.array([min_mz_ + step * i for i in range(bin_num)])


def norm_criterion(criterion, crit, **kwargs):
    if criterion == "entropy":
        return crit / np.sqrt(len(kwargs["classes"]))
    elif criterion == "gini":
        raise ValueError("Gini-Index cannot be normalized.")
    else:
        raise ValueError("Log-Loss cannot be normalized.")


def clean_highlights(highlights):
    best = {}
    for _, _, conf, cov, iso in highlights:
        if iso not in best:
            best[iso] = 0
        best[iso] = max(best[iso], conf * cov)

    return {(a, b): (conf, cov, iso) for a, b, conf, cov, iso in highlights if conf * cov >= best[iso] * 0.9}


class DOTTreeExporter(_BaseTreeExporter):
    def __init__(self, out_file=Union[str, Path], leaves_parallel=False, special_characters=False,
                 fontname="helvetica", ):
        super().__init__(
            max_depth=None,
            feature_names=None,
            class_names=None,
            label="all",
            filled=False,
            impurity=True,
            node_ids=False,
            proportion=False,
            rounded=False,
            precision=3,
        )
        self.leaves_parallel = leaves_parallel
        self.out_file = open(out_file, "w", encoding="utf-8")
        self.special_characters = special_characters
        self.fontname = fontname

        # PostScript compatibility for special characters
        if special_characters:
            self.characters = ["&#35;", "<SUB>", "</SUB>", "&le;", "<br/>", ">", "<"]
        else:
            self.characters = ["#", "[", "]", "<=", "\\n", '"', '"']

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

    @staticmethod
    def create_snfg(isomer, **kwargs):
        if isomer not in kwargs["imgs"]:
            (p := Path("imgs")).mkdir(exist_ok=True)
            img_name = re.sub(r'[-?]', "", isomer)
            img = (re.sub(r"[{ \-}]", "?", str((p / f"{img_name}.svg").absolute()).replace("\\", "/"))
                   .encode('ascii', 'replace')
                   .decode('ascii')
                   .replace("?", "X"))

            # when changing the output file format, also change the file extension in the line above
            GlycoDraw(isomer, suppress=True).save_svg(img)
            GlycoDraw(isomer, filepath=img[:-4] + ".pdf", suppress=True)

            kwargs["imgs"][isomer] = img
        return kwargs["imgs"][isomer]

    @staticmethod
    def compute_node_strings(node_id, val_path, val_y, test_path, test_y, values, **kwargs):
        maxdex = values.index(max(values))
        isomer = kwargs["classes"][maxdex]

        val_values = []
        for _cls in kwargs["classes"]:
            val_values += [sum(val_y[val_path[node_id] == 1] == _cls)]

        test_values = []
        for _cls in kwargs["classes"]:
            test_values += [sum(test_y[test_path[node_id] == 1] == _cls)]

        train_confidence = (values[maxdex] / kwargs['weights'][kwargs['classes'][maxdex]]) / sum([v / kwargs['weights'][c] for v, c in zip(values, kwargs['classes'])])
        train_coverage = (values[maxdex] / kwargs['weights'][isomer]) / kwargs['counts'][isomer]

        val_confidence = (val_values[maxdex] / kwargs['weights'][kwargs['classes'][maxdex]]) / sum([v / kwargs['weights'][c] for v, c in zip(val_values, kwargs['classes'])])
        val_coverage = (val_values[maxdex] / kwargs['weights'][isomer]) / sum(val_y == isomer)

        test_confidence = (test_values[maxdex] / kwargs['weights'][kwargs['classes'][maxdex]]) / sum([v / kwargs['weights'][c] for v, c in zip(test_values, kwargs['classes'])])
        test_coverage = (test_values[maxdex] / kwargs['weights'][isomer]) / sum(test_y == isomer)

        return (
            f"{isomer}<br/>"
            f"===== Train =====<br/>"
            f"Confidence: {train_confidence:.1%}<br/>"
            f"Coverage: {train_coverage:.1%}<br/>"
            f"===== Val =====<br/>"
            f"Confidence: {val_confidence:.1%}<br/>"
            f"Coverage: {val_coverage:.1%}<br/>"
            f"===== Test =====<br/>"
            f"Confidence: {test_confidence:.1%}<br/>"
            f"Coverage: {test_coverage:.1%}"
        ), isomer

    def export(self, decision_tree, class_weights, total, val_X, val_y, test_X, test_y, topo=None, isomer_map=None):
        kwargs = {
            "classes": decision_tree.classes_,
            "weights": class_weights,
            "counts": total,
            "topo": topo,
            "imgs": {"q": r"{}".format(Path("q.svg").absolute())},
            "isomer_map": isomer_map,
        }

        kwargs["highlights"] = clean_highlights(
            self.find_bold(decision_tree.tree_, 0, decision_tree.criterion, **kwargs))
        val_path = decision_tree.decision_path(val_X).toarray().transpose()
        test_path = decision_tree.decision_path(test_X).toarray().transpose()

        # each part writes to out_file
        self.head()

        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            self.recurse(decision_tree, 0, "impurity", val_path, val_y, test_path, test_y, **kwargs)
        else:
            self.recurse(decision_tree.tree_, 0, decision_tree.criterion, val_path, val_y, test_path, test_y, **kwargs)

        self.tail()

    def tail(self):
        # If required, draw leaf nodes at same depth as each other
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write(
                    "{rank=same ; " + "; ".join(r for r in self.ranks[rank]) + "} ;\n"
                )
        self.out_file.write("}")

    def head(self):
        self.out_file.write("digraph Tree {\n")

        # Specify node aesthetics
        self.out_file.write('node [shape=box, fontname="%s"] ;\n' % self.fontname)

        # Specify graph & edge aesthetics
        if self.leaves_parallel:
            self.out_file.write("graph [ranksep=equally, splines=polyline] ;\n")

        self.out_file.write('edge [fontname="%s"] ;\n' % self.fontname)

    def find_bold(self, tree, node_id, criterion, **kwargs) -> Optional[Union[dict, Tuple]]:
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            node_str = self.node_to_str(tree, node_id, criterion)
            crit, _, value = node_str[1:-1].split("\\n")[:3]
            crit = float(crit.split(" = ")[1])
            crit = norm_criterion(criterion, crit, **kwargs)
            if crit < 0.7:
                print(value)
                values = [float(x) for x in value.split(" = ")[1][1:-1].split(", ")]
                maxdex = values.index(max(values))
                confidence = (max(values) / kwargs["weights"][kwargs["classes"][maxdex]]) / sum(
                    [v / kwargs["weights"][c] for v, c in zip(values, kwargs["classes"])])
                if confidence >= 0.8:
                    return {"cov": (max(values) / kwargs['weights'][kwargs["classes"][maxdex]]) / kwargs['counts'][
                        kwargs["classes"][maxdex]], "conf": confidence, "iso": maxdex}
        else:
            output = []
            for child in [tree.children_left[node_id], tree.children_right[node_id]]:
                result = self.find_bold(tree, child, criterion, **kwargs)
                if isinstance(result, dict):
                    output.append((node_id, child, result["conf"], result["cov"], result["iso"]))
                elif isinstance(result, tuple):
                    output += result
                    _, _, confidences, coverages, isomers = zip(*result)
                    for conf, cov, iso in zip(confidences, coverages, isomers):
                        output.append((node_id, child, conf, cov, iso))
            if len(output) != 0:
                return tuple(output)
        return None

    def find_fragment(self, min_mz, max_mz, **kwargs):
        if kwargs["topo"] is None:
            return None, None
        mean = (min_mz + max_mz) / 2
        crumb = CandyCrumbs(kwargs["topo"], [mean], (max_mz - min_mz) / 2, simplify=True)
        if crumb is None or crumb[mean] is None:
            return None, None
        crumb = crumb[mean]
        img = self.create_snfg(domon_costello_to_fragIUPAC(kwargs["topo"], crumb["Domon-Costello nomenclatures"][0]),
                               **kwargs)
        return img, crumb["Theoretical fragment masses"][0]

    def recurse(self, tree, node_id, criterion, val_path, val_y, test_path, test_y, parent=None, depth=0, **kwargs):
        print("Start recurse...")
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Collect ranks for 'leaf' option in plot_options
        if left_child == _tree.TREE_LEAF:
            self.ranks["leaves"].append(str(node_id))
        elif str(depth) not in self.ranks:
            self.ranks[str(depth)] = [str(node_id)]
        else:
            self.ranks[str(depth)].append(str(node_id))

        def node_html(x, y):
            print(x)
            return (f"<"
                    f"<table border='0' cellspacing='0' cellpadding='0'> \n"
                    f"<tr><td>\n"
                    f"<IMG SRC=\"{x}\"/></td></tr> "
                    f"<tr><td> {y} </td></tr> "
                    f"</table>"
                    f">")

        node_str = self.node_to_str(tree, node_id, criterion)
        if left_child == _tree.TREE_LEAF:  # Leaf node
            crit, _, value = node_str[1:-1].split("\\n")[:3]
            crit = norm_criterion(criterion, float(crit.split(" = ")[1]), **kwargs)
            values = [float(x) for x in value.split(" = ")[1][1:-1].split(", ")]
            if crit < 0.7:
                coverage, isomer = self.compute_node_strings(node_id, val_path, val_y, test_path, test_y, values, **kwargs)
                if kwargs["isomer_map"] is not None and len(kwargs["isomer_map"][isomer]) == 1:
                    isomer = list(kwargs["isomer_map"][isomer])[0]

            else:
                isomer = "q"
                coverage = f"Coverage: <br/>{[int(v / kwargs['weights'][c] + 0.5) for v, c in zip(values, kwargs['classes'])]}"
            self.out_file.write(
                "%d [label=%s, shape=plain] ;\n" % (node_id, node_html(self.create_snfg(isomer, **kwargs), coverage)))
        else:
            bin_str = node_str[1:-1].split(" <= ")[0]  # intermediate node
            bin_idx = int(bin_str.split("[")[1].split("]")[0])
            mz_interval = f"{frames[bin_idx]:.4f}", f"{frames[bin_idx + 1]:.4f}"
            img, mass = self.find_fragment(frames[bin_idx], frames[bin_idx + 1], **kwargs)
            mass_str = f"m/z: {mass}" if mass is not None else f"m/z interval:<br/>{mz_interval[0]}, {mz_interval[1]}"
            self.out_file.write("%d [label=%s, shape=plain] ;\n" % (
                node_id, node_html(img or self.create_snfg('q', **kwargs), mass_str)))

        if parent is not None:
            # Add edge to parent
            parent_str = self.node_to_str(tree, parent, criterion)
            val = parent_str.split("\\n")[0].split(" <= ")[1]
            sign = " &le; " if node_id == tree.children_left[parent] else " &gt; "
            self.out_file.write("%d -> %d [label=<%s %s>, penwidth=%d, color=%s, fontcolor=%s] ;\n" % (
                parent,
                node_id,
                sign,
                f"{float(val):.1%}",
                3 if (parent, node_id) in kwargs["highlights"] else 1,
                "black" if (parent, node_id) in kwargs["highlights"] else "grey",
                "black" if (parent, node_id) in kwargs["highlights"] else "grey",
            ))

        if left_child != _tree.TREE_LEAF:
            for child in [left_child, right_child]:
                self.recurse(
                    tree,
                    child,
                    criterion,
                    val_path,
                    val_y,
                    test_path,
                    test_y,
                    parent=node_id,
                    depth=depth + 1,
                    **kwargs,
                )
