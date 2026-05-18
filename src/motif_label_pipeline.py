#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

warnings.filterwarnings("ignore")

TRIVIAL = {"*O*", "*N*", "*C*", "*CC*", "*CC", "*OC", "*C(*)*", "*S*", "*N(*)*"}
MIN_SUP = 0.01
J_COOC = 0.15
TOP_N = 10
MIN_COV = 5.0
K_MAX = 3
CATALOG_NAMES = ["BRENK", "CHEMBL_Dundee", "CHEMBL_LINT"]


def build_catalog():
    params = FilterCatalogParams()
    for name in CATALOG_NAMES:
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog(params)


def atom_count(smarts: str) -> int:
    try:
        m = Chem.MolFromSmarts(smarts)
        return int(m.GetNumAtoms()) if m else 0
    except Exception:
        return 0


def motif_intrinsic_alerts(smarts: str, catalog) -> frozenset[str]:
    mol = Chem.MolFromSmiles(smarts.replace("[*]", "C").replace("*", "C"))
    if mol is None:
        return frozenset()
    return frozenset(e.GetDescription() for e in catalog.GetMatches(mol))


def get_core(smarts: str):
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        return None
    rw = RWMol(mol)
    wildcards = sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)
    for idx in wildcards:
        rw.RemoveAtom(idx)
    try:
        core = rw.GetMol()
        if core.GetNumAtoms() == 0:
            return None
        Chem.FastFindRings(core)
        return core
    except Exception:
        return None


def heteroatoms(core) -> set[int]:
    return {a.GetAtomicNum() for a in core.GetAtoms() if a.GetAtomicNum() not in (0, 6)}


def too_generic(core) -> bool:
    if core.GetRingInfo().NumRings() > 0:
        return False
    if core.GetNumAtoms() >= 3:
        return False
    for bond in core.GetBonds():
        if bond.GetBondTypeAsDouble() >= 2.0:
            return False
    return True


def aliphatic_pure_c(core, het: set[int]) -> bool:
    if het:
        return False
    if core.GetRingInfo().NumRings() > 0:
        return False
    return not any(a.GetIsAromatic() for a in core.GetAtoms())


def check_sub(sa: str, sb: str) -> bool:
    """True if either motif core subsumes the other, with guard checks."""
    ca = get_core(sa)
    cb = get_core(sb)
    if ca is None or cb is None:
        return False
    ha = heteroatoms(ca)
    hb = heteroatoms(cb)

    ab = False
    try:
        ab = cb.HasSubstructMatch(ca)
    except Exception:
        pass
    if ab:
        if too_generic(ca):
            ab = False
        elif aliphatic_pure_c(ca, ha) and hb:
            ab = False
        elif ha and not ha <= hb:
            ab = False

    ba = False
    try:
        ba = ca.HasSubstructMatch(cb)
    except Exception:
        pass
    if ba:
        if too_generic(cb):
            ba = False
        elif aliphatic_pure_c(cb, hb) and ha:
            ba = False
        elif hb and not hb <= ha:
            ba = False
    return ab or ba


def label_dist(mask: np.ndarray, n: int) -> dict[str, float]:
    n1 = int(mask.astype(bool).sum())
    n0 = int(n - n1)
    return {
        "n1": n1,
        "n0": n0,
        "pct1": round(n1 / n * 100, 1) if n > 0 else 0.0,
        "pct0": round(n0 / n * 100, 1) if n > 0 else 0.0,
    }


def compute_alert_families(top_motifs: list[str], all_cands, catalog):
    top_set = set(top_motifs)
    top_alerts = {t: motif_intrinsic_alerts(t, catalog) for t in top_motifs}
    alert_to_tops: dict[str, set[str]] = defaultdict(set)
    for t, alerts in top_alerts.items():
        for al in alerts:
            alert_to_tops[al].add(t)

    alert_groups = defaultdict(lambda: defaultdict(list))
    for _, s, sv in all_cands:
        motif_als = motif_intrinsic_alerts(s, catalog)
        for al in motif_als:
            if al not in alert_to_tops:
                continue
            for top in alert_to_tops[al]:
                alert_groups[top][al].append(
                    {
                        "motif": s,
                        "support": round(float(sv) * 100, 1),
                        "in_support": bool(sv >= MIN_SUP),
                        "in_top": bool(s in top_set),
                    }
                )
    return top_alerts, {t: dict(g) for t, g in alert_groups.items()}


def compute_subsuming_families(top_motifs: list[str], all_cands):
    top_set = set(top_motifs)
    top_cores = {}
    top_hets = {}
    for t in top_motifs:
        c = get_core(t)
        if c is not None:
            top_cores[t] = c
            top_hets[t] = heteroatoms(c)

    sub_groups = defaultdict(list)
    for _, sb, sv_b in all_cands:
        cb = get_core(sb)
        if cb is None:
            continue
        hb = heteroatoms(cb)
        nb = cb.GetNumAtoms()

        for t in top_motifs:
            ca = top_cores.get(t)
            if ca is None:
                continue
            ha = top_hets.get(t, set())
            na = ca.GetNumAtoms()

            ab = False
            if na <= nb and (not ha or ha <= hb):
                try:
                    ab = cb.HasSubstructMatch(ca)
                except Exception:
                    pass
                if ab:
                    if too_generic(ca):
                        ab = False
                    elif aliphatic_pure_c(ca, ha) and hb:
                        ab = False
                    elif ha and not ha <= hb:
                        ab = False

            ba = False
            if nb <= na and (not hb or hb <= ha):
                try:
                    ba = ca.HasSubstructMatch(cb)
                except Exception:
                    pass
                if ba:
                    if too_generic(cb):
                        ba = False
                    elif aliphatic_pure_c(cb, hb) and ha:
                        ba = False
                    elif hb and not hb <= ha:
                        ba = False

            if ab and ba:
                if t.count("*") <= sb.count("*"):
                    ba = False
                else:
                    ab = False

            if ab or ba:
                sub_groups[t].append(
                    {
                        "motif": sb,
                        "support": round(float(sv_b) * 100, 1),
                        "in_support": bool(sv_b >= MIN_SUP),
                        "in_top": bool(sb in top_set),
                        "direction": "specific" if ab else "general",
                    }
                )
    return dict(sub_groups)


def cooc_profile(top_motifs: list[str], all_cands, all_masks, j_cooc: float):
    top_set = set(top_motifs)
    profile = {}
    cooc_groups = defaultdict(list)

    for t in top_motifs:
        ma = all_masks[t].astype(bool)
        na = int(ma.sum())
        if na == 0:
            continue
        for _, sb, sv_b in all_cands:
            if sb == t:
                continue
            mb = all_masks[sb].astype(bool)
            nb = int(mb.sum())
            if nb == 0:
                continue
            inter = int((ma & mb).sum())
            u = na + nb - inter
            j = round(inter / u, 3) if u else 0.0
            p_b_given_a = round(inter / na, 3) if na else 0.0
            p_a_given_b = round(inter / nb, 3) if nb else 0.0
            profile[(t, sb)] = {
                "J": j,
                "p_b_given_a": p_b_given_a,
                "p_a_given_b": p_a_given_b,
                "inter": inter,
            }
            if sb in top_set:
                profile[(sb, t)] = {
                    "J": j,
                    "p_b_given_a": p_a_given_b,
                    "p_a_given_b": p_b_given_a,
                    "inter": inter,
                }
            if j >= float(j_cooc):
                cooc_groups[t].append(
                    {
                        "motif": sb,
                        "support": round(float(sv_b) * 100, 1),
                        "in_support": bool(sv_b >= MIN_SUP),
                        "in_top": bool(sb in top_set),
                        "J": j,
                        "p_b_given_a": p_b_given_a,
                        "p_a_given_b": p_a_given_b,
                    }
                )

    for t in cooc_groups:
        cooc_groups[t].sort(key=lambda x: (-x["J"], -x["p_b_given_a"]))
    return profile, dict(cooc_groups)


def _cooccurring_pairs(top_motifs: list[str], masks: dict[str, np.ndarray], profile, j_cooc: float):
    pairs = []
    for a, b in combinations(top_motifs, 2):
        p = profile.get((a, b), {})
        if float(p.get("J", 0.0)) < float(j_cooc):
            continue
        if check_sub(a, b):
            continue
        or_mask = masks[a] | masks[b]
        and_mask = masks[a] & masks[b]
        pairs.append(
            {
                "motifs": [a, b],
                "J": float(p.get("J", 0.0)),
                "p_b_given_a": float(p.get("p_b_given_a", 0.0)),
                "p_a_given_b": float(p.get("p_a_given_b", 0.0)),
                "OR": label_dist(or_mask, int(or_mask.shape[0])),
                "AND": label_dist(and_mask, int(and_mask.shape[0])),
            }
        )
    pairs.sort(key=lambda x: (-x["J"], -x["AND"]["n1"]))
    return pairs


def _build_rule_combos(top_motifs: list[str], masks: dict[str, np.ndarray], n: int, k_max: int, pair_rules):
    combos = []
    idx = 0

    # OR combinations over singleton motifs.
    for k in range(1, int(k_max) + 1):
        for sel in combinations(top_motifs, k):
            final = np.zeros((n,), dtype=bool)
            for m in sel:
                final |= masks[m]
            dist = label_dist(final, n)
            motifs_str = " OR ".join(sel)
            combos.append(
                {
                    "rule_index": idx,
                    "rule_type": "or_combo",
                    "k": int(k),
                    "rule": motifs_str,
                    "cross": "OR",
                    "gates": ["OR"] * int(k),
                    "group_motifs": [[m] for m in sel],
                    **dist,
                }
            )
            idx += 1

    # Dedicated AND pair rules for valid co-occurring pairs.
    for pair in pair_rules:
        a, b = pair["motifs"]
        combos.append(
            {
                "rule_index": idx,
                "rule_type": "cooc_and",
                "k": 2,
                "rule": f"{a} AND {b}",
                "cross": "OR",
                "gates": ["AND"],
                "group_motifs": [[a, b]],
                **pair["AND"],
                "pair_stats": {
                    "J": pair["J"],
                    "p_b_given_a": pair["p_b_given_a"],
                    "p_a_given_b": pair["p_a_given_b"],
                },
            }
        )
        idx += 1

    combos.sort(key=lambda x: (-x["n1"], x["rule_type"] != "cooc_and", x["rule_index"]))
    for new_idx, c in enumerate(combos):
        c["rule_index"] = int(new_idx)
    return combos


def load_dataset_rulebook(
    *,
    data_root: str | Path,
    dataset_name: str,
    fold: int,
    min_sup: float = MIN_SUP,
    j_cooc: float = J_COOC,
    top_n: int = TOP_N,
    min_cov: float = MIN_COV,
    k_max: int = K_MAX,
) -> dict[str, Any]:
    data_root = Path(data_root)
    base = data_root / f"{dataset_name}_fold{int(fold)}"
    matrix_path = base / "graph_motif_matrix.npz"
    cols_path = base / "graph_motif_matrix_columns.csv"
    rows_path = base / "graph_motif_matrix_rows.csv"
    if not matrix_path.is_file() or not cols_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(
            f"Missing motif matrix exports for {dataset_name} fold={fold} under {base}. "
            "Expected graph_motif_matrix.npz / graph_motif_matrix_columns.csv / graph_motif_matrix_rows.csv."
        )

    x = sp.load_npz(matrix_path).toarray().astype(np.uint8)
    cols_df = pd.read_csv(cols_path)
    rows_df = pd.read_csv(rows_path)
    if "motif_identity" not in cols_df.columns:
        raise KeyError(f"{cols_path} must contain 'motif_identity' column")
    if "smiles" not in rows_df.columns:
        raise KeyError(f"{rows_path} must contain 'smiles' column")

    n = int(x.shape[0])
    sup = x.mean(axis=0) if x.size else np.zeros((0,), dtype=float)
    motif_names = [str(v) for v in cols_df["motif_identity"].tolist()]
    row_smiles = [str(v) for v in rows_df["smiles"].tolist()]

    all_cands = [
        (i, motif_names[i], float(sup[i]))
        for i in range(len(motif_names))
        if motif_names[i] not in TRIVIAL and atom_count(motif_names[i]) >= 2
    ]
    all_cands.sort(key=lambda t: -t[2])
    all_masks = {s: x[:, i].astype(bool) for i, s, _ in all_cands}

    cands = [(i, s, sv) for i, s, sv in all_cands if sv >= float(min_sup)]
    masks = {s: all_masks[s] for _, s, _ in cands}

    top_motifs = [s for _, s, _ in cands if float(masks[s].mean()) * 100 >= float(min_cov)][: int(top_n)]
    catalog = build_catalog()
    top_alerts, alert_groups = compute_alert_families(top_motifs, all_cands, catalog)
    sub_groups = compute_subsuming_families(top_motifs, all_cands)
    profile, cooc_groups = cooc_profile(top_motifs, all_cands, all_masks, float(j_cooc))
    pair_rules = _cooccurring_pairs(top_motifs, masks, profile, float(j_cooc))
    combos = _build_rule_combos(top_motifs, masks, n, int(k_max), pair_rules)

    motif_records = []
    for _, s, sv in cands:
        if s not in top_motifs:
            continue
        cooc_with = []
        for other in top_motifs:
            if other == s:
                continue
            p = profile.get((s, other), {})
            cooc_with.append(
                {
                    "motif": other,
                    "J": p.get("J", 0.0),
                    "p_other_given_this": p.get("p_b_given_a", 0.0),
                    "p_this_given_other": p.get("p_a_given_b", 0.0),
                }
            )
        cooc_with.sort(key=lambda z: (-z["J"], -z["p_other_given_this"]))
        motif_records.append(
            {
                "type": "motif_singleton",
                "label": s,
                "motif": s,
                "support": round(float(sv) * 100, 1),
                "alerts": sorted(top_alerts.get(s, set())),
                "alert_family": alert_groups.get(s, {}),
                "subsuming_family": sub_groups.get(s, []),
                "cooc_profile": cooc_with,
                "OR": label_dist(masks[s], n),
            }
        )

    row_motif_sets = []
    for i in range(n):
        idxs = np.where(x[i] > 0)[0].tolist()
        row_motif_sets.append({motif_names[j] for j in idxs})

    return {
        "dataset": dataset_name,
        "fold": int(fold),
        "n": n,
        "groups_json": motif_records,
        "pair_json": pair_rules,
        "combos": combos,
        "row_smiles": row_smiles,
        "row_motif_sets": row_motif_sets,
        "motif_names": motif_names,
        "top_motifs": top_motifs,
        "cooc_groups": cooc_groups,
        "config": {
            "min_sup": float(min_sup),
            "j_cooc": float(j_cooc),
            "top_n": int(top_n),
            "min_cov": float(min_cov),
            "k_max": int(k_max),
        },
    }


def choose_rule_interactive(
    rulebook: dict[str, Any],
    *,
    selected_index: int | None = None,
    interactive: bool = True,
) -> dict[str, Any]:
    combos = list(rulebook.get("combos", []))
    if not combos:
        raise ValueError("No valid rule combinations were generated for selection.")

    print("\n" + "=" * 80)
    print(f"[RULE SELECTION] Dataset={rulebook['dataset']} fold={rulebook['fold']} n={rulebook['n']}")
    print("=" * 80)
    for c in combos:
        print(
            f"[{c['rule_index']:03d}] type={c.get('rule_type', 'rule')} "
            f"k={c['k']} n1={c['n1']} ({c['pct1']}%) n0={c['n0']} ({c['pct0']}%)"
        )
        print(f"      {c['rule']}")

    if selected_index is not None:
        idx = int(selected_index)
        if not (0 <= idx < len(combos)):
            raise ValueError(f"selected_index={idx} out of range [0, {len(combos)-1}]")
        chosen = combos[idx]
        print(f"[INFO] Using preselected rule index {idx}")
        return chosen

    if interactive and hasattr(__import__("sys").stdin, "isatty") and __import__("sys").stdin.isatty():
        while True:
            raw = input(f"Select rule index [0..{len(combos)-1}] (default 0): ").strip()
            if raw == "":
                idx = 0
                break
            try:
                idx = int(raw)
            except ValueError:
                print("[WARN] Please enter an integer rule index.")
                continue
            if 0 <= idx < len(combos):
                break
            print(f"[WARN] Index out of range. Valid range: 0..{len(combos)-1}")
    else:
        idx = 0
        print(f"[WARN] Non-interactive session. Defaulting to rule index {idx}.")

    chosen = combos[idx]
    print(
        f"[INFO] Selected rule [{idx}] -> n1={chosen['n1']} ({chosen['pct1']}%), "
        f"n0={chosen['n0']} ({chosen['pct0']}%)"
    )
    return chosen


def evaluate_rule_on_motifs(present_motifs: set[str], selected_rule: dict[str, Any]) -> tuple[bool, set[str]]:
    """Evaluate selected rule against present motif set and return active motifs."""
    group_motifs = selected_rule.get("group_motifs", [])
    gates = selected_rule.get("gates", [])
    cross = str(selected_rule.get("cross", "OR")).upper()
    if not group_motifs:
        return False, set()

    group_hits = []
    for motifs, gate in zip(group_motifs, gates):
        motifs_set = {str(m) for m in motifs}
        if str(gate).upper() == "AND":
            ok = all(m in present_motifs for m in motifs_set)
            active = set(motifs_set) if ok else set()
        else:
            active = {m for m in motifs_set if m in present_motifs}
            ok = bool(active)
        group_hits.append((ok, active))

    if cross == "AND":
        positive = all(ok for ok, _ in group_hits)
        active_motifs = set().union(*[a for _, a in group_hits]) if positive else set()
    else:
        positive = any(ok for ok, _ in group_hits)
        active_motifs = set().union(*[a for ok, a in group_hits if ok]) if positive else set()
    return bool(positive), active_motifs


def save_rulebook_json(rulebook: dict[str, Any], selected_rule: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": rulebook.get("dataset"),
        "fold": int(rulebook.get("fold", 0)),
        "n": int(rulebook.get("n", 0)),
        "groups": rulebook.get("groups_json", []),
        "pairs": rulebook.get("pair_json", []),
        "combos": rulebook.get("combos", []),
        "selected_rule": selected_rule,
        "config": rulebook.get("config", {}),
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def run_pipeline(data_root: str, datasets: list[str], out_dir: str, fold: int = 0):
    out = {}
    for dataset_name in datasets:
        out[dataset_name] = load_dataset_rulebook(
            data_root=data_root,
            dataset_name=dataset_name,
            fold=int(fold),
            min_sup=MIN_SUP,
            j_cooc=J_COOC,
            top_n=TOP_N,
            min_cov=MIN_COV,
            k_max=K_MAX,
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "motif_label_results.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] Saved: {out_path / 'motif_label_results.json'}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["Mutagenicity", "Benzene", "BBBP", "hERG", "Alkane_Carbonyl"])
    parser.add_argument("--out_dir", default="./pipeline_output")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--min_sup", type=float, default=MIN_SUP)
    parser.add_argument("--j_cooc", type=float, default=J_COOC)
    parser.add_argument("--top_n", type=int, default=TOP_N)
    parser.add_argument("--min_cov", type=float, default=MIN_COV)
    parser.add_argument("--k_max", type=int, default=K_MAX)
    args = parser.parse_args()

    MIN_SUP = float(args.min_sup)
    J_COOC = float(args.j_cooc)
    TOP_N = int(args.top_n)
    MIN_COV = float(args.min_cov)
    K_MAX = int(args.k_max)
    run_pipeline(args.data_root, args.datasets, args.out_dir, fold=int(args.fold))

