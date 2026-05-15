#!/usr/bin/env python
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

warnings.filterwarnings("ignore")

TRIVIAL = {
    "*O*",
    "*N*",
    "*C*",
    "*CC*",
    "*CC",
    "*OC",
    "*C(*)*",
    "*S*",
    "*N(*)*",
}
MIN_SUP = 0.01
J_COOC = 0.15
TOP_N = 5
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


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    i = int((a & b).sum())
    u = int((a | b).sum())
    return round(i / u, 3) if u else 0.0


def filter_trivials(sl: list[str], sup: np.ndarray, min_sup: float) -> list[tuple[int, str, float]]:
    cands = [
        (i, sl[i], float(sup[i]))
        for i in range(len(sl))
        if sl[i] not in TRIVIAL and sup[i] >= min_sup and atom_count(sl[i]) >= 2
    ]
    cands.sort(key=lambda x: -x[2])
    return cands


def find_alert_families(cands, masks, catalog, n):
    m_al = {s: motif_intrinsic_alerts(s, catalog) for _, s, _ in cands}
    al2m = defaultdict(list)
    for _, s, sv in cands:
        for a in m_al[s]:
            al2m[a].append((s, sv))

    seen, groups = {}, []
    for al, members in sorted(al2m.items(), key=lambda x: -len(x[1])):
        if len(members) < 2:
            continue
        members.sort(key=lambda x: -x[1])
        key = frozenset(m for m, _ in members)
        if key in seen:
            continue
        seen[key] = al
        or_m = np.zeros(n, dtype=bool)
        for s, _ in members:
            if s in masks:
                or_m |= masks[s]
        groups.append(
            {
                "type": "alert_family",
                "label": al,
                "members": members,
                "or_mask": or_m,
            }
        )
    return groups


def find_subsuming(cands, masks, n):
    mot_sup = {s: sv for _, s, sv in cands}
    cores = {}
    for _, s, _ in cands:
        c = get_core(s)
        if c is not None:
            cores[s] = c

    gen_to_spec = defaultdict(set)
    spec_to_gens = defaultdict(set)

    for ai, bi in combinations(range(len(cands)), 2):
        _, sa, _ = cands[ai]
        _, sb, _ = cands[bi]
        ca = cores.get(sa)
        cb = cores.get(sb)
        if ca is None or cb is None:
            continue
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

        if ab and ba:
            wc_a = sa.count("*")
            wc_b = sb.count("*")
            if wc_a <= wc_b:
                gen_to_spec[sa].add(sb)
                spec_to_gens[sb].add(sa)
            else:
                gen_to_spec[sb].add(sa)
                spec_to_gens[sa].add(sb)
        elif ab:
            gen_to_spec[sa].add(sb)
            spec_to_gens[sb].add(sa)
        elif ba:
            gen_to_spec[sb].add(sa)
            spec_to_gens[sa].add(sb)

    roots = sorted(
        [s for s in gen_to_spec if not spec_to_gens[s]],
        key=lambda r: (-mot_sup.get(r, 0), cores[r].GetNumAtoms() if r in cores else 0),
    )

    used = set()
    groups = []
    for root in roots:
        members = [root] + sorted([m for m in gen_to_spec[root] if m not in used], key=lambda m: -mot_sup.get(m, 0))
        if len(members) < 2:
            continue
        or_m = np.zeros(n, dtype=bool)
        for m in members:
            if m in masks:
                or_m |= masks[m]
        groups.append(
            {
                "type": "subsuming",
                "label": " | ".join(m[:14] for m in members[:2]),
                "members": [(m, mot_sup.get(m, 0)) for m in members],
                "or_mask": or_m,
            }
        )
        used.update(members)

    return groups


def find_cooccurring(cands, masks, n, j_cooc: float):
    seen, groups = set(), []
    for ai, bi in combinations(range(len(cands)), 2):
        _, sa, sva = cands[ai]
        _, sb, svb = cands[bi]
        j = jaccard(masks[sa], masks[sb])
        if j < j_cooc:
            continue
        if check_sub(sa, sb):
            continue
        key = frozenset([sa, sb])
        if key in seen:
            continue
        seen.add(key)
        groups.append(
            {
                "type": "co_occurring",
                "label": f"{sa[:10]}+{sb[:10]}",
                "members": [(sa, sva), (sb, svb)],
                "or_mask": masks[sa] | masks[sb],
                "and_mask": masks[sa] & masks[sb],
                "J": j,
            }
        )
    return groups


def find_standalone(cands, masks, all_grouped):
    groups = []
    for _, s, sv in cands:
        if s in all_grouped:
            continue
        groups.append(
            {
                "type": "standalone",
                "label": s,
                "members": [(s, sv)],
                "or_mask": masks[s],
                "and_mask": masks[s],
            }
        )
    return groups


def select_top(all_groups, n, min_cov: float, top_n: int):
    for g in all_groups:
        g["or_cov"] = round(float(g["or_mask"].mean()) * 100, 1)
        if "and_mask" in g:
            g["and_cov"] = round(float(g["and_mask"].mean()) * 100, 1)
    all_groups.sort(key=lambda x: -x["or_cov"])
    return [g for g in all_groups if g["or_cov"] >= min_cov][:top_n]


def label_dist(mask: np.ndarray, n: int) -> dict[str, float]:
    n1 = int(mask.astype(bool).sum())
    n0 = int(n - n1)
    return {
        "n1": n1,
        "n0": n0,
        "pct1": round(n1 / n * 100, 1) if n > 0 else 0.0,
        "pct0": round(n0 / n * 100, 1) if n > 0 else 0.0,
    }


def enumerate_combinations(top, n: int, k_max: int):
    combos = []
    for k in range(1, k_max + 1):
        for idx in combinations(range(len(top)), k):
            grps = [top[i] for i in idx]
            gate_choices = []
            for g in grps:
                if g["type"] == "co_occurring":
                    gate_choices.append(["OR", "AND"])
                else:
                    gate_choices.append(["OR"])
            for gates in product(*gate_choices):
                or_motifs = set()
                and_motifs = set()
                for i, gate in enumerate(gates):
                    motifs = {s for s, _ in grps[i]["members"]}
                    if gate == "OR":
                        or_motifs |= motifs
                    else:
                        and_motifs |= motifs
                if or_motifs & and_motifs:
                    continue

                gms = [grps[i]["or_mask"] if gates[i] == "OR" else grps[i]["and_mask"] for i in range(k)]

                for cross in ["OR", "AND"]:
                    if cross == "OR":
                        final = np.zeros(n, dtype=bool)
                        for gm in gms:
                            final |= gm
                    else:
                        final = np.ones(n, dtype=bool)
                        for gm in gms:
                            final &= gm
                    if not final.any():
                        continue
                    dist = label_dist(final, n)

                    def motif_str(g, gate):
                        motifs = [s for s, _ in g["members"]]
                        return f"({', '.join(motifs)})[{gate}]"

                    rule_parts = [motif_str(grps[i], gates[i]) for i in range(k)]
                    rule = f" {cross} ".join(rule_parts)
                    combos.append(
                        {
                            "k": int(k),
                            "rule": rule,
                            "cross": cross,
                            "gates": list(gates),
                            "group_labels": [g["label"][:25] for g in grps],
                            "group_types": [g["type"] for g in grps],
                            "group_motifs": [[s for s, _ in g["members"]] for g in grps],
                            **dist,
                        }
                    )
    combos.sort(key=lambda x: -x["n1"])
    for i, c in enumerate(combos):
        c["rule_index"] = int(i)
    return combos


def _group_json_row(g, n: int):
    row = {
        "type": g["type"],
        "label": g["label"],
        "or_cov": g["or_cov"],
        "members": [{"motif": s, "sup": round(float(sv) * 100, 1)} for s, sv in g["members"]],
        "OR": label_dist(g["or_mask"], n),
    }
    if "and_mask" in g:
        row["AND"] = label_dist(g["and_mask"], n)
        row["and_cov"] = g.get("and_cov", 0)
        row["J"] = round(g.get("J", 0), 3)
    return row


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
    catalog = build_catalog()

    cands = filter_trivials(motif_names, sup, min_sup=min_sup)
    masks = {s: x[:, i].astype(bool) for i, s, _ in cands}

    alert_groups = find_alert_families(cands, masks, catalog, n)
    subsuming_groups = find_subsuming(cands, masks, n)
    all_grouped = {s for g in alert_groups + subsuming_groups for s, _ in g["members"]}
    cooc_groups = find_cooccurring(cands, masks, n, j_cooc=j_cooc)
    all_grouped = {s for g in alert_groups + subsuming_groups + cooc_groups for s, _ in g["members"]}
    standalone_groups = find_standalone(cands, masks, all_grouped)
    top = select_top(alert_groups + subsuming_groups + cooc_groups + standalone_groups, n, min_cov=min_cov, top_n=top_n)
    combos = enumerate_combinations(top, n, k_max=k_max)

    row_motif_sets = []
    for i in range(n):
        idxs = np.where(x[i] > 0)[0].tolist()
        row_motif_sets.append({motif_names[j] for j in idxs})

    return {
        "dataset": dataset_name,
        "fold": int(fold),
        "n": n,
        "groups_json": [_group_json_row(g, n) for g in top],
        "combos": combos,
        "row_smiles": row_smiles,
        "row_motif_sets": row_motif_sets,
        "motif_names": motif_names,
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
            f"[{c['rule_index']:03d}] k={c['k']} cross={c['cross']} "
            f"n1={c['n1']} ({c['pct1']}%) n0={c['n0']} ({c['pct0']}%)"
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
        "combos": rulebook.get("combos", []),
        "selected_rule": selected_rule,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

