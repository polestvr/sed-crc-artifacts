"""Collect results/exp_*.json into leaderboard.md and print a compact table.

Frontier design: three operating points
  c60 = collar@alpha=0.6, i45 = intersect70@alpha=0.45, i20 = intersect50@alpha=0.2.
Overall champion = gate-passing config with lowest mean FP/h at the HEADLINE
point (intersect50, alpha=0.2), excluding reference rows (E6*/OR*/oracle/valtuned).
Per-point champions also reported.
"""
import glob
import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

POINTS = {
    ("collar", 0.6): "c60",
    ("intersect70", 0.45): "i45",
    ("intersect50", 0.2): "i20",
}


def is_reference(exp_id, route):
    return (exp_id.startswith(("E6", "OR")) or "oracle" in route
            or route.startswith("valtuned") or route == "fixed05")


def is_ablation(exp_id, config):
    """Protocol deviations (cal_frac!=0.5, shifted seeds, delta sweeps) stay in
    the table but never compete for champion."""
    return (exp_id.startswith(("CS-", "VER-", "DL-", "sw-"))
            or config.get("cal_frac", 0.5) != 0.5
            or config.get("seed_base", 0) != 0)


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "exp_*.json"))):
        d = json.load(open(p))
        s = d["summary"]
        c = d["config"]
        route = c.get("route", "-")
        rows.append({
            "exp_id": d["exp_id"],
            "matching": c.get("matching", "collar"),
            "variant": c.get("variant", "-"),
            "route": route,
            "grouping": c.get("grouping", "-"),
            "alpha": c.get("alpha", 0.1),
            "n_ok": s["n_splits_ok"],
            "gate": "PASS" if s["gate_pass"] else "fail",
            "mean_miss": round(s["mean_miss_share"], 4),
            "q95_miss": round(s["q95_miss_share"], 4),
            "fph": round(s["fp_per_h_mean"], 2),
            "ref": is_reference(d["exp_id"], route),
            "abl": is_ablation(d["exp_id"], c),
        })

    champions = {}
    for (m, a), tag in POINTS.items():
        cand = [r for r in rows if r["gate"] == "PASS" and not r["ref"] and not r["abl"]
                and r["matching"] == m and abs(r["alpha"] - a) < 1e-9]
        champions[tag] = min(cand, key=lambda r: r["fph"]) if cand else None

    lines = ["# Leaderboard", "",
             "| exp | match | variant | route | grouping | alpha | n_ok/100 | gate | mean miss | q95 miss | FP/h | ref |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["ref"], r["gate"] != "PASS", r["fph"])):
        lines.append("| {exp_id} | {matching} | {variant} | {route} | {grouping} | {alpha} | "
                     "{n_ok} | {gate} | {mean_miss} | {q95_miss} | {fph} | {ref} |".format(**r))
    lines.append("")
    for tag in ("i20", "i45", "c60"):
        ch = champions[tag]
        lines.append(f"**Champion {tag}:** " +
                     (f"{ch['exp_id']} (FP/h {ch['fph']}, n_ok {ch['n_ok']}/100)" if ch else "none"))
    lines.append("")
    lines.append("**Overall champion (headline i20):** " +
                 (champions["i20"]["exp_id"] if champions["i20"] else "none"))
    with open(os.path.join(RESULTS, "leaderboard.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
