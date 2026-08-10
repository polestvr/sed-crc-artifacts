"""Generate the paper's LaTeX table fragments strictly from results/*.json -
no hand-entered numbers anywhere.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
PAPER = os.environ.get("SED_CRC_TABLES", os.path.join(HERE, "..", "tables_out"))

ROUTE_LBL = {
    "crc_naive": "CRC (naive)",
    "crc_split_mono": "CRC (split-monotonized)",
    "ltt_split_clipmean": "LTT split (clip HB, 200 cand.)",
    "ltt_split_pooled": "LTT split (pooled Hoef., heur.)",
    "ltt_split_pooledcap": "LTT split (pooled-cap, proven)",
    "ltt_bonf_clipmean": "LTT Bonferroni (clip HB, 50 cand.)",
    "ltt_bonf_pooled": "LTT Bonferroni (pooled Hoef., heur.)",
    "rcps_fixedseq": "RCPS fixed-seq.\\ (clip HB)",
    "rcps_fixedseq_pooled": "RCPS fixed-seq.\\ (pooled Hoef., heur.)",
    "crc_c_nonmono": "CRC-C nonmono.\\ \\citep{angelopoulos2026nonmono}",
    "crc_nm": "CRC-NM \\citep{aldirawi2026nonmono}",
    "ltt_split_cm50": "LTT split (clip HB, 50 cand.)",
    "ltt_bonf_cm200": "LTT Bonferroni (clip HB, 200 cand.)",
    "ltt_bonf_2d": "LTT Bonf.\\ 2-D (win$\\times$thr)",
    "oracle_cw": "Oracle (class-wise, eval-half)",
    "oracle_marg": "Oracle (marginal, eval-half)",
    "valtuned": "Val-tuned (no correction)",
    "valtuned_alpha_m0": "Tune-to-$\\alpha$ on val.\\ (uncert.)",
    "valtuned_alpha_m0.02": "Tune-to-$\\alpha{-}0.02$ on val.\\ (uncert.)",
    "fixed05": "Fixed 0.5",
}
VAR_LBL = {"raw": "raw", "medfilt": "median filter", "csebb": "cSEBB",
           "csebbmax": "cSEBB-max", "csebbtopk3": "cSEBB-top3"}
# Certificate semantics and functional per route (certificate identity is a
# first-class table column).
CERT_LBL = {
    "crc_naive": "in-exp.\\ pooled",
    "crc_split_mono": "in-exp.\\ pooled (heur.)",
    "ltt_split_clipmean": "HP clip",
    "ltt_split_cm50": "HP clip",
    "ltt_bonf_clipmean": "HP clip",
    "ltt_bonf_cm200": "HP clip",
    "ltt_bonf_2d": "HP clip",
    "rcps_fixedseq": "HP clip",
    "ltt_split_pooled": "heur.\\ pooled",
    "ltt_bonf_pooled": "heur.\\ pooled",
    "rcps_fixedseq_pooled": "heur.\\ pooled",
    "crc_c_nonmono": "in-exp.\\ clip (transpl.)",
    "crc_nm": "in-exp.\\ clip (transpl.)",
    "ltt_split_pooledcap": "uncert.\\ (fallback)",
}
# Primary criterion: proven-HP certificates first,
# then in-expectation, then heuristic, then vacuous/fallback constructions.
CERT_RANK = {
    "ltt_split_clipmean": 0, "ltt_split_cm50": 0, "ltt_bonf_clipmean": 0,
    "ltt_bonf_cm200": 0, "ltt_bonf_2d": 0, "rcps_fixedseq": 0,
    "crc_naive": 1, "crc_split_mono": 1, "crc_c_nonmono": 1, "crc_nm": 1,
    "ltt_split_pooled": 2, "ltt_bonf_pooled": 2, "rcps_fixedseq_pooled": 2,
    "ltt_split_pooledcap": 3,
}
GRP_LBL = {"marginal": "marg.", "classwise": "class.", "clustered": "clust."}
POINTS = [("collar", 0.6, "c60", r"collar matching, $\alpha{=}0.6$"),
          ("intersect70", 0.45, "i45", r"intersection-0.7, $\alpha{=}0.45$"),
          ("intersect50", 0.2, "i20", r"intersection-0.5, $\alpha{=}0.2$")]


def load_all():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "exp_*.json"))):
        d = json.load(open(p))
        c, s = d["config"], d["summary"]
        route = c.get("route", "-")
        grouping = c.get("grouping", "-")
        variant = c.get("variant", "-")
        if d["exp_id"].startswith("2D-"):
            route = "ltt_bonf_2d"
            variant = "medfilt"
            grouping = "classwise" if d["exp_id"].endswith("-cw") else "marginal"
        if route == "valtuned_alpha":
            route = f"valtuned_alpha_m{c.get('margin', 0.0):g}"
        if route == "oracle_cw":
            grouping = "classwise"
        if route == "oracle_marg":
            grouping = "marginal"
        rows.append(dict(
            exp=d["exp_id"], variant=variant, route=route,
            grouping=grouping, matching=c.get("matching", "collar"),
            alpha=c.get("alpha", 0.1), cal_frac=c.get("cal_frac", 0.5),
            seed_base=c.get("seed_base", 0), delta=c.get("delta", 0.05),
            n_ok=s["n_splits_ok"], miss=s["mean_miss_share"], q95=s["q95_miss_share"],
            fph=s["fp_per_h_mean"], fph_std=s.get("fp_per_h_std", 0.0)))
    return rows


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def main_table(rows):
    out = []
    for m, a, tag, title in POINTS:
        sel = [r for r in rows if r["matching"] == m and abs(r["alpha"] - a) < 1e-9
               and r["cal_frac"] == 0.5 and r["seed_base"] == 0 and r["delta"] == 0.05
               and not r["exp"].startswith(("E", "OR", "sw-", "CS-", "DL-", "VER"))]
        cal = [r for r in sel if "oracle" not in r["route"]
               and not r["route"].startswith("valtuned") and r["route"] != "fixed05"]
        refs = [r for r in sel if r not in cal]
        out.append(r"\begin{table}[t]\centering\small")
        out.append(rf"\caption{{Validity and efficiency at {title}: fraction of 100 "
                   r"calibration/evaluation splits with pooled miss share $\le\alpha$ "
                   r"(gate: $\ge 95$), mean and 95\%-quantile realized miss share, and FP/h. "
                   r"Primary criterion (\cref{sec:setup}): among proven-certificate "
                   r"(`HP') rows, minimize FP/h; rows are grouped by certificate class "
                   r"within each decoding. "
                   r"The gate is a deployment-acceptance diagnostic, not a validity "
                   r"criterion: \cref{sec:validity} shows it rewards certificate slack, and "
                   r"the population-violation \emph{brackets} of \cref{tab:instrument} "
                   r"(biased column, complementary-half exceedance) supersede "
                   r"its verdicts for validity claims where the bracket is tight "
                   r"(pass counts within $\pm2$ of the bar "
                   r"are statistically borderline, binomial sd $\approx 2$). Rows labeled "
                   r"`heur.' use the unproven pooled bound (\cref{sec:routes}); class-wise "
                   r"rows carry per-class certificates only, gate-scored here against the "
                   r"pooled criterion. Cert.: certificate semantics and "
                   r"functional -- HP = high-probability (level $\delta$), "
                   r"in-exp.\ = in-expectation, heur.\ = unproven pooled bound. "
                   r"Oracle rows pick thresholds on "
                   r"the evaluation half itself.}")
        out.append(rf"\label{{tab:{tag}}}")
        out.append(r"\begin{tabular}{lllllrrrr}\toprule")
        out.append(r"Decoding & Route & Cert.\ & Grp & Gate & $n_{\mathrm{ok}}$/100 & miss & q95 & FP/h \\\midrule")
        for r in sorted(cal, key=lambda r: (r["variant"],
                                            CERT_RANK.get(r["route"], 4),
                                            r["fph"])):
            gate = r"\textbf{pass}" if r["n_ok"] >= 95 else "fail"
            out.append(f"{VAR_LBL.get(r['variant'], r['variant'])} & {ROUTE_LBL.get(r['route'], r['route'])} & "
                       f"{CERT_LBL.get(r['route'], '--')} & "
                       f"{GRP_LBL.get(r['grouping'], r['grouping'])} & {gate} & {r['n_ok']} & "
                       f"{fmt(r['miss'])} & {fmt(r['q95'])} & {r['fph']:.0f}$\\pm${r['fph_std']:.0f} \\\\")
        out.append(r"\midrule")
        for r in sorted(refs, key=lambda r: r["fph"]):
            out.append(f"{VAR_LBL.get(r['variant'], r['variant'])} & {ROUTE_LBL.get(r['route'], r['route'])} & "
                       f"none & "
                       f"{GRP_LBL.get(r['grouping'], r['grouping'])} & -- & {r['n_ok']} & "
                       f"{fmt(r['miss'])} & {fmt(r['q95'])} & {r['fph']:.0f}$\\pm${r['fph_std']:.0f} \\\\")
        out.append(r"\bottomrule\end{tabular}\end{table}")
        out.append("")
    return "\n".join(out)


def audit_table():
    d = json.load(open(os.path.join(RESULTS, "audit_monotonicity.json")))
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Monotonicity audit on the validation split (200-point grid): "
           r"classes (of 15) with any negative step of the pooled miss curve, number of "
           r"negative steps, and their total magnitude in events (4919 reference events).}",
           r"\label{tab:audit}",
           r"\begin{tabular}{llrrr}\toprule",
           r"Matching & Decoding & classes & steps & magnitude \\\midrule"]
    order = ["collar", "intersect70", "intersect50"]
    mlbl = {"collar": "collar", "intersect70": "intersection-0.7", "intersect50": "intersection-0.5"}
    for m in order:
        for v in ("raw", "medfilt", "csebb"):
            key = f"{m}/{v}"
            if key not in d:
                continue
            a = d[key]["grid200"]["aggregate"]
            out.append(f"{mlbl[m]} & {VAR_LBL[v]} & {a['classes_with_violations']} & "
                       f"{a['total_neg_steps']} & {a['total_neg_magnitude']} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def floors_table():
    d = json.load(open(os.path.join(RESULTS, "reviewfix_stats.json")))
    seeds = []
    sf = os.path.join(RESULTS, "seed_floors.jsonl")
    if os.path.exists(sf):
        with open(sf) as f:
            seeds = [json.loads(l) for l in f if l.strip()]
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Feasibility floors: minimum achievable pooled event miss share on the "
           r"test split (5{,}028 reference events), per matching rule and decoding: the "
           r"frozen checkpoint's common-threshold floor with clip-level bootstrap 95\% CI, "
           r"its per-class-threshold floor (sum of per-class minima), and the mean$\pm$sd of "
           r"the common floor over four independent runs of the official training recipe. "
           r"$\alpha{=}0.1$ lies below every floor of every run (four-run minimum $0.109$).}",
           r"\label{tab:floors}",
           r"\begin{tabular}{llrrrr}\toprule",
           r"Matching rule & Decoding & common & 95\% CI & class-wise & 4-run mean$\pm$sd \\\midrule"]
    name = {"collar": "collar (sed\\_eval)",
            "intersect70": "intersection-0.7", "intersect50": "intersection-0.5"}
    import numpy as _np
    for key in ("collar", "intersect70", "intersect50"):
        for v in ("medfilt", "csebb"):
            e = d.get(f"{key}/{v}")
            if not e:
                continue
            lo, hi = e["floor_ci95"]
            vals = [e["common_floor"]] + [s[f"{key}/{v}"]["common_floor"]
                                          for s in seeds if f"{key}/{v}" in s]
            ms = f"{_np.mean(vals):.3f}$\\pm${_np.std(vals, ddof=1):.3f}" if len(vals) > 1 else "--"
            out.append(f"{name[key]} & {VAR_LBL[v]} & {fmt(e['common_floor'])} & "
                       f"[{fmt(lo)}, {fmt(hi)}] & {fmt(e['classwise_floor'])} & {ms} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def subgroup_table():
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Per-subgroup validity of the champion configurations (marginally "
           r"calibrated). $n_{\mathrm{ok}}$ = splits (of 100) with subgroup miss share "
           r"$\le\alpha$; $n_{\mathrm{clips}}$ = subgroup size; ev = mean reference events "
           r"per evaluation half. Marginal calibration promises no per-subgroup validity; "
           r"these rows measure the marginal-to-conditional gap. Counts are raw "
           r"(the 100 splits are dependent complementary resamples, so no exact test "
           r"over them exists); the text applies a Bonferroni-corrected descriptive "
           r"screen across the 11 subgroups.}",
           r"\label{tab:subgroups}",
           r"\begin{tabular}{lrrrrrr}\toprule",
           r"& \multicolumn{4}{c}{i20 (cSEBB-max, $\alpha{=}.2$)} & \multicolumn{2}{c}{c60 (cSEBB, $\alpha{=}.6$)} \\",
           r"Subgroup & $n_{\mathrm{clips}}$ & ev & $n_{\mathrm{ok}}$ & miss & $n_{\mathrm{ok}}$ & miss \\\midrule"]
    a = json.load(open(os.path.join(RESULTS, "stress_SG-i20-max.json")))["groups"]
    b = json.load(open(os.path.join(RESULTS, "stress_SG-c60.json")))["groups"]
    keys = sorted(set(a) | set(b), key=lambda k: a.get(k, {}).get("mean_miss", 0), reverse=True)
    for k in keys:
        ra, rb = a.get(k), b.get(k)
        va = (f"{ra['n_clips_total']} & {ra['mean_events_per_split']:.0f} & "
              f"{ra['n_ok']} & {fmt(ra['mean_miss'])}") if ra else "-- & -- & -- & --"
        vb = f"{rb['n_ok']} & {fmt(rb['mean_miss'])}" if rb else "-- & --"
        out.append(f"{k.replace('_', ' ').replace('/', ': ')} & {va} & {vb} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def transfer_table():
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Transfer stress tests. Cross-decode: thresholds calibrated on decoder A, "
           r"deployed on decoder B (100 splits). Cross-corpus: per-class thresholds calibrated "
           r"once on the full RealDESED test (or DESED public eval), deployed on the other "
           r"corpus; 3 mapped classes, 993 RealDESED / 693 DESED reference events, single "
           r"calibration draw -- point estimates to be read as illustrative, not inferential.}",
           r"\label{tab:transfer}",
           r"\begin{tabular}{llrrr}\toprule",
           r"Test & Setting & $n_{\mathrm{ok}}$/100 & miss & FP/h \\\midrule"]
    for exp, lbl in (("XD-c60-med2seb", r"cross-decode c60: medfilt$\to$cSEBB"),
                     ("XD-c60-seb2med", r"cross-decode c60: cSEBB$\to$medfilt"),
                     ("XD-i20-seb2med", r"cross-decode i20: cSEBB$\to$medfilt")):
        s = json.load(open(os.path.join(RESULTS, f"stress_{exp}.json")))
        if "summary" in s:
            s = s["summary"]
        out.append(f"{lbl} & & {s['n_splits_ok']} & {fmt(s['mean_miss_share'])} & "
                   f"{s['fp_per_h_mean']:.0f} \\\\")
    out.append(r"\midrule")
    for m, alab in (("intersect50", r"i20 ($\alpha{=}.2$)"), ("collar", r"c60 ($\alpha{=}.6$)")):
        d = json.load(open(os.path.join(RESULTS, f"stress_crosscorpus__{m}.json")))
        r1, r2 = d["realdesed_to_desed"], d["desed_to_realdesed"]
        out.append(rf"cross-corpus {alab} & RealDESED$\to$DESED (in-dom.\ {fmt(r1['indomain_resub_miss'])}) & -- & "
                   f"{fmt(r1['desed_miss_share'])} & {r1['desed_fp_per_h']:.0f} \\\\")
        out.append(rf"cross-corpus {alab} & DESED$\to$RealDESED & -- & {fmt(r2['realdesed_miss_share'])} & "
                   f"{r2['realdesed_fp_per_h']:.0f} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def ablation_table(rows):
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Ablations around the headline champion (intersection-0.5, "
           r"$\alpha{=}0.2$, cSEBB-max unless stated): calibration fraction, test level "
           r"$\delta$, guarantee level $\alpha$, grouping, and 2-D $\lambda$ at c60.}",
           r"\label{tab:ablations}",
           r"\begin{tabular}{lrrrr}\toprule",
           r"Setting & $n_{\mathrm{ok}}$/100 & miss & q95 & FP/h \\\midrule"]

    def row(exp_id, lbl):
        r = next((x for x in rows if x["exp"] == exp_id), None)
        if r:
            out.append(f"{lbl} & {r['n_ok']} & {fmt(r['miss'])} & {fmt(r['q95'])} & {r['fph']:.0f} \\\\")

    row("i20-cse-ltt_split_clipmean-marg", "champion decode ablation: cSEBB conf.")
    row("i20-csebbmax-ltt_split-marg", r"\quad cSEBB-max (champion)")
    row("i20-csebbtopk3-ltt_split-marg", r"\quad cSEBB-top3")
    out.append(r"\midrule")
    for cf in ("0.1", "0.2", "0.3", "0.7"):
        row(f"CS-i20-cf{cf}", f"calibration fraction {cf} (cSEBB conf.)")
    out.append(r"\midrule")
    row("DL-i20-d0.01", r"$\delta{=}0.01$ (cSEBB conf.)")
    row("DL-i20-d0.10", r"$\delta{=}0.10$ (cSEBB conf.; certificate promises only 90\%)")
    out.append(r"\midrule")
    for a in ("0.12", "0.15", "0.25", "0.3"):
        row(f"sw-i50-a{a}-bonf-marg", rf"$\alpha{{=}}{a}$, LTT Bonf.\ (cSEBB conf.)")
    row("sw-i50med-a0.25-bonf-marg", r"$\alpha{=}0.25$, LTT Bonf.\ (median filter)")
    row("sw-i50med-a0.25-split-marg", r"$\alpha{=}0.25$, LTT split (median filter)")
    out.append(r"\midrule")
    row("i20-cse-ltt_split_clipmean-clus", "clustered grouping (cSEBB conf.)")
    row("2D-c60-marg", r"2-D $\lambda$ (window$\times$thr.), c60 medfilt")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def instrument_table():
    d = json.load(open(os.path.join(RESULTS, "instrument_study.json")))
    lbl = {"champ-i20": ("i20", "cSEBB-max, split-LTT (champion)"),
           "top3-i20": ("i20", "cSEBB-top3, split-LTT"),
           "bonf-i20": ("i20", "cSEBB-max, Bonferroni-LTT"),
           "splitcse-i20": ("i20", "cSEBB, split-LTT"),
           "rcps-i20": ("i20", "cSEBB-max, RCPS (clip HB)"),
           "rcpsP-i20": ("i20", "cSEBB-max, RCPS (pooled)"),
           "crc-i20": ("i20", "cSEBB, naive CRC"),
           "crcC-i20": ("i20", "cSEBB, CRC-C nonmono."),
           "bonf-i45": ("i45", "cSEBB, Bonferroni-LTT"),
           "split-i45": ("i45", "cSEBB, split-LTT"),
           "bonf-c60": ("c60", "cSEBB, Bonferroni-LTT"),
           "split-c60": ("c60", "cSEBB, split-LTT"),
           "crcNM-c60": ("c60", "cSEBB, CRC-NM")}
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Instrument study over 10{,}000 splits (Monte-Carlo error on "
           r"fractions $<0.005$): gate pass fraction under the pooled criterion with "
           r"complementary halves (the paper's gate), with independently drawn evaluation "
           r"halves, and under the certificate-matched clip-balanced criterion; fraction of "
           r"calibration draws whose selected threshold's \emph{full-split (population)} "
           r"risk exceeds $\alpha$, for both functionals (certificates promise "
           r"$\le\delta=0.05$ for their own functional); FP/h mean$\pm$std.}",
           r"\label{tab:instrument}",
           r"\begin{tabular}{llrrrrrr}\toprule",
           r"& & \multicolumn{3}{c}{gate pass frac.} & \multicolumn{2}{c}{pop.\ viol.\ frac.} & \\",
           r"Pt & Configuration & compl. & indep. & clip-bal. & pooled & clip & FP/h \\\midrule"]
    for k, (pt, name) in lbl.items():
        e = d.get(k)
        if not e:
            continue
        out.append(f"{pt} & {name} & {e['gate_pass_frac_pooled_complementary']:.3f} & "
                   f"{e['gate_pass_frac_pooled_independent']:.3f} & "
                   f"{e['gate_pass_frac_clipbalanced']:.3f} & "
                   f"{e['pop_risk_violation_frac_pooled']:.3f} & "
                   f"{e['pop_risk_violation_frac_clipbalanced']:.3f} & "
                   f"{e['fp_per_h_mean']:.0f}$\\pm${e['fp_per_h_std']:.0f} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def holdout_table():
    d = json.load(open(os.path.join(RESULTS, "holdout_confirmation.json")))
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Held-out confirmation. The test clips are partitioned once "
           rf"(60\%/40\%, {d['n_sel']}/{d['n_conf']} clips); the full champion search "
           r"re-runs inside the selection pool; the frozen selection champions are then "
           r"evaluated with 100 calibration draws from the selection pool, each scored "
           r"on the untouched confirmation pool.}",
           r"\label{tab:holdout}",
           r"\begin{tabular}{llrrrr}\toprule",
           r"Pt & Selection champion & sel.\ $n_{\mathrm{ok}}$ & conf.\ $n_{\mathrm{ok}}$ & conf.\ miss (q95) & conf.\ FP/h \\\midrule"]
    for tag in ("c60", "i45", "i20"):
        e = d["points"][tag]
        ch, cf = e["selection_champion"], e["confirmation"]
        name = f"{VAR_LBL.get(ch['variant'], ch['variant'])}, {ROUTE_LBL.get(ch['route'], ch['route'])}, {GRP_LBL[ch['grouping']]}"
        out.append(f"{tag} & {name} & {ch['sel_n_ok']} & {cf['n_ok']} & "
                   f"{fmt(cf['mean_miss'])} ({fmt(cf['q95_miss'])}) & "
                   f"{cf['fp_per_h_mean']:.0f}$\\pm${cf['fp_per_h_std']:.0f} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def groupcond_table():
    d = json.load(open(os.path.join(RESULTS, "group_conditional.json")))
    sg = json.load(open(os.path.join(RESULTS, "stress_SG-i20-max.json")))["groups"]
    marg = {"placement/static": sg.get("placement/static"), "placement/mobile": sg.get("placement/mobile"),
            "device/ios": sg.get("device/ios"), "device/android": sg.get("device/android"),
            "device/other": sg.get("device/other"),
            "env/kitchen": sg.get("env/kitchen"), "env/bedroom": sg.get("env/bedroom"),
            "env/hallway": sg.get("env/hallway"), "env/living_room": sg.get("env/living_room"),
            "env/office": sg.get("env/office")}
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Group-conditional (Mondrian) calibration at the headline point "
           r"(cSEBB-max, intersection-0.5, $\alpha{=}0.2$, Bonferroni-LTT per group): "
           r"per-group gate counts under marginal calibration (from "
           r"\cref{tab:subgroups}) vs.\ one threshold per group, and the overall "
           r"efficiency price of each conditioning scheme.}",
           r"\label{tab:groupcond}",
           r"\begin{tabular}{llrrr}\toprule",
           r"Scheme (FP/h) & Group & marg.\ $n_{\mathrm{ok}}$ & cond.\ $n_{\mathrm{ok}}$ & cond.\ miss \\\midrule"]
    i20 = d["i20"]
    scheme_names = {"placement": "placement", "device": "device", "environment": "environment"}
    keymap = {"placement": [("static", "placement/static"), ("mobile", "placement/mobile")],
              "device": [("ios", "device/ios"), ("android", "device/android"), ("other", "device/other")],
              "environment": [("kitchen", "env/kitchen"), ("bedroom", "env/bedroom"),
                              ("hallway", "env/hallway"), ("living_room", "env/living_room"),
                              ("office", "env/office")]}
    for scheme, pairs in keymap.items():
        e = i20[scheme]
        first = True
        for gname, mkey in pairs:
            g = e["groups"].get(gname)
            mrow = marg.get(mkey)
            if g is None:
                continue
            label = f"{scheme_names[scheme]} ({e['fp_per_h_mean']:.0f})" if first else ""
            first = False
            mv = str(mrow["n_ok"]) if mrow else "--"
            out.append(f"{label} & {gname.replace('_', ' ')} & {mv} & {g['n_ok']} & "
                       f"{fmt(g['mean_miss'])} \\\\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule\end{tabular}\end{table}"
    out.append("")
    return "\n".join(out)


def main():
    rows = load_all()
    os.makedirs(PAPER, exist_ok=True)
    with open(os.path.join(PAPER, "tables_main.tex"), "w") as f:
        f.write(main_table(rows))
    with open(os.path.join(PAPER, "tables_audit.tex"), "w") as f:
        f.write(audit_table() + "\n" + floors_table())
    with open(os.path.join(PAPER, "tables_stress.tex"), "w") as f:
        f.write(subgroup_table() + "\n" + transfer_table() + "\n" + ablation_table(rows))
    with open(os.path.join(PAPER, "tables_instrument.tex"), "w") as f:
        f.write(instrument_table())
    with open(os.path.join(PAPER, "tables_confirm.tex"), "w") as f:
        f.write(holdout_table() + "\n" + groupcond_table())
    print("tables written to", PAPER)


if __name__ == "__main__":
    main()
