"""Reproduce every analysis of Naresh & Dullam (2026), ETRR 18:59, from the
public STATS19 files.

    python run_paper.py                       # full run, OSM road network (needs internet)
    python run_paper.py --graph synthetic     # offline test of the routing code
    python run_paper.py --fast                # skip random forests, CV on 200k rows

Outputs (outputs/): tables/*.csv, figures/*.png|html, models/system_bundle.joblib,
results.json and CLAIMS_AUDIT.md (every claim of the article next to the value
this code produces). Nothing is copied from the article into any result.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rsr import config as C
from rsr import data, figures, models, routing, spatial, live

warnings.filterwarnings("ignore")


def banner(s):
    print("\n" + "=" * 90 + f"\n{s}\n" + "=" * 90, flush=True)


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cv-rows", type=int, default=None,
                    help="rows of the training split used for 5-fold CV (default: all)")
    ap.add_argument("--graph", default="osm", help="osm | synthetic | path/to.graphml")
    ap.add_argument("--center", default="51.5074,-0.1278", help="lat,lon of routing area")
    ap.add_argument("--radius-m", type=int, default=6000)
    ap.add_argument("--od-pairs", type=int, default=300)
    ap.add_argument("--graph-cache", default=None,
                    help="GraphML file: the OSM network is downloaded once, saved here and "
                         "re-used, so Table 9 can be re-run on an identical network")
    ap.add_argument("--case-origin", default=None, help="lat,lon of a named case-study origin")
    ap.add_argument("--case-dest", default=None, help="lat,lon of a named case-study destination")
    ap.add_argument("--skip-table3", action="store_true")
    ap.add_argument("--skip-core11", action="store_true",
                    help="skip the eleven-original-predictors sensitivity run")
    ap.add_argument("--sample", type=int, default=None, help="TEST ONLY: random subsample of rows")
    ap.add_argument("--export-dataset", action="store_true",
                    help="write the analytical dataset (large) and test IDs")
    args = ap.parse_args()
    C.ensure_dirs()
    T0 = time.time()
    R = {"config": {"random_state": C.RANDOM_STATE, "test_size": C.TEST_SIZE,
                    "mlp": {k: str(v) for k, v in C.MLP_PARAMS.items()},
                    "cell_size_m": C.CELL_SIZE_M, "alpha": C.ROUTE_ALPHA, "beta": C.ROUTE_BETA}}

    # ------------------------------------------------------------------ 1. data
    banner("1. DATA INTEGRATION AND PREPROCESSING (Sect. 3.1, 3.4.1-3.4.2)")
    D, audit = data.load_dataset()
    if args.sample:
        D = D.sample(args.sample, random_state=C.RANDOM_STATE).reset_index(drop=True)
        print(f"*** TEST MODE: random sample of {args.sample:,} rows ***")
    audit.to_csv(C.TABLE_DIR / "data_audit.csv", index=False)
    dist = D["y"].map(C.CLASS_NAMES).value_counts().reindex(C.CLASS_NAMES.values())
    t4 = pd.DataFrame({"records": dist, "share_%": (100 * dist / dist.sum()).round(2)})
    t4.to_csv(C.TABLE_DIR / "table4_severity_distribution.csv")
    print(t4)
    R["dataset"] = {"rows": int(len(D)), "class_counts": dist.to_dict(),
                    "years": [int(D.Year.min()), int(D.Year.max())]}

    banner("2. EXPLORATORY ANALYSIS (Sect. 3.4.3, 4.2.1, 4.2.6; Figs 3, 6)")
    corr_cols = ["Longitude", "Latitude", "Accident_Severity", "Day_of_Week", "Road_Type",
                 "Speed_limit", "Junction_Detail", "Light_Conditions", "Weather_Conditions",
                 "Road_Surface_Conditions", "Urban_or_Rural_Area",
                 "Did_Police_Officer_Attend_Scene_of_Accident", "Hour", "Age_of_Driver",
                 "Age_of_Vehicle", "Engine_Capacity_(CC)", "Age_of_Casualty"]
    _, corr = figures.fig3_correlation(D[corr_cols].apply(pd.to_numeric, errors="coerce"))
    corr.to_csv(C.TABLE_DIR / "fig3_correlation_matrix.csv")
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    R["correlation"] = {"max_abs_offdiag": float(off.max().max()),
                        "share_pairs_abs_lt_0.3": float((off.stack() < 0.3).mean())}
    _, speed = figures.fig6_speed(D)
    speed.to_csv(C.TABLE_DIR / "fig6_speed_zones.csv")
    R["speed_zones_%"] = speed.to_dict()
    print("Speed zones (%):", speed.to_dict())

    # records that make the run auditable: exact predictor lists and package versions
    (C.OUTPUT_DIR / "feature_lists.json").write_text(json.dumps(
        {k: dict(zip(("categorical", "numeric"), data.feature_lists(k)))
         for k in ("core11", "interface", "table3")}
        | {"forbidden_as_predictors": sorted(C.FORBIDDEN_FEATURES)}, indent=1))
    pkgs = {}
    for m in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "networkx", "osmnx",
              "pyproj", "matplotlib", "joblib"):
        try:
            pkgs[m] = __import__(m).__version__
        except Exception:
            pkgs[m] = "not installed"
    (C.OUTPUT_DIR / "environment.json").write_text(json.dumps(
        {"python": sys.version, "platform": platform.platform(), "packages": pkgs,
         "command": " ".join(sys.argv)}, indent=1))

    # ------------------------------------------------------------------ 2. split
    idx_tr, idx_te = train_test_split(D.index, test_size=C.TEST_SIZE, stratify=D["y"],
                                      random_state=C.RANDOM_STATE)
    Dtr, Dte = D.loc[idx_tr], D.loc[idx_te]
    ytr, yte = Dtr["y"], Dte["y"]
    pd.DataFrame({"subset": ["train", "test"], "rows": [len(Dtr), len(Dte)]}).to_csv(
        C.TABLE_DIR / "split_summary.csv", index=False)
    Dte[[C.ID_COL]].to_csv(C.OUTPUT_DIR / "test_accident_ids.csv", index=False)
    R["split"] = {"train": int(len(Dtr)), "test": int(len(Dte)),
                  "test_support": yte.map(C.CLASS_NAMES).value_counts().to_dict()}
    print("Split:", R["split"])
    if args.export_dataset:
        cat3, num3 = data.feature_lists("table3")
        D[[C.ID_COL, C.TARGET] + cat3 + num3 + ["Location_Easting_OSGR", "Location_Northing_OSGR",
                                                 "Date"]].to_csv(
            C.OUTPUT_DIR / "analytical_dataset.csv.gz", index=False, compression="gzip")

    cat, num = data.feature_lists("interface")
    X = data.to_model_frame(D, "interface")
    Xtr, Xte = X.loc[idx_tr], X.loc[idx_te]
    rank = models.feature_ranking(Xtr, ytr, cat, num)
    rank.to_csv(C.TABLE_DIR / "feature_ranking_anova_mi.csv", index=False)
    print("Top features by mutual information:", rank.feature.head(5).tolist())

    # ------------------------------------------------------------------ 3. models
    banner("3. SEVERITY CLASSIFICATION (Sect. 4.2, 4.4; Tables 7, 8, 15; Figs 4, 5, 12)")
    summary, fitted, probas = [], {}, {}
    # The checkpoint folder is keyed by a fingerprint of the data, the settings and the
    # modelling code, so a checkpoint from a different sample, seed or code version can
    # never be silently re-used.
    fp = hashlib.sha1(json.dumps({
        "rows": len(D), "ids": int(pd.util.hash_pandas_object(D[C.ID_COL], index=False).sum() % 2**63),
        "sample": args.sample, "fast": args.fast, "seed": C.RANDOM_STATE, "test": C.TEST_SIZE,
        "mlp": R["config"]["mlp"],
        "code": [hashlib.sha1((Path(__file__).parent / f).read_bytes()).hexdigest()
                 for f in ("rsr/config.py", "rsr/data.py", "rsr/models.py")]}, sort_keys=True
    ).encode()).hexdigest()[:12]
    cache = C.OUTPUT_DIR / "cache" / fp
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint folder: {cache}")
    KEEP = ("Proposed MLP", "Proposed MLP (class-weighted)")   # needed later; others freed

    FS_LABEL = {"interface": "interface (Fig. 7)", "table3": "Table 3 variables",
                "core11": "11 original predictors"}

    def run_model(name, est, w, c_, n_, Xa, Xb, fs):
        """Fit + evaluate one model, with an on-disk checkpoint so a crash resumes."""
        key = (name + "__" + fs).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        jf, pf = cache / f"{key}.json", cache / f"{key}.joblib"
        if jf.exists() and (name not in KEEP or fs != "interface" or pf.exists()):
            res = json.load(open(jf))
            pipe = joblib.load(pf) if pf.exists() else None
            print(f"  [cached] {res['model']}")
            return res, pipe
        pipe, secs = models.fit(est, w, c_, n_, Xa, ytr)
        res, proba = models.evaluate(name if fs == "interface" else f"{name} [{FS_LABEL[fs]}]",
                                     pipe, Xb, yte)
        res.update(train_seconds=secs, weighting=w or "none", feature_set=FS_LABEL[fs])
        json.dump(res, open(jf, "w"), default=jdefault)
        if name in KEEP and fs == "interface":
            joblib.dump(pipe, pf)
            np.save(cache / f"{key}_proba.npy", proba)
        else:
            pipe = None
        pc = res["per_class"]
        print(f"  {res['model']:<45} acc {res['accuracy']:.4f}  macro-F1 {res['macro_f1']:.4f}  "
              f"bal-acc {res['balanced_accuracy']:.4f}  recall F/S/Sl "
              f"{pc['Fatal']['recall']:.3f}/{pc['Serious']['recall']:.3f}/{pc['Slight']['recall']:.3f}"
              f"  ({secs:.0f}s)", flush=True)
        return res, pipe

    for name, (est, w) in models.model_zoo(args.fast).items():
        res, pipe = run_model(name, est, w, cat, num, Xtr, Xte, "interface")
        summary.append(res)
        if pipe is not None:
            fitted[name] = pipe
        del pipe
    for name in KEEP:
        key = (name + "__interface").replace(" ", "_").replace("(", "").replace(")", "")
        probas[name] = np.load(cache / f"{key}_proba.npy")
    sens_sets = []
    if not args.skip_table3:
        sens_sets.append(("table3", ("Proposed MLP", "Proposed MLP (class-weighted)", "LightGBM",
                                     "LightGBM (class-weighted)")))
    if not args.skip_core11:
        sens_sets.append(("core11", ("Logistic regression", "Proposed MLP", "LightGBM")))
    for fs, names in sens_sets:
        cs_, ns_ = data.feature_lists(fs)
        Xs = data.to_model_frame(D, fs)
        zoo = models.model_zoo(fast=True)
        for name in names:
            if name in zoo:
                est, w = zoo[name]
                res, _ = run_model(name, est, w, cs_, ns_, Xs.loc[idx_tr], Xs.loc[idx_te], fs)
                summary.append(res)
        del Xs

    flat = pd.DataFrame([{
        "model": r["model"], "feature_set": r["feature_set"], "class_weighting": r["weighting"],
        "accuracy": r["accuracy"], "accuracy_ci_low": r["accuracy_ci"][0],
        "accuracy_ci_high": r["accuracy_ci"][1], "balanced_accuracy": r["balanced_accuracy"],
        "macro_precision": r["macro_precision"], "macro_recall": r["macro_recall"],
        "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"],
        **{f"recall_{c}": r["per_class"][c]["recall"] for c in figures.CLS},
        **{f"auc_{c}": r["per_class"][c]["roc_auc"] for c in figures.CLS},
        "train_seconds": r.get("train_seconds"), "n_test": r["n_test"]} for r in summary])
    flat.to_csv(C.TABLE_DIR / "table15_model_comparison.csv", index=False)
    json.dump(summary, open(C.TABLE_DIR / "model_results_full.json", "w"), indent=1, default=jdefault)

    mlp = next(r for r in summary if r["model"] == "Proposed MLP")
    t8 = pd.DataFrame([{"class": c, **{k: mlp["per_class"][c][k] for k in
                                       ("precision", "recall", "f1", "support")}} for c in figures.CLS]
                      + [{"class": "accuracy", "f1": mlp["accuracy"], "support": mlp["n_test"]},
                         {"class": "macro avg", "precision": mlp["macro_precision"],
                          "recall": mlp["macro_recall"], "f1": mlp["macro_f1"], "support": mlp["n_test"]},
                         {"class": "weighted avg", "precision": mlp["weighted_precision"],
                          "recall": mlp["weighted_recall"], "f1": mlp["weighted_f1"],
                          "support": mlp["n_test"]}])
    t8.to_csv(C.TABLE_DIR / "table8_mlp_classification.csv", index=False)
    print(t8.round(4).to_string(index=False))
    for r in summary:
        pd.DataFrame(r["confusion_matrix"], index=[f"true_{c}" for c in figures.CLS],
                     columns=[f"pred_{c}" for c in figures.CLS]).to_csv(
            C.TABLE_DIR / f"confusion_{r['model'].replace(' ', '_').replace('/', '')}.csv")
    figures.fig4_classwise(mlp, "Proposed MLP")
    figures.fig5_roc(models.roc_points(yte, probas["Proposed MLP"]), "Proposed MLP")
    figures.fig12_models(flat[flat.feature_set.str.startswith("interface")])
    R["models"] = flat.to_dict(orient="records")

    banner("4. FIVE-FOLD CROSS-VALIDATION OF THE PROPOSED MLP (Sect. 4.2.2, 4.3.1)")
    cv_rows = args.cv_rows or (200_000 if args.fast else None)
    cv_file = cache / f"cv_{cv_rows or 'all'}.csv"
    if cv_file.exists():
        cv = pd.read_csv(cv_file)
    else:
        cv = models.cross_validate_mlp(Xtr, ytr, cat, num, max_rows=cv_rows)
        cv.to_csv(cv_file, index=False)
    cv.to_csv(C.TABLE_DIR / "mlp_5fold_cv.csv", index=False)
    R["cv"] = {"rows": int(cv.rows_train.iloc[0] + cv.rows_val.iloc[0]),
               "accuracy_mean": cv.accuracy.mean(), "accuracy_sd": cv.accuracy.std(),
               "macro_f1_mean": cv.macro_f1.mean(), "macro_f1_sd": cv.macro_f1.std()}

    # ------------------------------------------------------------------ 5. spatial
    banner("5. SPATIAL RISK AND HOTSPOTS (Sect. 3.4.2, 4.3; Figs 8, 13)")
    grid = spatial.HotspotGrid().fit(Dtr)
    grid_te = spatial.HotspotGrid().fit(Dte)
    hs = grid.hotspots()
    share = hs["n_ksi"].sum() / grid.cells["n_ksi"].sum()
    area_share = len(hs) / len(grid.cells)
    hs.head(500).to_csv(C.TABLE_DIR / "hotspot_cells_top500.csv")
    figures.fig8_hotspots(grid)
    lat0, lon0 = map(float, args.center.split(","))
    figures.fig8_hotspots(grid, region=(lat0 - 0.12, lat0 + 0.12, lon0 - 0.2, lon0 + 0.2))
    figures.fig8_folium(grid, (lat0, lon0))
    # do training hotspots predict where held-out KSI accidents occur?
    te_rate = grid.ksi_rate_at(Dte["Location_Easting_OSGR"], Dte["Location_Northing_OSGR"])
    hs_thr = hs["ksi_rate_smoothed"].min() if len(hs) else np.inf
    in_hs = te_rate >= hs_thr
    sev_te = np.isin(yte, C.SEVERE_CLASSES)
    R["hotspots"] = {"cells_with_accidents": int(len(grid.cells)), "hotspot_cells": int(len(hs)),
                     "share_of_training_KSI_in_hotspots_%": 100 * share,
                     "share_of_cells_%": 100 * area_share,
                     "share_of_heldout_KSI_in_hotspots_%": 100 * float(in_hs[sev_te].mean()),
                     "share_of_heldout_slight_in_hotspots_%": 100 * float(in_hs[~sev_te].mean())}
    print(R["hotspots"])

    # risk system uses the class-weighted MLP (same Table 7 architecture)
    sys_name = "Proposed MLP (class-weighted)"
    ctx = spatial.SeverityContext(fitted[sys_name], list(X.columns), cat, Xtr)
    val_idx = Dtr.sample(min(300_000, int(0.15 * len(Dtr))), random_state=1).index
    ref_idx = Dtr.index.difference(val_idx)
    # Two candidate alert scores; the one with the higher validation F1 for
    # detecting Fatal/Serious accidents is used by the live system.
    cands = {}
    for label, sp_flag in (("combined (spatial x model)", True), ("model only P(severe)", False)):
        sc = spatial.RiskScorer(grid, ctx, spatial=sp_flag).fit_reference(Dtr.loc[ref_idx], Xtr.loc[ref_idx])
        Rv = sc.score(sc.raw(grid.ksi_rate_loo(Dtr.loc[val_idx]), ctx.p_severe(Xtr.loc[val_idx])))
        t_, f_ = spatial.select_tau(Rv, ytr.loc[val_idx])
        cands[label] = (sc, t_, f_)
        print(f"  validation: {label:<28} tau* = {t_}  F1 = {f_:.3f}")
    alert_label = max(cands, key=lambda k: cands[k][2])
    scorer, tau, f1v = cands[alert_label]
    tabs = []
    for label, (sc, t_, _) in cands.items():
        tb = spatial.threshold_table(sc.score_records(Dte, Xte), yte, sorted(set(C.THRESHOLDS) | {t_}))
        tb.insert(0, "score", label)
        tabs.append(tb)
    t10 = pd.concat(tabs, ignore_index=True)
    t10.to_csv(C.TABLE_DIR / "table10_threshold_sensitivity.csv", index=False)
    print(t10.round(2).to_string(index=False))
    print(f"Alert score used by the system: {alert_label}; tau = {tau} (validation F1 {f1v:.3f})")
    R["threshold"] = {"alert_score": alert_label, "tau_selected": tau,
                      "validation": {k: {"tau": v[1], "f1": v[2]} for k, v in cands.items()},
                      "table10": t10.to_dict(orient="records")}
    figures.fig_threshold(t10[t10.score == alert_label])
    figures.fig13_predicted_density(Dte["Latitude"].values, Dte["Longitude"].values,
                                    ctx.p_severe(Xte))

    profile = {c: (Xtr[c].mode().iloc[0] if c in cat else float(Xtr[c].median())) for c in X.columns}
    bundle = C.MODEL_DIR / "system_bundle.joblib"
    live.save_bundle(bundle, fitted[sys_name], list(X.columns), cat, grid, ctx, scorer, tau,
                     {"model": sys_name, "trained_rows": int(len(Dtr)), "alert_score": alert_label,
                      "default_profile": {k: v for k, v in profile.items()
                                          if k not in ("Latitude", "Longitude")}})
    joblib.dump(fitted["Proposed MLP"], C.MODEL_DIR / "mlp_proposed.joblib")
    print("System bundle saved:", bundle)

    # ------------------------------------------------------------------ 6. routing
    banner("6. RISK-AWARE ROUTING vs SHORTEST PATH (Sect. 3.4.8, 4.3.2; Table 9, Fig. 10)")
    graph_file = None
    if args.graph == "osm":
        graph_file = args.graph_cache or str(C.OUTPUT_DIR / "cache" /
                                             f"osm_{lat0}_{lon0}_{args.radius_m}.graphml")
        reused = os.path.exists(graph_file)
        G = routing.load_osm_graph((lat0, lon0), args.radius_m, cache_path=graph_file)
        graph_note = (f"OpenStreetMap drive network, {args.radius_m} m around {lat0},{lon0} "
                      f"({'re-used from' if reused else 'downloaded and saved to'} {Path(graph_file).name})")
    elif args.graph == "synthetic":
        G = routing.synthetic_grid_graph((lat0, lon0))
        graph_note = "SYNTHETIC TEST GRID (code check only - not a result)"
    else:
        import osmnx as ox
        G = ox.load_graphml(args.graph)
        graph_file = args.graph
        graph_note = f"graph file {args.graph}"
    conditions = {**profile, "Weather_Conditions": "1", "Road_Surface_Conditions": "1",
                  "Light_Conditions": "1"}
    routing.annotate_graph(G, grid, ctx, conditions, grid_heldout=grid_te)
    od = routing.evaluate_od_pairs(G, n_pairs=args.od_pairs)
    od.to_csv(C.TABLE_DIR / "routing_od_pairs.csv", index=False)
    summ = routing.summarise_od(od)
    sens = []
    for a, b in C.ALPHA_BETA_GRID:
        s = routing.summarise_od(routing.evaluate_od_pairs(G, n_pairs=min(100, args.od_pairs),
                                                           alpha=a, beta=b))
        sens.append({"alpha": a, "beta": b, **s})
    pd.DataFrame(sens).to_csv(C.TABLE_DIR / "routing_alpha_beta_sensitivity.csv", index=False)
    changed = od[~od.same_route]
    # representative example: the changed pair closest to the median held-out reduction
    if len(changed):
        med = changed["risk_heldout_reduction_%"].median()
        ex = changed.loc[(changed["risk_heldout_reduction_%"] - med).abs().idxmin()]
    else:
        ex = od.iloc[0]
    sp, ra, a_, b_ = routing.compare_routes(G, ex.origin, ex.target)
    figures.fig10_routes(G, sp, ra, f"Fig. 10  Example route pair ({graph_note})")
    t9 = pd.DataFrame({
        "metric": ["Travel distance (km)", "Estimated travel time (min)",
                   "Cumulative risk (expected KSI/yr, training hotspots)",
                   "Cumulative risk on held-out accidents", "Risk reduction on held-out (%)"],
        "example_shortest": [a_["distance_km"], a_["time_min"], a_["risk"], a_["risk_heldout"], None],
        "example_risk_aware": [b_["distance_km"], b_["time_min"], b_["risk"], b_["risk_heldout"],
                               100 * (1 - b_["risk_heldout"] / a_["risk_heldout"]) if a_["risk_heldout"] else None],
        "all_pairs_shortest_median": [od.sp_distance_km.median(), od.sp_time_min.median(),
                                      od.sp_risk.median(), od.sp_risk_heldout.median(), None],
        "all_pairs_risk_aware_median": [od.ra_distance_km.median(), od.ra_time_min.median(),
                                        od.ra_risk.median(), od.ra_risk_heldout.median(),
                                        summ["risk_reduction_%_total_heldout_accidents"]]})
    t9.to_csv(C.TABLE_DIR / "table9_routing.csv", index=False)
    print(t9.round(3).to_string(index=False))
    print(summ)
    R["routing"] = {"graph": graph_note, "graph_meta": routing.graph_metadata(G, graph_file),
                    "alpha": C.ROUTE_ALPHA, "beta": C.ROUTE_BETA, "od_seed": C.RANDOM_STATE,
                    "summary": summ, "alpha_beta": sens,
                    "example": {"shortest": a_, "risk_aware": b_}}
    if args.case_origin and args.case_dest:
        # Table 9 in the paper's own format: ONE named origin-destination pair.
        cs = routing.case_study(G, tuple(map(float, args.case_origin.split(","))),
                                tuple(map(float, args.case_dest.split(","))))
        pd.DataFrame({
            "metric": ["Travel distance (km)", "Estimated travel time (min)",
                       "Cumulative risk (expected KSI/yr, training hotspots)",
                       "Cumulative risk on held-out accidents"],
            "shortest": [cs["shortest"][k] for k in ("distance_km", "time_min", "risk", "risk_heldout")],
            "risk_aware": [cs["risk_aware"][k] for k in ("distance_km", "time_min", "risk", "risk_heldout")],
        }).to_csv(C.TABLE_DIR / "table9_case_study.csv", index=False)
        R["routing"]["case_study"] = {k: v for k, v in cs.items() if not k.startswith("path_")}
        print("Named case study:", R["routing"]["case_study"])
    lk = Path(os.environ.get("ACCIDENT_OUTPUT_DIR", "leakage_diagnostic_outputs")) / "leakage_diagnostic_summary.csv"
    if lk.exists():
        L_ = pd.read_csv(lk)
        R["leakage_negative_control"] = {"source_file": str(lk), "rows": L_.to_dict(orient="records"),
                                         "script": "leakage_diagnostic_pipeline.py"}
    else:
        R["leakage_negative_control"] = "not run: python leakage_diagnostic_pipeline.py"
    R["runtime_minutes"] = (time.time() - T0) / 60
    json.dump(R, open(C.OUTPUT_DIR / "results.json", "w"), indent=1, default=jdefault)
    write_audit(R)
    banner(f"DONE in {R['runtime_minutes']:.1f} min -> {C.OUTPUT_DIR.resolve()}")


def write_audit(R):
    """CLAIMS_AUDIT.md: each quantitative claim of the article vs this code."""
    m = {r["model"]: r for r in R["models"]}
    mlp, base = m["Proposed MLP"], m["Majority baseline"]
    rows = []

    def add(claim, paper, ours, verdict):
        rows.append(f"| {claim} | {paper} | {ours} | {verdict} |")

    n = R["dataset"]["rows"]
    add("Records analysed (Sect. 3.1)", "≈60,000", f"{n:,}",
        "Different: the article's original notebook used a 61,450-row subset")
    sup = R["split"]["test_support"]
    add("Table 8 test support", "500 / 450 / 550 (1,500)",
        f"{sup.get('Fatal', 0):,} / {sup.get('Serious', 0):,} / {sup.get('Slight', 0):,}",
        "Not reproduced (natural class mix)")
    add("MLP accuracy (Abstract, Table 8)", "91.2%",
        f"{100 * mlp['accuracy']:.2f}% (95% CI {100 * mlp['accuracy_ci_low']:.2f}–{100 * mlp['accuracy_ci_high']:.2f}); "
        f"majority baseline {100 * base['accuracy']:.2f}%",
        "Not reproduced" if abs(100 * mlp["accuracy"] - 91.2) > 1 else "Reproduced")
    add("MLP macro P / R / F1", "0.91 / 0.91 / 0.91",
        f"{mlp['macro_precision']:.2f} / {mlp['macro_recall']:.2f} / {mlp['macro_f1']:.2f}",
        "Not reproduced" if mlp["macro_f1"] < 0.85 else "Reproduced")
    add("Fatal / Serious recall", "0.92 / 0.90 (Table 8)",
        f"{mlp['recall_Fatal']:.2f} / {mlp['recall_Serious']:.2f}",
        "Not reproduced" if mlp["recall_Fatal"] < 0.5 else "Reproduced")
    add("ROC AUC (Fig. 5)", "0.840 / 0.806 / 0.741",
        f"Fatal {mlp['auc_Fatal']:.3f} / Serious {mlp['auc_Serious']:.3f} / Slight {mlp['auc_Slight']:.3f}",
        "Compare (class order in the article is ambiguous)")
    order = sorted([r for r in R["models"] if r["feature_set"].startswith("interface")
                    and r["model"] != "Majority baseline"], key=lambda r: -r["accuracy"])
    add("MLP best of RF / XGBoost / LightGBM (Table 15)", "MLP 91.2 > LGBM 90.7 > XGB 90.3 > RF 89.4",
        "; ".join(f"{r['model']} {100 * r['accuracy']:.2f}" for r in order[:6]),
        ("Not reproduced: every unweighted model is within the baseline's 95% CI"
         if all(base["accuracy_ci_low"] <= r["accuracy"] <= base["accuracy_ci_high"] + 0.002
                for r in order if r["class_weighting"] == "none")
         else "Not reproduced: differs from the article's ranking; see Table 15 output"))
    cv = R["cv"]
    add("5-fold CV consistency", "consistent across folds",
        f"acc {100 * cv['accuracy_mean']:.2f} ± {100 * cv['accuracy_sd']:.2f}%, macro-F1 "
        f"{cv['macro_f1_mean']:.3f} ± {cv['macro_f1_sd']:.3f} ({cv['rows']:,} rows)",
        "Reproduced (consistency)" if cv["accuracy_sd"] < 0.005 else "Not consistent across folds")
    sz = R["speed_zones_%"]
    add("Speed zones (Fig. 6)", "30: 77.6%, 40: 8.7%, 60: 8.7%, 70: 2.6%, 50: 2.1%, 20: 0.3%",
        ", ".join(f"{k}: {v}%" for k, v in sz.items()), "Compare")
    add("Correlations weak (Fig. 3)", "mostly weak, no strong multicollinearity",
        f"max |r| off-diagonal {R['correlation']['max_abs_offdiag']:.2f}; "
        f"{100 * R['correlation']['share_pairs_abs_lt_0.3']:.0f}% of pairs |r|<0.3", "Compare")
    h = R["hotspots"]
    add("Accidents concentrate in hotspots (Sect. 4.4.5)", "qualitative",
        f"{h['share_of_cells_%']:.1f}% of cells hold {h['share_of_training_KSI_in_hotspots_%']:.1f}% of KSI; "
        f"they capture {h['share_of_heldout_KSI_in_hotspots_%']:.1f}% of held-out KSI",
        "Supported" if h["share_of_heldout_KSI_in_hotspots_%"] > 1.5 * h["share_of_cells_%"]
        else "Weak: hotspots capture little more than their share of cells")
    t = R["threshold"]
    t07 = next((r for r in t["table10"] if abs(r["threshold"] - 0.7) < 1e-9
                and r["score"] == t["alert_score"]), None)
    add("Threshold 0.70: detection rate (Table 10)", "91.2%",
        f"{t07['detection_rate_%']:.1f}% of held-out KSI accidents flagged; precision "
        f"{t07['precision_%']:.1f}% vs base {t07['base_rate_%']:.1f}%; score: {t['alert_score']}; tau selected on validation = {t['tau_selected']}"
        if t07 else "n/a", "Different definition; see Table 10 output")
    r = R["routing"]["summary"]
    cs = R["routing"].get("case_study")
    add("Route-risk reduction (Table 9)", "40.2% (one case; +1.2 km)",
        f"{r['risk_reduction_%_total_heldout_accidents']:.1f}% on held-out accidents "
        f"({r['risk_reduction_%_total_training_hotspots']:.1f}% on the training hotspots the objective uses) "
        f"over {r['od_pairs']} OD pairs, +{r['extra_distance_%_total']:.1f}% distance; route changed in "
        f"{r['pairs_where_route_changed_%']:.0f}% of pairs ({R['routing']['graph']})"
        + (f"; named case study: {cs['risk_reduction_%_training_hotspots']:.1f}% (training hotspots), "
           f"{cs['risk_reduction_%_heldout_accidents']:.1f}% (held-out), {cs['extra_km']:+.2f} km" if cs else ""),
        "See routing outputs; the 40.2% is reproduced only if the case-study row matches")
    lkc = R.get("leakage_negative_control")
    if isinstance(lkc, dict):
        mm = next((x for x in lkc["rows"] if "LEAKAGE" in str(x.get("feature_set", ""))
                   and x["model"] == "Proposed MLP"), None)
        add("Leakage negative control (concern 6)", "reviewer: 96.01% with casualty outcome",
            f"MLP + Casualty_Severity: {100 * mm['accuracy']:.2f}% accuracy, macro-F1 {mm['macro_f1']:.3f}"
            if mm else "n/a", "Explains >90% accuracies; not a candidate model")
    else:
        add("Leakage negative control (concern 6)", "reviewer: 96.01% with casualty outcome",
            "not run (python leakage_diagnostic_pipeline.py)", "Pending")
    add("Tables 11–13 (latency, scalability, reliability)", "1.3/0.9/1.8/3.7 s; 98.9%; …",
        "measured by run_system.py from live runs (logs/system_logs.csv)", "Live; values vary by run")
    txt = ["# Claims audit — Naresh & Dullam (2026), ETRR 18:59", "",
           "Every value in the third column is produced by `run_paper.py` from the public STATS19 "
           "files; nothing is copied from the article.", "",
           "| Claim | Article | This code | Verdict |", "|---|---|---|---|", *rows]
    (C.OUTPUT_DIR / "CLAIMS_AUDIT.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
