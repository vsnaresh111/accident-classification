# Road-safety framework — full reproduction code

Reproduction materials for **Naresh & Dullam (2026), "Data-driven road safety
enhancement: neural network–based accident classification and safe route
identification using spatial network analysis", *European Transport Research
Review* 18:59, doi:10.1186/s12544-026-00824-2.**

The code regenerates every table and figure of the article from the public
STATS19 files and runs the complete decision-support system end to end. Every
number is computed at run time; **nothing is copied from the article**.
`outputs/CLAIMS_AUDIT.md` places each quantitative claim of the article next to
the value the code produces.

## Quick start

**Google Colab (recommended):** open `reproduce_paper.ipynb` and run all cells.

**Local:**
```bash
pip install -r requirements.txt
export ACCIDENT_DATA_DIR=/path/to/csvs       # accidents.csv, vehicles.csv, casualties.csv
python run_paper.py --graph osm              # full paper (1.5–3 h on 2 CPU cores)
                                             # (full_reproduction_pipeline.py is an alias of run_paper.py)
python run_paper.py --fast --graph osm       # ~20 min check (no random forests, CV on 200k rows)
# Table 9 in the paper's format (one named origin-destination pair), on a saved network:
python run_paper.py --graph osm --graph-cache outputs/cache/london.graphml \
       --case-origin "LAT,LON" --case-dest "LAT,LON"
python leakage_diagnostic_pipeline.py        # negative control (concern 6), run after or before run_paper
python run_system.py --destination "51.5033,-0.1196" --origin "51.5155,-0.0922"
python -m pytest -q tests                    # offline tests of the live system (mocked services)
```

## Other scripts

* `replicate_original_analysis.py` re-runs the original analysis notebook
  exactly: the first 200,000 rows (2005), an outer join giving one row per
  vehicle, and removal of rows with any −1. It shows where the article's
  "≈60,000 records", its 12,290-row test set and Fig. 6 came from, and what
  that analysis produced (all models at the 89.6% majority rate).
* `leakage_diagnostic_pipeline.py` is a negative control. It re-uses the `rsr` data
  preparation, split, preprocessing and models, and fits each model twice on the same split:
  without and with `Casualty_Severity`, so the only difference is that one predictor.
  `run_paper.py` reads its summary file into `results.json` and `CLAIMS_AUDIT.md`. **Not a
  candidate model.** The earlier version (different preparation) is kept in `legacy/` for
  traceability only.

## What produces what

| Article item | Produced by | Output |
|---|---|---|
| Table 4 (severity classes) | `run_paper.py` §1 | `tables/table4_severity_distribution.csv` |
| Tables 5–6 (preprocessing) | `rsr/data.py`, `rsr/models.py` | `tables/data_audit.csv` (row count at every step) |
| Fig. 3 (correlation) | §2 | `figures/fig3_correlation.png` |
| Sect. 3.4.3 (ANOVA, information gain) | §2 | `tables/feature_ranking_anova_mi.csv` |
| Fig. 6 (speed zones) | §2 | `figures/fig6_speed_zones.png` |
| Table 7 (MLP) | `rsr/config.py` `MLP_PARAMS` | — |
| Table 8, Fig. 4 | §3 | `tables/table8_mlp_classification.csv`, `figures/fig4_classwise_metrics.png` |
| Fig. 5 (ROC) | §3 | `figures/fig5_roc.png` |
| Table 15, Fig. 12 (model comparison) | §3 | `tables/table15_model_comparison.csv`, `figures/fig12_model_comparison.png` |
| 5-fold cross-validation | §4 | `tables/mlp_5fold_cv.csv` |
| Fig. 8 (accident-prone locations) | §5 | `figures/fig8_hotspots*.png`, `figures/fig8_hotspots_map.html` |
| Table 10 (threshold) | §5 | `tables/table10_threshold_sensitivity.csv` |
| Fig. 13 (predicted density) | §5 | `figures/fig13_predicted_density.png` |
| Table 9, Fig. 10 (routing) | §6 | `tables/table9_routing.csv`, `tables/routing_od_pairs.csv`, `tables/routing_alpha_beta_sensitivity.csv`, `figures/fig10_route_example.png` |
| Table 9 for one named pair | `run_paper.py --case-origin --case-dest` | `tables/table9_case_study.csv` (+ `results.json`) |
| Predictor lists, package versions | `run_paper.py` | `feature_lists.json`, `environment.json` |
| Figs 7, 9, 11 (interface, tracking, SMS) | `run_system.py` / notebook §3 | `route_map.html`, `logs/system_logs.csv` |
| Tables 11–13 (latency, scalability, reliability) | `run_system.py --tables`, `--benchmark` | `tables/table11_*.csv`, `table12_*.csv`, `table13_*.csv` |

## Methods

**Data (Sect. 3.1, 3.4.1–3.4.2).** All accidents in the three files. One row per
accident: the vehicle and the casualty with the lowest reference number;
pedestrian involvement from all casualties; left joins validated one-to-one.
STATS19 code −1 is treated as missing. Records without severity or valid
coordinates are removed (Table 6). From 2015 the Vehicles and Casualties files
carry one extra trailing column; `read_stats19()` keeps those rows (pandas'
`on_bad_lines="skip"` would silently drop every 2015 record).

**Classification (Sect. 3.3–3.4.5, 4.2, 4.4).** Stratified 80:20 split at
accident level, seed 42. Preprocessing as in Table 6: mean / mode imputation,
Min–Max scaling, one-hot encoding, all fitted on the training split only. MLP as
in Table 7: (100, 50), ReLU, Adam, learning rate 0.001, early stopping. Compared
with Random Forest, XGBoost, LightGBM and a majority-class baseline. Variants
marked *class-weighted* use balanced sample weights (training loss only). **No
resampling** is used anywhere, and the test set keeps the natural class mix.
`Casualty_Severity` is never a predictor of the reported models (enforced by an assertion). Two
predictor sets: the fields of the Fig. 7 interface, and all Table 3 variables.

**Spatial risk (Sect. 3.4.2, 4.3).** Training accidents are binned into 200 m
cells on the British National Grid. The KSI (killed or seriously injured) rate
per year is smoothed over the 3×3 neighbourhood. Hotspots are the top 1% of
accident cells. Whether hotspots generalise is checked against held-out
accidents.

**Segment risk and routing (Sect. 3.4.8).**
`r_e(t) = KSI exposure of segment e × RR(x_e, t)`, where RR is the relative
severity risk under the current conditions from the trained model. The route
minimises `Σ (α·d_e + β·r_e / r_ref)` (Dijkstra on the OpenStreetMap drive
network). The benefit is evaluated over many origin–destination pairs against
**held-out accidents**, so it is not circular. The live system scores the Google
Routes API alternatives with the same objective.

**High-risk detection (Sect. 3.4.6, 4.3.3).** The risk score R ∈ [0, 1] is an
empirical CDF over training accidents. Two candidates are compared on a
validation split (spatial × model, and model only). The one with the higher F1
for detecting Fatal/Serious accidents is used, with τ chosen on validation.
Table 10 reports alerts, detection rate and precision on the test split for a
range of thresholds.

**Live system (Sect. 3.2, 4.3.4).** GPS (browser geolocation in Colab, else IP,
else manual) → live weather (Open-Meteo, mapped to STATS19 codes) → Google Routes
API alternatives → risk scoring → recommendation → SMS through Twilio with
delivery receipt when R ≥ τ → log row. Tables 11–13 are computed from
`logs/system_logs.csv`. Live timings depend on network, place and time, so they
vary between runs.

## What is and is not reproducible in the routing / live-system results (concern 7)

* **Reproducible exactly (same data, same code, same network file):** classification,
  hotspots, thresholds and the routing evaluation. The OSM network is saved once as GraphML
  (`--graph-cache`); re-using that file (deposit it with the results) fixes the road network,
  and the origin-destination sample is seeded. `results.json` records the file's SHA-256.
* **Not reproducible number-for-number, by construction:** anything that calls a live service
  (browser/IP location, Open-Meteo weather, Google Routes alternatives, Twilio SMS): outputs and
  timings depend on place, time, traffic and network. What is reproducible is the *procedure*
  and the *decision logic*, tested offline with mocked services (`tests/`). Tables 11-13 are
  computed from `logs/system_logs.csv`; deposit that log with the paper.
* Runs with a typed-in origin (`gps_source = MANUAL`) never call GPS, so they are excluded
  from every GPS statistic in Tables 11 and 13.
* The route-risk figure is a property of the network, the accident data, alpha/beta and the
  chosen origin-destination pair. A single case study (Table 9) is one pair; the multi-pair
  evaluation reports the total reduction over many pairs, on held-out accidents and on the
  training hotspots the objective itself uses. Report which one a number refers to.

## Coverage and limits

* The accident data cover Great Britain (2005–2015). Spatial risk and alerts are
  only available there. For locations elsewhere, the system reports "outside
  coverage" and never raises an alert.
* `--graph synthetic` is a code test on an artificial street grid. It is **not**
  a result.

## Credentials

Set as environment variables or Colab secrets: `GOOGLE_MAPS_API_KEY` (Routes API
enabled), `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`,
`EMERGENCY_PHONE_NUMBER`. Without `--live-sms` / `dry_run=False` no SMS is
sent; the decision is still computed and logged.
