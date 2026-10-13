"""Paper-aligned earthquake directionality experiment using existing pipeline outputs.

Primary experiment follows the paper's magnitude >= 5 and inter-site distance <= 5 km.
With the current dataset (8 close pairs), the script reports that the paper protocol is
not statistically estimable rather than fitting a misleading model. A supplementary
100-km experiment applies an 80/10/10 split, Ridge alpha=1.5, and validation-based MLP
architecture/learning-rate selection. The 100-km results are explicitly NOT a paper reproduction.

Run in Colab after Drive is mounted:
python paper_aligned_experiment.py --out /content/drive/MyDrive/Earthquake_ML_MiniProject/outputs_final
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
TARGET = "angle_difference_deg"
GROUP = "event"

def metrics(y, pred):
    return {
        "mae_deg": float(mean_absolute_error(y, pred)),
        "rmse_deg": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
    }

def split_80_10_10(df, seed=SEED):
    """Paper-style random example split; use only as descriptive replication with caveat.
    Event grouping is not specified in the paper. Pairs sharing stations/events can therefore
    cross splits, so results must not be interpreted as event-generalization performance.
    """
    train, temp = train_test_split(df, test_size=0.2, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    return train.copy(), val.copy(), test.copy()

def make_model(estimator):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), estimator)

def run_models(df, feature_cols, do_grid=True):
    if len(df) < 10:
        return {"status": "insufficient_examples", "n_examples": int(len(df)),
                "message": "Too few examples for a defensible 80/10/10 model evaluation."}
    tr, va, te = split_80_10_10(df)
    Xtr, Xva, Xte = tr[feature_cols], va[feature_cols], te[feature_cols]
    ytr, yva, yte = tr[TARGET], va[TARGET], te[TARGET]
    results = {"n_train":len(tr), "n_validation":len(va), "n_test":len(te),
               "split":"random example-level 80/10/10; not event-held-out",
               "models":{}}
    # Paper benchmark: simple linear regression using only site separation.
    dist_col = ["station_distance_km"]
    benchmark = make_model(LinearRegression())
    benchmark.fit(Xtr[dist_col], ytr)
    results["models"]["distance_only_linear"] = metrics(yte, np.clip(benchmark.predict(Xte[dist_col]),0,90))
    ridge = make_model(Ridge(alpha=1.5))
    ridge.fit(Xtr, ytr)
    results["models"]["ridge_alpha_1_5"] = metrics(yte, np.clip(ridge.predict(Xte),0,90))

    best = None
    grid_results = []
    grid = ParameterGrid({
        "hidden_layer_sizes": [(16,), (32,), (64,), (32,16), (64,32), (100,)],
        "learning_rate_init": [0.0005, 0.001, 0.005],
    }) if do_grid else [{"hidden_layer_sizes":(32,16),"learning_rate_init":0.001}]
    for params in grid:
        model = make_model(MLPRegressor(
            activation="relu", solver="adam", alpha=0.0001,
            max_iter=1500, early_stopping=False, random_state=SEED,
            **params))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtr, ytr)
        pred_val = np.clip(model.predict(Xva),0,90)
        score = mean_absolute_error(yva,pred_val)
        grid_results.append({"params":params,"validation_mae_deg":float(score)})
        if best is None or score < best[0]: best=(score,params)
    # Refit chosen architecture on train+validation, then evaluate once on test.
    trval = pd.concat([tr,va], ignore_index=True)
    p = best[1]
    mlp = make_model(MLPRegressor(activation="relu",solver="adam",alpha=0.0001,
        max_iter=1500,early_stopping=False,random_state=SEED,**p))
    mlp.fit(trval[feature_cols],trval[TARGET])
    results["models"]["mlp_validation_selected"] = metrics(yte,np.clip(mlp.predict(Xte),0,90))
    results["mlp_best_validation_params"] = p
    results["mlp_grid"] = grid_results
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Folder containing same_event_pairs.csv and same_event_pairs_within_5km.csv")
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    p100=pd.read_csv(out/"same_event_pairs.csv")
    p5=pd.read_csv(out/"same_event_pairs_within_5km.csv")
    # Filter pair table to the paper's magnitude criterion using pair-level mean magnitude.
    magcol = "mean_magnitude" if "mean_magnitude" in p100.columns else "mean_magnitude"
    p100=p100[pd.to_numeric(p100[magcol],errors="coerce")>=5].copy()
    p5=p5[pd.to_numeric(p5[magcol],errors="coerce")>=5].copy()
    # Normalize existing pipeline column names.
    rename={"station_distance_km":"station_distance_km", "angle_difference_deg":"angle_difference_deg"}
    p100=p100.rename(columns=rename); p5=p5.rename(columns=rename)
    feature_cols=[c for c in ["station_distance_km","mean_magnitude","absdiff_magnitude","mean_rjb_km","absdiff_rjb_km","mean_vs30","absdiff_vs30","mean_pga","absdiff_pga","mean_pgv","absdiff_pgv"] if c in p100.columns]
    primary={"protocol":"paper target subset: magnitude >= 5 and site separation <= 5 km",
             "n_pairs":int(len(p5)),"n_events":int(p5["event"].nunique()) if "event" in p5 else None,
             "status":"not_estimable_reliably" if len(p5)<30 else "model_evaluation_attempted",
             "reason":"Only 8 pairs are available in the current 5 km subset; an 80/10/10 split would yield about 6 training, 1 validation and 1 test pair. No reliable model comparison is possible."}
    if len(p5)>=10:
        primary["evaluation"]=run_models(p5,feature_cols)
    # Supplementary analysis: existing 100 km data, explicitly marked as not paper protocol.
    supplementary={"protocol":"supplementary only: magnitude >= 5 and site separation <= 100 km; differs from paper's 5 km limit",
                   "n_pairs":int(len(p100)),"n_events":int(p100["event"].nunique()) if "event" in p100 else None,
                   "evaluation":run_models(p100,feature_cols) if len(p100)>=10 else {"status":"insufficient_examples"}}
    report={"seed":SEED,"paper_reference_metrics":{"distance_benchmark":{"r2":0.025,"mae_deg":20.2},"ridge":{"r2":0.245,"mae_deg":16.9},"neural_network":{"r2":0.354,"mae_deg":13.2},"note":"Reference values transcribed from the supplied paper; do not treat as results from this dataset."},
            "primary_paper_aligned_experiment":primary,"supplementary_experiment":supplementary,
            "limitations":["Current waveform sample is a limited, non-random subset of NGA-West2.","The 5 km subset contains only 8 pairs from 5 events, insufficient for reliable training/testing.","Available pair features are not the paper's verified 21 selected features.","Random 80/10/10 split is used for supplementary comparison; pairs can share events/stations across splits.","Confirm orientation target and component azimuth convention against the paper before claiming exact reproduction."]}
    (out/"paper_aligned_results.json").write_text(json.dumps(report,indent=2,default=str))
    pd.DataFrame([{"experiment":"paper_aligned_5km","pairs":primary["n_pairs"],"events":primary["n_events"],"status":primary["status"]},
                  {"experiment":"supplementary_100km","pairs":supplementary["n_pairs"],"events":supplementary["n_events"],"status":supplementary["evaluation"].get("status","evaluated")}]).to_csv(out/"experiment_comparison.csv",index=False)
    print(json.dumps(report,indent=2,default=str))
    print("\nSaved paper_aligned_results.json and experiment_comparison.csv in",out)
if __name__=="__main__": main()
