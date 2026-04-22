"""Pipeline end-to-end: load → split → preprocess → WoE/IV → train → SHAP → leaderboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_raw, stratified_split
from src.features.preprocessing import encode_categoricals, prepare_target, save_processed
from src.features.stat_tests import anova_f_test, chi_square_test, compute_vif, drop_high_vif
from src.features.woe_iv import iv_table, select_by_iv
from src.evaluation.metrics import evaluate, leaderboard
from src.models.baselines import (
    compute_scale_pos_weight,
    make_catboost,
    make_logistic_regression,
    make_random_forest,
    make_xgboost,
)
from src.models.focal_lgbm import predict_proba, train_focal_lgbm
from src.models.tune_focal import tune as tune_focal
from src.explain.shap_utils import explain_tree, save_summary, save_waterfall


def main(config_path: str = "configs/config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = cfg["project"]["random_seed"]
    target = cfg["data"]["target_col"]

    print("=" * 70)
    print("[1/7] Load + split dataset")
    print("=" * 70)
    df = load_raw(Path(cfg["data"]["raw_dir"]))
    df = prepare_target(df, target, cfg["data"]["target_positive_is_default"])
    print(f"  rows={len(df)}  default_rate={df[target].mean():.4f}")

    train, val, test = stratified_split(
        df, target,
        cfg["split"]["train_size"], cfg["split"]["val_size"], seed,
    )
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train.columns if c != target and c not in cat_cols]

    print("\n" + "=" * 70)
    print("[2/7] Statistical tests (Chi-square, ANOVA)")
    print("=" * 70)
    chi2 = chi_square_test(train, cat_cols, target)
    anova = anova_f_test(train, num_cols, target)
    print("Chi-square (p < 0.05 significant):")
    print(chi2.to_string(index=False))
    print("\nANOVA F-test:")
    print(anova.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("[3/7] WoE/IV feature selection")
    print("=" * 70)
    iv_df = iv_table(train, cat_cols + num_cols, target)
    print(iv_df.to_string(index=False))
    iv_min = cfg["feature_engineering"]["iv_min"]
    selected = select_by_iv(iv_df, iv_min)
    print(f"\nSelected (IV >= {iv_min}): {len(selected)}/{len(iv_df)} features")

    print("\n" + "=" * 70)
    print("[4/7] Encode categoricals + VIF check")
    print("=" * 70)
    train, val, test, _ = encode_categoricals(train, val, test, cat_cols)

    selected_num = [c for c in selected if c in num_cols]
    if selected_num:
        vif = compute_vif(train, selected_num)
        print("VIF (diagnostic — not dropping, tree models tolerant to multicollinearity):")
        print(vif.head(10).to_string(index=False))
    features = selected

    print(f"\nFinal features: {len(features)}")
    save_processed(
        train[features + [target]], val[features + [target]], test[features + [target]],
        Path(cfg["data"]["processed_dir"]),
    )

    X_tr, y_tr = train[features], train[target].astype(int).values
    X_va, y_va = val[features], val[target].astype(int).values
    X_te, y_te = test[features], test[target].astype(int).values

    print("\n" + "=" * 70)
    print("[5/7] Train baselines")
    print("=" * 70)
    results: dict[str, dict] = {}

    print("  [LR]")
    lr = make_logistic_regression()
    lr.fit(X_tr, y_tr)
    results["LogisticRegression"] = evaluate(y_te, lr.predict_proba(X_te)[:, 1])

    print("  [RF]")
    rf = make_random_forest(random_state=seed)
    rf.fit(X_tr, y_tr)
    results["RandomForest"] = evaluate(y_te, rf.predict_proba(X_te)[:, 1])

    print("  [XGBoost]")
    spw = compute_scale_pos_weight(y_tr)
    xgb_m = make_xgboost(scale_pos_weight=spw, random_state=seed)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    results["XGBoost"] = evaluate(y_te, xgb_m.predict_proba(X_te)[:, 1])

    print("  [CatBoost]")
    cb = make_catboost(random_seed=seed)
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    results["CatBoost"] = evaluate(y_te, cb.predict_proba(X_te)[:, 1])

    print("\n" + "=" * 70)
    print("[6/7] Train Focal-LightGBM (baseline + Optuna-tuned)")
    print("=" * 70)
    fc = cfg["models"]["focal_lgbm"]
    print("  [baseline]")
    booster_base = train_focal_lgbm(
        X_tr, y_tr, X_va, y_va,
        params={
            "num_leaves": fc["num_leaves"],
            "learning_rate": fc["learning_rate"],
            "min_child_samples": fc["min_child_samples"],
            "reg_lambda": fc["reg_lambda"],
        },
        alpha=fc["alpha"], gamma=fc["gamma"],
        num_boost_round=fc["n_estimators"],
        early_stopping_rounds=fc["early_stopping_rounds"],
    )
    results["FocalLightGBM"] = evaluate(y_te, predict_proba(booster_base, X_te))

    print(f"\n  [Optuna tune — {cfg['optuna']['n_trials']} trials]")
    best = tune_focal(X_tr, y_tr, X_va, y_va, n_trials=cfg["optuna"]["n_trials"], seed=seed)
    print(f"  Best params: {best}")
    tuned_alpha = best.pop("alpha")
    tuned_gamma = best.pop("gamma")
    booster = train_focal_lgbm(
        X_tr, y_tr, X_va, y_va,
        params=best, alpha=tuned_alpha, gamma=tuned_gamma,
        num_boost_round=3000, early_stopping_rounds=150,
    )
    results["FocalLightGBM_Tuned"] = evaluate(y_te, predict_proba(booster, X_te))

    print("\n" + "=" * 70)
    print("[7/7] Leaderboard")
    print("=" * 70)
    lb = leaderboard(results)
    print(lb.round(4).to_string())

    reports = Path("reports")
    figures = reports / "figures"
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    lb.to_csv(reports / "leaderboard.csv")
    iv_df.to_csv(reports / "iv_table.csv", index=False)
    chi2.to_csv(reports / "chi2_test.csv", index=False)
    anova.to_csv(reports / "anova_test.csv", index=False)
    (reports / "selected_features.json").write_text(json.dumps(features, indent=2))
    (reports / "best_params.json").write_text(
        json.dumps({"params": best, "alpha": tuned_alpha, "gamma": tuned_gamma}, indent=2)
    )
    booster.save_model(str(reports / "focal_lgbm_tuned.txt"))

    print("\n" + "=" * 70)
    print("[8] SHAP Explainability")
    print("=" * 70)
    explainer, shap_values, X_sample = explain_tree(booster, X_te, sample=3000, seed=seed)
    save_summary(shap_values, X_sample, figures / "shap_summary_dot.png", plot_type="dot")
    save_summary(shap_values, X_sample, figures / "shap_summary_bar.png", plot_type="bar")
    for i, name in enumerate(["sample_0", "sample_1", "sample_2"]):
        save_waterfall(explainer, shap_values, X_sample, row=i, path=figures / f"shap_waterfall_{name}.png")
    print(f"  SHAP plots saved to {figures}/")

    print(f"\nAll artifacts in: {reports}/")
    return booster, X_te, y_te, features


if __name__ == "__main__":
    main()
