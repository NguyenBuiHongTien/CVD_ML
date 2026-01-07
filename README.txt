Clinical Decision Support & Treatment Optimization – DS1 (Heart Disease)

This project implements a clinical decision support workflow for heart
disease screening and treatment‑intervention optimization on dataset
DS1. The system includes: data preprocessing, model training and
evaluation, risk‑based threshold tuning, and an intervention
optimization engine with a Streamlit demo application.

1) Project Structure

-   CDV.ipynb — end‑to‑end notebook (EDA → preprocessing → CV → test
    evaluation → threshold tuning → artifact export).
-   Data/models/
    -   best_ds1_RandomForest.pkl — trained final model (train+val).
    -   best_ds1_RandomForest_threshold.joblib — tuned decision
        threshold metadata.
-   Core/
    -   intervention_core_optimized_bp_chol.py — optimization engine for
        clinical intervention on RestingBP & Cholesterol.
-   App/
    -   app_bp_chol.py — Streamlit demo app (screening + optimization).
-   Reports/
    -   522H0089_522H0042.docx — project report.

2) Dataset (DS1)

-   918 records, 11 clinical features: Age, Sex, ChestPainType,
    RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR,
    ExerciseAngina, Oldpeak, ST_Slope.
-   Values 0 for RestingBP / Cholesterol are treated as missing and
    imputed.
-   Outliers are clipped using IQR bounds saved inside the preprocessing
    pipeline.

3) Model Pipeline

-   Candidate models: Logistic Regression, RandomForest, XGBoost,
    CatBoost.
-   5‑fold stratified cross‑validation on train+val.
-   Selected model: RandomForest (best mean ROC‑AUC among candidates).
-   Exported artifacts:
    -   estimator pipeline,
    -   IQR‑clipper & imputers,
    -   label encoder / one‑hot encoders,
    -   tuned decision threshold metadata.

4) Evaluation Summary (DS1)

-   Metrics reported on held‑out test set:
    -   ROC‑AUC
    -   PR‑AUC
    -   Brier score
-   Threshold tuning:
    -   default = 0.5 (baseline screening)
    -   tuned threshold (from validation) prioritises recall for
        positive class.
-   Comparison tables in notebook include:
    -   baseline vs tuned threshold,
    -   precision / recall / specificity / F1 changes.

5) Intervention Optimization Engine

-   Implemented in intervention_core_optimized_bp_chol.py.

-   Decision objective:

    J = P_after + λ * Σ |Δx_j| / scale_j

    where:

    -   P_after — predicted post‑intervention probability,
    -   λ — intervention cost weight,
    -   Δx_j — change on variable j,
    -   scale_j — clinical scaling factor.

-   Default intervention variables:

    -   RestingBP (Δ in [−60, 0], clinical clamp 80–200)
    -   Cholesterol (Δ in [−150, 0], clinical clamp 90–600)

-   Search strategy:

    -   random scan → multi‑start SLSQP → Powell fallback.

6) Streamlit Demo App

Run locally:

    streamlit run app_bp_chol.py

Main functions:

1.  Patient input → compute p_before (risk probability).
2.  Run optimization → compute p_after and Δp.
3.  Display:
    -   optimizer status,
    -   intervention deltas,
    -   p_before vs p_after chart,
    -   CSV export of recommendations.

If DS2 or risk‑bridge artifacts are not present, the app safely skips
them and keeps DS1 demo operational.

7) Re‑training & Reproducibility

To regenerate artifacts:

1.  Open CDV.ipynb.
2.  Run all cells up to:
    -   model selection & CV,
    -   test evaluation,
    -   threshold tuning,
    -   artifact export.
3.  Ensure exported paths match those configured in the app.

8) Notes on Consistency

-   Report, notebook, and app should use the same exported model
    artifacts.
-   If new training is performed, re‑export the model & threshold and
    update the report numbers accordingly.

------------------------------------------------------------------------

Author(s): 522H0089 — 522H0042 Faculty of Information Technology Ho Chi
Minh City Open University
