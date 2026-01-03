import numpy as np

MIN_CHOL = 100.0     # sàn sinh lý hợp lý
MIN_BP   = 90.0

DEFAULT_LAMBDA = 0.2
DEFAULT_C_SCALE = 50.0
DEFAULT_BP_SCALE = 20.0

DEFAULT_C_BOUNDS = (-120.0, 0.0)
DEFAULT_BP_BOUNDS = (-60.0, 0.0)



def predict_heart_risk(patient_features: dict,
                       pipe,
                       feature_cols):
    """
    Trả về xác suất mắc bệnh tim (lớp dương) của bệnh nhân.
    """
    row = {c: patient_features.get(c, np.nan) for c in feature_cols}
    import pandas as pd
    X = pd.DataFrame([row])
    proba = pipe.predict_proba(X)[0, 1]
    return float(proba)


def intervention_objective(
    deltas,
    base_patient: dict,
    pipe,
    feature_cols,
    lam: float = DEFAULT_LAMBDA,
    C_scale: float = DEFAULT_C_SCALE,
    BP_scale: float = DEFAULT_BP_SCALE
):
    """
    Hàm mục tiêu tối ưu hóa:
        J = P_after + λ * Cost(ΔC, ΔBP)
    Trong đó:
        Cost = |ΔC|/C_scale + |ΔBP|/BP_scale
    """

    dC, dBP = deltas

    C0  = base_patient["Cholesterol"]
    BP0 = base_patient["RestingBP"]

    C_new  = C0 + dC
    BP_new = BP0 + dBP

    # chặn nghiệm phi sinh lý
    if C_new < MIN_CHOL or BP_new < MIN_BP:
        return 1e6

    patient_new = dict(base_patient)
    patient_new["Cholesterol"] = C_new
    patient_new["RestingBP"]   = BP_new

    p_after = predict_heart_risk(
        patient_new,
        pipe=pipe,
        feature_cols=feature_cols
    )

    cost = (abs(dC) / C_scale) + (abs(dBP) / BP_scale)

    return float(p_after + lam * cost)


def optimize_intervention_for_patient(
    base_patient: dict,
    pipe,
    feature_cols,
    lam: float = DEFAULT_LAMBDA,
    C_bounds = DEFAULT_C_BOUNDS,
    BP_bounds = DEFAULT_BP_BOUNDS,
    C_scale: float = DEFAULT_C_SCALE,
    BP_scale: float = DEFAULT_BP_SCALE,
):
    """
    Tối ưu hóa ΔCholesterol & ΔRestingBP bằng SLSQP.

    Trả về:
        - kết quả tối ưu hóa
        - nguy cơ trước / sau can thiệp
        - giá trị biến can thiệp tối ưu
    """

    from scipy.optimize import minimize

    x0 = np.array([0.0, 0.0], dtype=float)
    bounds = [C_bounds, BP_bounds]

    result = minimize(
        fun=intervention_objective,
        x0=x0,
        args=(base_patient, pipe, feature_cols, lam, C_scale, BP_scale),
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": 300}
    )

    dC_opt, dBP_opt = result.x

    C0  = base_patient["Cholesterol"]
    BP0 = base_patient["RestingBP"]

    patient_after = dict(base_patient)
    patient_after["Cholesterol"] = C0 + dC_opt
    patient_after["RestingBP"]   = BP0 + dBP_opt

    p_before = predict_heart_risk(base_patient, pipe, feature_cols)
    p_after  = predict_heart_risk(patient_after, pipe, feature_cols)

    # fallback hợp lý: nếu không cải thiện thì ghi nhận
    no_gain = (p_after >= p_before) or (abs(dC_opt) < 1e-6 and abs(dBP_opt) < 1e-6)

    return {
        "success": bool(result.success),
        "message": result.message,
        "lambda": lam,

        "dC_opt": float(dC_opt),
        "dBP_opt": float(dBP_opt),

        "Cholesterol_before": float(C0),
        "RestingBP_before": float(BP0),

        "Cholesterol_after": float(patient_after["Cholesterol"]),
        "RestingBP_after": float(patient_after["RestingBP"]),

        "p_before": float(p_before),
        "p_after": float(p_after),
        "delta_p": float(p_before - p_after),

        "fun_opt": float(result.fun),
        "no_effect": bool(no_gain)
    }
