import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple

# =============================
# CẤU HÌNH MẶC ĐỊNH
# =============================

# Khuyến nghị: lambda nhỏ hơn để optimizer "dám" can thiệp trong giai đoạn nghiên cứu.
# Bạn có thể override khi gọi optimize_intervention_multi(lam=...)
DEFAULT_LAMBDA: float = 0.02

# Treat dataset-specific 0 values as missing for these columns
ZERO_AS_MISSING = {"Cholesterol", "RestingBP"}


# (Giữ lại cấu hình đầy đủ để tham chiếu)
INTERVENTION_VARS_DS1_ALL: List[Dict[str, Any]] = [
    {
        "name": "RestingBP",
        "delta_bounds": (-60.0, 0.0),
        "scale": 5.0,
        "min_value": 80.0,
        "max_value": 200.0,
    },
    {
        "name": "Cholesterol",
        "delta_bounds": (-150.0, 0.0),
        "scale": 10.0,
        "min_value": 90.0,
        "max_value": 600.0,
    },
    {
        "name": "Oldpeak",
        "delta_bounds": (-3.0, 0.0),
        "scale": 1.0,
        "min_value": 0.0,   # Oldpeak thường >= 0
        "max_value": None,
    },
]

# Mặc định tối ưu 2 biến có tính can thiệp lâm sàng rõ ràng: RestingBP + Cholesterol.
# (Oldpeak được giữ trong *_ALL để tham chiếu/mở rộng nếu cần.)
INTERVENTION_VARS_DS1: List[Dict[str, Any]] = [
    INTERVENTION_VARS_DS1_ALL[0],  # RestingBP
    INTERVENTION_VARS_DS1_ALL[1],  # Cholesterol
]


# =============================
# TIỆN ÍCH
# =============================

def _clamp(x: float, lo: Optional[float], hi: Optional[float]) -> float:
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x


def _apply_deltas(
    base_patient: Dict[str, Any],
    deltas: Sequence[float],
    var_specs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Áp delta lên base_patient và clamp theo delta_bounds + min/max_value.
    """
    p = dict(base_patient)
    for d, spec in zip(deltas, var_specs):
        name = spec["name"]
        dmin, dmax = spec["delta_bounds"]
        vmin = spec.get("min_value", None)
        vmax = spec.get("max_value", None)

        d = float(_clamp(float(d), float(dmin), float(dmax)))
        v0 = float(p[name])
        v_new = v0 + d
        v_new = float(_clamp(v_new, vmin, vmax))
        p[name] = v_new
    return p


# =============================
# HÀM DỰ ĐOÁN NGUY CƠ
# =============================

def predict_heart_risk(
    patient_features: Dict[str, Any],
    pipe,
    feature_cols: Sequence[str],
) -> float:
    """
    Trả về xác suất mắc bệnh (lớp dương) cho một bệnh nhân.
    """
    row = {c: patient_features.get(c, np.nan) for c in feature_cols}
    X = pd.DataFrame([row])
    # Normalize missing encoding: 0 -> NaN for selected columns
    for c in ZERO_AS_MISSING:
        if c in X.columns:
            X.loc[X[c] == 0, c] = np.nan
    proba = pipe.predict_proba(X)[0, 1]
    return float(proba)


# =============================
# HÀM MỤC TIÊU ĐA BIẾN
# =============================

def intervention_objective_multi(
    deltas: Sequence[float],
    base_patient: Dict[str, Any],
    pipe,
    feature_cols: Sequence[str],
    var_specs: Sequence[Dict[str, Any]],
    lam: float = DEFAULT_LAMBDA,
) -> float:
    """
    Hàm mục tiêu:
        J = P_after + lam * Σ_j |Δx_j| / scale_j
    """
    # Nếu có NaN ở biến can thiệp -> phạt lớn (không tối ưu trên điểm dữ liệu lỗi)
    for spec in var_specs:
        name = spec["name"]
        v = base_patient.get(name)
        if name not in base_patient or pd.isna(v) or (name in ZERO_AS_MISSING and float(v)==0.0):
            return 1e6

    # Áp delta có clamp để tránh drift; nếu clamp làm thay đổi nhiều, penalty vẫn tính theo delta gốc
    patient_new = _apply_deltas(base_patient, deltas, var_specs)

    # xác suất sau can thiệp
    p_after = predict_heart_risk(patient_new, pipe, feature_cols)

    # chi phí can thiệp
    total_cost = 0.0
    for d, spec in zip(deltas, var_specs):
        scale = float(spec["scale"])
        total_cost += abs(float(d)) / scale

    J = float(p_after) + float(lam) * float(total_cost)
    return float(J)


# =============================
# HÀM TỐI ƯU ĐA BIẾN (ROBUST)
# =============================

def optimize_intervention_multi(
    base_patient: Dict[str, Any],
    pipe,
    feature_cols: Sequence[str],
    var_specs: Optional[Sequence[Dict[str, Any]]] = None,
    lam: float = DEFAULT_LAMBDA,
    method: str = "auto",
    maxiter: int = 300,
    n_random: int = 64,
    n_best_starts: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tối ưu đồng thời nhiều biến can thiệp.

    method:
      - "auto": random-search + multi-start SLSQP (khuyến nghị)
      - "slsqp": chỉ SLSQP với x0 = 0 (nhanh nhưng hay kẹt với model cây)
      - "random": chỉ random-search (rất robust, nhưng kém mượt)

    Trả về dict:
      - p_before, p_after, delta_p
      - changes: list {name, before, after, delta}
      - success, no_effect, fun_opt
    """
    from scipy.optimize import minimize

    if var_specs is None:
        var_specs = INTERVENTION_VARS_DS1

    var_specs = list(var_specs)

    
    n_vars = len(var_specs)
    bounds: List[Tuple[float, float]] = [tuple(map(float, spec["delta_bounds"])) for spec in var_specs]

    # Các điểm khởi tạo "deterministic"
    x0_zero = np.zeros(n_vars, dtype=float)
    x0_mid  = np.array([0.5 * b[0] for b in bounds], dtype=float)  # nửa đường về dmin
    x0_dmin = np.array([b[0] for b in bounds], dtype=float)

    rng = np.random.default_rng(int(random_state))

    def _rand_point() -> np.ndarray:
        xs = []
        for (lo, hi) in bounds:
            xs.append(rng.uniform(lo, hi))
        return np.array(xs, dtype=float)

    # Tập candidates cho auto/random
    candidates: List[np.ndarray] = [x0_zero, x0_mid, x0_dmin]
    if n_random and n_random > 0:
        candidates.extend([_rand_point() for _ in range(int(n_random))])

    # Evaluate candidates (random search)
    cand_scores = []
    for x in candidates:
        J = intervention_objective_multi(x, base_patient, pipe, feature_cols, var_specs, lam)
        cand_scores.append(float(J))

    order = np.argsort(cand_scores)
    best_idx = order[: max(1, int(n_best_starts))]
    best_starts = [candidates[i] for i in best_idx]

    # Nếu chỉ random
    if method.lower() == "random":
        x_best = best_starts[0]
        deltas_opt = x_best
        patient_after = _apply_deltas(base_patient, deltas_opt, var_specs)
        p_before = predict_heart_risk(base_patient, pipe, feature_cols)
        p_after = predict_heart_risk(patient_after, pipe, feature_cols)
        delta_p = float(p_before - p_after)
        changes = []
        for d, spec in zip(deltas_opt, var_specs):
            name = spec["name"]
            v0 = float(base_patient[name])
            v_new = float(patient_after[name])
            changes.append({"name": name, "before": v0, "after": v_new, "delta": float(v_new - v0)})
        no_effect = (delta_p <= 1e-9) or np.all(np.abs(np.array([c["delta"] for c in changes])) < 1e-6)
        return {
            "success": True,
            "message": "random-search only",
            "lambda": float(lam),
            "p_before": float(p_before),
            "p_after": float(p_after),
            "delta_p": float(delta_p),
            "changes": changes,
            "fun_opt": float(cand_scores[int(best_idx[0])]),
            "no_effect": bool(no_effect),
            "raw_deltas": [float(d) for d in deltas_opt],
            "method_used": "random",
        }

    # Nếu chỉ SLSQP legacy
    if method.lower() == "slsqp":
        best_starts = [x0_zero]

    # Multi-start SLSQP: chạy từ vài điểm tốt nhất
    best_result = None
    for x0 in best_starts:
        res = minimize(
            fun=intervention_objective_multi,
            x0=np.array(x0, dtype=float),
            args=(base_patient, pipe, feature_cols, var_specs, lam),
            bounds=bounds,
            method="SLSQP",
            options={"maxiter": int(maxiter)},
        )
        if best_result is None or float(res.fun) < float(best_result.fun):
            best_result = res

    # Fallback: nếu SLSQP fail rõ rệt, thử Powell (gradient-free) với bounds
    # (Powell hỗ trợ bounds trong SciPy hiện đại)
    if best_result is not None and (not bool(best_result.success)):
        try:
            res2 = minimize(
                fun=intervention_objective_multi,
                x0=np.array(best_starts[0], dtype=float),
                args=(base_patient, pipe, feature_cols, var_specs, lam),
                bounds=bounds,
                method="Powell",
                options={"maxiter": int(maxiter)},
            )
            if float(res2.fun) < float(best_result.fun):
                best_result = res2
        except Exception:
            pass

    result = best_result
    deltas_opt = np.array(result.x, dtype=float)

    # Áp deltas với clamp/min/max để tạo phương án sau can thiệp
    patient_after = _apply_deltas(base_patient, deltas_opt, var_specs)

    p_before = predict_heart_risk(base_patient, pipe, feature_cols)
    p_after = predict_heart_risk(patient_after, pipe, feature_cols)
    delta_p = float(p_before - p_after)

    changes = []
    for spec in var_specs:
        name = spec["name"]
        v0 = float(base_patient[name])
        v_new = float(patient_after[name])
        changes.append({"name": name, "before": v0, "after": v_new, "delta": float(v_new - v0)})

    # no_effect: không giảm được nguy cơ hoặc hầu như không can thiệp
    no_effect = (delta_p <= 1e-9) or np.all(np.abs(np.array([c["delta"] for c in changes])) < 1e-6)

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "lambda": float(lam),
        "p_before": float(p_before),
        "p_after": float(p_after),
        "delta_p": float(delta_p),
        "changes": changes,
        "fun_opt": float(result.fun),
        "no_effect": bool(no_effect),
        "raw_deltas": [float(d) for d in deltas_opt],
        "method_used": ("slsqp" if method.lower() == "slsqp" else "auto"),
        "n_candidates": int(len(candidates)),
        "n_multistarts": int(len(best_starts)),
    }
