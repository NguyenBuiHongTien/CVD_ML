# project/app/app.py
# Streamlit demo: DS1 Screening + Treatment Optimization (RestingBP, Oldpeak)
# Optional: DS2 Prognosis + Risk Score Bridge (if artifacts exist)

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import importlib.util


# =========================
# PATHS (MATCH YOUR STRUCTURE)
# =========================
APP_DIR = Path(__file__).resolve().parent           
ROOT = APP_DIR.parent                                
DATA_DIR = ROOT / "Data"
MODEL_DIR = DATA_DIR / "models"                      
CORE_FILE = ROOT / "Core" / "intervention_core_optimized.py"  

# DS1
DS1_MODEL_PATH = MODEL_DIR / "best_ds1_RandomForest.pkl"

# Optional DS2
DS2_MODEL_PATH = MODEL_DIR / "best_ds2_RandomForest.pkl"

# Optional Risk Score Bridge artifacts (if you want to show it)
RISK_BRIDGE_MODEL_PATH = MODEL_DIR / "risk_score_bridge_lr_ds1.pkl"
RISK_BRIDGE_COEF_CSV = MODEL_DIR / "risk_score_bridge_coef_or.csv"

# Optional datasets for auto-infer categorical options (if exists)
# You can adjust these if your DS1 file name differs.
DS1_DATA_CANDIDATES = [
    DATA_DIR / "heart.csv",
    DATA_DIR / "heartcsv",  # if you used that name
    DATA_DIR / "heart_failure_clinical_records_dataset.csv",  # not DS1 but keep as fallback
]

st.set_page_config(page_title="Cardiovascular Risk Demo", layout="wide")


# =========================
# LOAD CORE MODULE BY FILE (no need Core as a package)
# =========================
@st.cache_resource
def load_core(core_path: Path):
    if not core_path.exists():
        raise FileNotFoundError(f"Missing core file: {core_path}")

    spec = importlib.util.spec_from_file_location("intervention_core_optimixed", str(core_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    required = ["optimize_intervention_multi", "predict_heart_risk", "INTERVENTION_VARS_DS1", "DEFAULT_LAMBDA"]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise ImportError(f"Core file is missing required symbols: {missing}")

    return module

from sklearn.base import BaseEstimator, TransformerMixin

class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers using IQR rule:
      lower = Q1 - k*IQR
      upper = Q3 + k*IQR
    Works with pandas DataFrame.
    """
    def __init__(self, columns=None, k=1.5):
        self.columns = columns
        self.k = k

    def fit(self, X, y=None):
        import pandas as pd
        Xdf = X.copy()
        if not isinstance(Xdf, pd.DataFrame):
            Xdf = pd.DataFrame(Xdf)

        # If columns not specified, clip numeric columns
        if self.columns is None:
            cols = Xdf.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = list(self.columns)

        self.columns_ = cols
        self.bounds_ = {}

        for c in self.columns_:
            s = pd.to_numeric(Xdf[c], errors="coerce")
            q1 = np.nanquantile(s, 0.25)
            q3 = np.nanquantile(s, 0.75)
            iqr = q3 - q1
            lo = q1 - self.k * iqr
            hi = q3 + self.k * iqr
            self.bounds_[c] = (float(lo), float(hi))

        return self

    def transform(self, X):
        import pandas as pd
        Xdf = X.copy()
        if not isinstance(Xdf, pd.DataFrame):
            Xdf = pd.DataFrame(Xdf)

        # If fit chưa chạy (trường hợp hiếm), cố gắng không làm gì
        cols = getattr(self, "columns_", None)
        bounds = getattr(self, "bounds_", None)
        if not cols or not bounds:
            return Xdf

        for c in cols:
            if c in Xdf.columns:
                lo, hi = bounds[c]
                Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce").clip(lo, hi)

        return Xdf


@st.cache_resource
def load_joblib(path: Path):
    return joblib.load(path)


@st.cache_resource
def try_load_ds1_dataframe() -> pd.DataFrame | None:
    """
    Best-effort: load DS1 dataset for categorical option inference.
    If not found or fails, return None.
    """
    for p in DS1_DATA_CANDIDATES:
        if p.exists():
            try:
                # Allow common separators
                df = pd.read_csv(p)
                if df.shape[1] >= 5:
                    return df
            except Exception:
                pass
    return None


# =========================
# HELPERS
# =========================
def ensure_required_files():
    missing = []
    if not DS1_MODEL_PATH.exists():
        missing.append(str(DS1_MODEL_PATH))
    if not CORE_FILE.exists():
        missing.append(str(CORE_FILE))
    return missing


def infer_feature_cols_from_model(model) -> list[str]:
    """
    Prefer sklearn's feature_names_in_. Otherwise fallback to DS1 standard 11 features.
    """
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        if cols:
            return cols

    # DS1 standard (Heart Disease dataset) 11 columns
    return [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
        "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
    ]


def make_schema_from_data_or_default(df_ds1: pd.DataFrame | None, use_numeric_cats: bool) -> dict:
    """
    If df_ds1 is available, infer categorical unique values.
    Otherwise use default known categories (strings) or numeric codes if requested.
    """
    # Default label schema (typical Heart Disease dataset)
    default_label_schema = {
        "Sex": ["M", "F"],
        "ChestPainType": ["ASY", "ATA", "NAP", "TA"],
        "RestingECG": ["LVH", "Normal", "ST"],
        "ExerciseAngina": ["N", "Y"],
        "ST_Slope": ["Down", "Flat", "Up"],
        "FastingBS": [0, 1],
    }

    # Default numeric schema (if someone encoded categories)
    default_numeric_schema = {
        "Sex": [0, 1],             # depends on encoding in your training
        "ChestPainType": [0, 1, 2, 3],
        "RestingECG": [0, 1, 2],
        "ExerciseAngina": [0, 1],
        "ST_Slope": [0, 1, 2],
        "FastingBS": [0, 1],
    }

    if df_ds1 is not None:
        cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"]
        schema = {}
        for c in cat_cols:
            if c in df_ds1.columns:
                vals = df_ds1[c].dropna().unique().tolist()
                # Keep stable ordering
                try:
                    vals = sorted(vals)
                except Exception:
                    vals = list(vals)
                schema[c] = vals

        # If we inferred something meaningful, use it.
        if schema:
            # Ensure FastingBS is present
            if "FastingBS" not in schema:
                schema["FastingBS"] = [0, 1]
            return {"categorical": schema}

    # No DS1 df available -> fallback
    return {"categorical": (default_numeric_schema if use_numeric_cats else default_label_schema)}


def risk_group_from_p(p: float) -> str:
    # You can adjust thresholds to match your report
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Medium"
    return "High"


def build_patient_input(feature_cols: list[str], schema: dict) -> dict:
    """
    Build a patient dict according to feature_cols.
    Uses dropdowns for categorical columns (from schema) and numeric inputs for numeric ones.
    """
    cat_map = schema.get("categorical", {})

    c1, c2, c3 = st.columns(3)
    cols_ui = [c1, c2, c3]
    patient: dict = {}

    def num_int(name: str, min_v: int, max_v: int, default: int, step: int = 1) -> int:
        return int(st.number_input(name, min_value=min_v, max_value=max_v, value=default, step=step))

    def num_float(name: str, min_v: float, max_v: float, default: float, step: float = 0.1) -> float:
        return float(st.number_input(name, min_value=min_v, max_value=max_v, value=default, step=step))

    for i, col in enumerate(feature_cols):
        with cols_ui[i % 3]:
            if col in cat_map:
                opts = cat_map[col]
                if not opts:
                    patient[col] = st.text_input(col, value="")
                else:
                    patient[col] = st.selectbox(col, options=opts, index=0)
            else:
                # Common DS1 numeric defaults
                if col == "Age":
                    patient[col] = num_int(col, 1, 120, 50)
                elif col == "RestingBP":
                    patient[col] = num_int(col, 50, 250, 130)
                elif col == "Cholesterol":
                    patient[col] = num_int(col, 0, 800, 200)
                elif col == "MaxHR":
                    patient[col] = num_int(col, 40, 250, 150)
                elif col == "Oldpeak":
                    patient[col] = num_float(col, -5.0, 10.0, 1.0, 0.1)
                else:
                    # Generic numeric fallback
                    patient[col] = float(st.number_input(col, value=0.0))

    return patient


def changes_dataframe(res: dict) -> pd.DataFrame:
    df = pd.DataFrame(res.get("changes", []))
    if not df.empty and "delta" in df.columns:
        df["abs_delta"] = df["delta"].abs()
    return df


# =========================
# APP HEADER + FILE CHECKS
# =========================
st.title("Cardiovascular Risk – Screening & Treatment Optimization (Demo)")

missing_files = ensure_required_files()
if missing_files:
    st.error("Thiếu file bắt buộc. Kiểm tra lại các đường dẫn sau:")
    for m in missing_files:
        st.write("- " + m)
    st.stop()

# Load core
core = load_core(CORE_FILE)
optimize_intervention_multi = core.optimize_intervention_multi
predict_heart_risk = core.predict_heart_risk
INTERVENTION_VARS_DS1 = core.INTERVENTION_VARS_DS1
DEFAULT_LAMBDA = float(core.DEFAULT_LAMBDA)

# Load DS1 model
ds1_pipe = load_joblib(DS1_MODEL_PATH)
ds1_feature_cols = infer_feature_cols_from_model(ds1_pipe)

# Optional: load DS1 dataset for schema inference
df_ds1 = try_load_ds1_dataframe()

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Controls")

    st.write("Artifacts:")
    st.write("- DS1 model:", DS1_MODEL_PATH.name)
    st.write("- Core:", CORE_FILE.name)

    use_numeric_cats = st.checkbox(
        "Categorical inputs are numeric codes (0/1/2/3)",
        value=False,
        help="Bật nếu model của bạn train trên dữ liệu đã encode số thay vì nhãn chữ."
    )

    lam = st.number_input("Lambda (penalty)", min_value=0.0, max_value=1.0, value=float(DEFAULT_LAMBDA), step=0.005)
    n_random = st.slider("Random candidates", min_value=0, max_value=256, value=64, step=16)
    n_best = st.slider("Multi-start (top-k)", min_value=1, max_value=10, value=4, step=1)

    st.write("Intervention vars:", [v["name"] for v in INTERVENTION_VARS_DS1])

    st.caption("Khuyến nghị: thử lambda 0.02 / 0.01 / 0.005 nếu muốn thấy can thiệp rõ hơn.")


ds1_schema = make_schema_from_data_or_default(df_ds1, use_numeric_cats)


# =========================
# DS1 SCREENING + OPTIMIZE UI
# =========================
st.subheader("Module 1 (DS1) – Screening Risk + Optimization")
st.info("Lưu ý: Kết quả tối ưu là mô phỏng what-if theo mô hình ML, không phải khuyến cáo điều trị lâm sàng.")

patient = build_patient_input(ds1_feature_cols, ds1_schema)

b1, b2, _ = st.columns([1, 1, 2])
do_predict = b1.button("Predict (p_before)", use_container_width=True)
do_opt = b2.button("Optimize (p_after)", use_container_width=True)

if do_predict or do_opt:
    try:
        p_before = float(predict_heart_risk(patient, ds1_pipe, ds1_feature_cols))
        st.session_state["p_before"] = p_before
        st.session_state["patient"] = patient
    except Exception as e:
        st.error("Predict failed. Lý do thường gặp: sai kiểu dữ liệu categorical (nhãn chữ vs mã số).")
        st.exception(e)
        st.stop()

if "p_before" in st.session_state:
    p_before = float(st.session_state["p_before"])
    st.metric("p_before (P(HeartDisease))", f"{p_before:.6f}")
    st.write("Risk group:", risk_group_from_p(p_before))

if do_opt:
    base_patient = dict(st.session_state.get("patient", patient))
    try:
        res = optimize_intervention_multi(
            base_patient=base_patient,
            pipe=ds1_pipe,
            feature_cols=ds1_feature_cols,
            var_specs=INTERVENTION_VARS_DS1,
            lam=float(lam),
            method="auto",
            n_random=int(n_random),
            n_best_starts=int(n_best),
            random_state=42
        )
        st.session_state["opt_res"] = res
    except Exception as e:
        st.error("Optimization failed.")
        st.exception(e)

if "opt_res" in st.session_state:
    res = st.session_state["opt_res"]
    p_before = float(res["p_before"])
    p_after = float(res["p_after"])
    delta_p = float(res["delta_p"])

    c1, c2, c3 = st.columns(3)
    c1.metric("p_before", f"{p_before:.6f}")
    c2.metric("p_after", f"{p_after:.6f}")
    c3.metric("delta_p (before - after)", f"{delta_p:.6f}")

    st.write("Optimizer status:", "SUCCESS" if bool(res["success"]) else "FAIL")
    st.write("no_effect:", bool(res["no_effect"]))
    st.write("Message:", res.get("message", ""))

    df_changes = changes_dataframe(res)
    st.subheader("Suggested changes (simulation)")
    if df_changes.empty:
        st.warning("No changes returned.")
    else:
        st.dataframe(df_changes, use_container_width=True)
        st.bar_chart(df_changes.set_index("name")["delta"])

        csv = df_changes.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download changes CSV",
            data=csv,
            file_name="intervention_changes.csv",
            mime="text/csv"
        )


# =========================
# OPTIONAL: DS2 Prognosis
# =========================
st.divider()
st.subheader("Module 2 (DS2) – Prognosis (optional)")

if DS2_MODEL_PATH.exists():
    ds2_pipe = load_joblib(DS2_MODEL_PATH)

    # Best effort: infer DS2 feature columns
    if hasattr(ds2_pipe, "feature_names_in_"):
        ds2_cols = list(ds2_pipe.feature_names_in_)
    else:
        ds2_cols = []  # You can hardcode if needed

    if not ds2_cols:
        st.caption("DS2 model found, but cannot infer feature columns (feature_names_in_ missing).")
        st.caption("Nếu bạn muốn demo DS2, hãy train/serialize pipeline sklearn có feature_names_in_.")
    else:
        st.caption(f"DS2 model detected: {DS2_MODEL_PATH.name}")
        st.write("Provide DS2 inputs:")

        ds2_patient = {}
        cc = st.columns(3)
        for i, col in enumerate(ds2_cols):
            with cc[i % 3]:
                ds2_patient[col] = float(st.number_input(f"DS2::{col}", value=0.0))

        if st.button("Predict mortality risk (DS2)", use_container_width=True):
            X = pd.DataFrame([{c: ds2_patient.get(c, np.nan) for c in ds2_cols}])
            try:
                p_death = float(ds2_pipe.predict_proba(X)[0, 1])
                st.metric("P(DEATH_EVENT)", f"{p_death:.6f}")
            except Exception as e:
                st.error("DS2 prediction failed.")
                st.exception(e)
else:
    st.caption("Không thấy DS2 model trong Data/models/. Bỏ qua demo DS2.")


# =========================
# OPTIONAL: Risk Score Bridge (read-only display)
# =========================
st.divider()
st.subheader("Risk Score Bridge (optional)")

if RISK_BRIDGE_MODEL_PATH.exists():
    st.caption(f"Risk bridge model detected: {RISK_BRIDGE_MODEL_PATH.name}")

    show_coef = st.checkbox("Show coefficients / OR table (if available)", value=False)
    if show_coef and RISK_BRIDGE_COEF_CSV.exists():
        try:
            df_coef = pd.read_csv(RISK_BRIDGE_COEF_CSV)
            st.dataframe(df_coef, use_container_width=True)
        except Exception as e:
            st.error("Cannot read coefficient CSV.")
            st.exception(e)

    st.caption("Gợi ý: nếu muốn tính risk score trực tiếp trong app, cần xác định input schema (các biến) của model bridge.")
else:
    st.caption("Không thấy risk_score_bridge_lr_ds1.pkl trong Data/models/.")
