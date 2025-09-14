# -*- coding: utf-8 -*-
# Streamlit deployment of SWI Wedge Length Ratio – Smart Predictor
# Layout:
#   Left  = Inputs + large prediction readout
#   Right = Reference sketch (auto-fit, with upload fallback)
#   Bottom = Predict, Clear, Recall, Save Inputs (JSON), Load Inputs
# Tabs:
#   Predict, Explain (SHAP), Batch, History, Article Info
# Deterministic CPU predictions; cached model; no unhashable-arg caching.

import json
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Page / theme config
# ------------------------------
st.set_page_config(
    page_title="Machine Learning-Based Modeling of Artificial Recharge and Cutoff Walls for Seawater Intrusion Control in Coastal Aquifers",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Source+Serif+4:wght@500;700&display=swap');

      :root {
        --ui-bg: #ffffff;          /* main background */
        --ui-card: #ffffff;        /* cards */
        --ui-border: #cccccc;      /* borders */
        --ui-text: #000000;        /* main text */
        --ui-text-muted: #666666;  /* muted text */
        --ui-accent: #000000;      /* accent (black) */
      }

      .stApp { background: var(--ui-bg); color: var(--ui-text); }
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      body, .stApp, p, div, span, label, input, select, textarea {
        font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: var(--ui-text);
        font-size: 16px;
      }
      h1, h2, h3, h4, h5, h6 {
        font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
        color: var(--ui-text);
        font-weight: 700;
      }

      .card {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--ui-border);
        background: var(--ui-card);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      }
      .big-number {
        font-size: 44px;
        font-weight: 800;
        color: var(--ui-accent);
        margin: .2rem 0 .8rem;
      }

      /* Inputs */
      input, textarea, select {
        background: var(--ui-card) !important;
        color: var(--ui-text) !important;
        border: 1px solid var(--ui-border) !important;
        border-radius: 10px !important;
      }
      .stNumberInput input {
        background: var(--ui-card) !important;
        color: var(--ui-text) !important;
        border: 1px solid var(--ui-border) !important;
        border-radius: 10px !important;
      }
      .stSelectbox div[role="combobox"] {
        background: var(--ui-card);
        border: 1px solid var(--ui-border);
        border-radius: 10px;
        color: var(--ui-text);
      }

      /* Buttons */
      .stButton > button[kind="primary"] {
        background: #ffffff !important;  /* White background for buttons */
        color: #000000 !important;       /* Black text color */
        border: 1px solid var(--ui-border) !important; /* Light border */
        border-radius: 10px;             /* Rounded corners */
        padding: .6rem 1rem;
        font-weight: 800;
      }
      .stButton > button[kind="secondary"] {
        background: #f0f0f0 !important;
        color: var(--ui-text) !important;
        border: 1px solid var(--ui-border);
        border-radius: 10px;
        padding: .55rem 1rem;
        font-weight: 700;
      }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
      }
      .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        color: var(--ui-text);
        border: 1px solid var(--ui-border);
        border-radius: 10px;
        padding: .45rem 1rem;
        font-weight: 700;
      }
      .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: var(--ui-accent) !important;
        border-color: var(--ui-accent) !important;
      }

      /* Tables */
      .stDataFrame {
        background: var(--ui-card);
        border: 1px solid var(--ui-border);
        border-radius: 12px;
        padding: .25rem;
      }
      [data-testid="stDataFrame"] thead th {
        background: #f0f0f0 !important;
        color: var(--ui-text);
        font-weight: 800;
      }

      /* Uploader */
      [data-testid="stFileUploader"] section {
        border: 1px dashed var(--ui-border);
        background: #f9f9f9;
        border-radius: 12px;
        padding: .8rem;
        color: var(--ui-text);
      }

      .muted {
        color: var(--ui-text-muted);
        font-size: 0.95rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Config / constants (ADAPTED TO YOUR VARIABLES)
# ------------------------------
MODEL_PATH = "C:/Users/asus1/Desktop/CGB1.joblib"  # your CatBoost (or other) model file
IMAGE_CANDIDATES = [
    Path("assets/sketch.png"),
    Path("assets/sketch22.png"),
    Path("sketch.png"),
    Path("sketch22.png"),
    Path("C:/Users/asus1/Desktop/sketch.png"),
]

FEATURE_KEYS = ['X1','X2','X3','X4','X5','X6','X7','X8']  # must match model training order
LABELS = {
    'X1': "ρs/ρf   (Relative density)",
    'X2': "K/Ko    (Relative hydraulic conductivity)",
    'X3': "Qi/(Ko·Lo²)  (Relative recharge rate)",
    'X4': "i       (Hydraulic gradient)",
    'X5': "Xi/Lo   (Relative well distance)",
    'X6': "Yi/Lo   (Relative well depth)",
    'X7': "Xb/Lo   (Relative barrier wall distance)",
    'X8': "Db/Lo   (Relative barrier wall depth)",
}

# Bounds (sane defaults; adjust to your dataset if known)
FEATURE_RANGES = {
    'X1': (1.000, 1.100, 1.025),  # seawater/freshwater density ratio ~ 1.02–1.03 typically
    'X2': (0.10,  10.00, 1.00),
    'X3': (0.000000, 1.000000, 0.050000),
    'X4': (0.000, 0.500, 0.010),
    'X5': (0.000, 5.000, 1.000),
    'X6': (0.000, 1.000, 0.500),
    'X7': (0.000, 5.000, 1.000),
    'X8': (0.000, 1.000, 0.500),
}

# Per-feature numeric input resolution/format
SLIDER_SPEC = {
    'X1': dict(step=1e-4,  fmt="%.6f"),
    'X2': dict(step=1e-3,  fmt="%.3f"),
    'X3': dict(step=1e-6,  fmt="%.6f"),
    'X4': dict(step=1e-3,  fmt="%.3f"),
    'X5': dict(step=1e-2,  fmt="%.2f"),
    'X6': dict(step=1e-3,  fmt="%.3f"),
    'X7': dict(step=1e-2,  fmt="%.2f"),
    'X8': dict(step=1e-3,  fmt="%.3f"),
}

# Presets (feel free to adjust)
PRESETS = {
    "— choose a preset —": None,
    "Baseline":          [1.025, 1.00, 0.050000, 0.010, 1.00, 0.50, 1.00, 0.50],
    "High Recharge":     [1.025, 1.00, 0.200000, 0.010, 1.00, 0.50, 1.00, 0.50],
    "High Gradient":     [1.025, 1.00, 0.050000, 0.100, 1.00, 0.50, 1.00, 0.50],
    "Deep Barrier":      [1.025, 1.00, 0.050000, 0.010, 1.00, 0.50, 1.00, 0.90],
}

# ------------------------------
# Helpers
# ------------------------------
def _lock_model_deterministic(model):
    """Try to force single-threaded CPU prediction and fetch expected feature names."""
    expected = None
    # XGBoost style
    try:
        model.get_booster().set_param({"predictor": "cpu_predictor", "nthread": 1})
    except Exception:
        pass
    try:
        model.set_params(n_jobs=1, nthread=1, thread_count=1)
    except Exception:
        # CatBoost native has set_params too; if not, ignore.
        try:
            if hasattr(model, "set_params"):
                model.set_params(thread_count=1)
        except Exception:
            pass
    # Feature names detection (works for many sklearn-compatible models)
    for attr in ("feature_names_in_", "feature_names_", "feature_names"):
        if hasattr(model, attr):
            try:
                names = getattr(model, attr)
                expected = list(map(str, names))
                break
            except Exception:
                expected = None
    return expected

def _ordered_df(values: dict, expected_names):
    """Build 1-row DataFrame with correct column order."""
    if expected_names and set(map(str, expected_names)) == set(FEATURE_KEYS):
        cols = list(map(str, expected_names))
    else:
        cols = FEATURE_KEYS[:]
    row = [values[c] for c in cols]
    return pd.DataFrame([row], columns=cols).astype(np.float32)

def json_download_bytes(obj):
    buf = BytesIO()
    buf.write(json.dumps(obj, indent=2).encode("utf-8"))
    buf.seek(0)
    return buf

def sample_background_df(ranges: dict, n: int = 256, seed: int | None = None) -> pd.DataFrame:
    """Uniformly sample within slider ranges for background SHAP."""
    rng = np.random.default_rng(None if seed is None else int(seed))
    data = {}
    for k in FEATURE_KEYS:
        lo, hi, _ = ranges[k]
        data[k] = rng.uniform(lo, hi, size=n)
    return pd.DataFrame(data).astype(np.float32)

def find_local_image() -> Image.Image | None:
    for p in IMAGE_CANDIDATES:
        if Path(p).exists():
            try:
                return Image.open(p)
            except Exception:
                pass
    return None

def ranges_key_tuple() -> tuple:
    """Hashable key for caching background SHAP when ranges change."""
    return tuple((k, tuple(map(float, FEATURE_RANGES[k]))) for k in FEATURE_KEYS)

def clip_to_bounds(vals: list[float]) -> list[float]:
    out = []
    for v, k in zip(vals, FEATURE_KEYS):
        lo, hi, _ = FEATURE_RANGES[k]
        out.append(min(max(float(v), lo), hi))
    return out

# ------------------------------
# Caching (no unhashable params!)
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_expected():
    m = joblib.load(MODEL_PATH)
    expected = _lock_model_deterministic(m)
    return m, expected

@st.cache_resource(show_spinner=False)
def get_explainer():
    model, _ = load_model_and_expected()
    # Generic Explainer auto-selects TreeExplainer if supported (CatBoost/XGB/LightGBM)
    return shap.Explainer(model)

@st.cache_data(show_spinner=False)
def shap_background_values_uniform(n: int, rk: tuple, seed: int | None):
    """Cached global SHAP on synthetic (uniform) background."""
    model, expected = load_model_and_expected()
    explainer = get_explainer()
    df_bg = sample_background_df(FEATURE_RANGES, n, seed)
    # Align to model's column order
    X_bg = _ordered_df({k: 0.0 for k in FEATURE_KEYS}, expected)
    X_bg = df_bg[X_bg.columns]
    sv = explainer(X_bg)
    return sv, X_bg

@st.cache_data(show_spinner=False)
def shap_background_values_dataset(file_bytes: bytes, n: int, seed: int | None):
    """Cached global SHAP on dataset background (uploaded CSV)."""
    model, expected = load_model_and_expected()
    explainer = get_explainer()
    df = pd.read_csv(BytesIO(file_bytes))
    if not set(FEATURE_KEYS).issubset(df.columns):
        missing = [c for c in FEATURE_KEYS if c not in df.columns]
        raise ValueError(f"Dataset missing columns: {missing}")
    ordered_cols = _ordered_df({k: 0.0 for k in FEATURE_KEYS}, expected).columns
    if len(df) > n:
        df = df.sample(n=n, random_state=None if seed is None else int(seed))
    X_bg = df[ordered_cols].astype(np.float32)
    sv = explainer(X_bg)
    return sv, X_bg

def predict_one(values_dict):
    model, expected = load_model_and_expected()
    try:
        model.get_booster().set_param({"predictor": "cpu_predictor", "nthread": 1})
    except Exception:
        pass
    try:
        model.set_params(n_jobs=1, nthread=1, thread_count=1)
    except Exception:
        pass
    X = _ordered_df(values_dict, expected)
    y = model.predict(X)
    # Many libs return shape (1,) or [[y]]
    return float(np.ravel(y)[0])

# ------------------------------
# Session state
# ------------------------------
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, each with time + Xs + pred
if "current_pred" not in st.session_state:
    st.session_state.current_pred = None
if "current_inputs" not in st.session_state:
    st.session_state.current_inputs = {k: FEATURE_RANGES[k][2] for k in FEATURE_KEYS}
if "sketch_bytes" not in st.session_state:
    st.session_state.sketch_bytes = None
if "bg_file_bytes" not in st.session_state:
    st.session_state.bg_file_bytes = None

# ------------------------------
# Header
# ------------------------------
st.title("Machine Learning-Based Modeling of Artificial Recharge and Cutoff Walls for Seawater Intrusion Control in Coastal Aquifers")
st.caption("For users, technicians, water resources engineers, and hydrogeologists – quick, reliable, and explainable.")

# Tabs
tab_predict, tab_explain, tab_batch, tab_hist, tab_article = st.tabs(
    ["Predict", "Explain", "Batch", "History", "Article Info"]
)

# ==============================
# PREDICT TAB
# ==============================
with tab_predict:
    col_left, col_right = st.columns([3, 2], gap="large")

    # LEFT: prediction + inputs
    with col_left:
        st.markdown("#### Prediction")
        big = "—" if st.session_state.current_pred is None else f"{st.session_state.current_pred:.6f}"
        st.markdown(
            f"<div class='card'><div class='big-number'>{big}</div>"
            f"<div>Predicted Relative SWI wedge length (L/Lo)</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Input Parameters (Dimensionless)")
        # Preset
        preset = st.selectbox("Preset", list(PRESETS.keys()), index=0)
        if PRESETS.get(preset):
            vals = clip_to_bounds(PRESETS[preset])
            st.session_state.current_inputs = {k: float(v) for k, v in zip(FEATURE_KEYS, vals)}

        # Number inputs (4 per row)
        ordered_keys = FEATURE_KEYS[:]
        for i in range(0, len(ordered_keys), 4):
            cols = st.columns(4)
            for j, k in enumerate(ordered_keys[i:i+4]):
                lo, hi, df = FEATURE_RANGES[k]
                spec = SLIDER_SPEC[k]
                default_val = float(st.session_state.current_inputs.get(k, df))
                default_val = float(min(max(default_val, float(lo)), float(hi)))
                with cols[j]:
                    st.session_state.current_inputs[k] = st.number_input(
                        LABELS[k],
                        min_value=float(lo),
                        max_value=float(hi),
                        value=default_val,
                        step=float(spec["step"]),
                        format=spec["fmt"],
                    )

        # Bottom row buttons
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
        with c1:
            if st.button("Predict", use_container_width=True, type="primary"):
                try:
                    values = {k: float(st.session_state.current_inputs[k]) for k in FEATURE_RANGES.keys()}
                    y = predict_one(values)
                    st.session_state.current_pred = y
                    st.session_state.last_inputs = [values[k] for k in FEATURE_KEYS]
                    rec = {"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **values, "Prediction": round(float(y), 6)}
                    st.session_state.history.append(rec)
                    st.success("Prediction complete.")
                except FileNotFoundError:
                    st.error(f"Model not found at `{MODEL_PATH}`. Adjust MODEL_PATH.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.current_pred = None
                st.session_state.current_inputs = {k: FEATURE_RANGES[k][2] for k in FEATURE_KEYS}
                st.info("Cleared.")
        with c3:
            disabled = st.session_state.last_inputs is None
            if st.button("Recall Last", use_container_width=True, disabled=disabled):
                if st.session_state.last_inputs is not None:
                    for k, v in zip(FEATURE_KEYS, st.session_state.last_inputs):
                        st.session_state.current_inputs[k] = float(v)
                    st.success("Recalled last inputs.")
        with c4:
            buf = json_download_bytes(st.session_state.current_inputs)
            st.download_button("Save Inputs (JSON)", data=buf, file_name="inputs.json",
                               mime="application/json", use_container_width=True)
        with c5:
            up = st.file_uploader("Load Inputs", type=["json"], label_visibility="collapsed", key="upl_json_predict")
            if up is not None:
                try:
                    data = json.loads(up.read().decode("utf-8"))
                    for k in FEATURE_KEYS:
                        if k in data:
                            st.session_state.current_inputs[k] = float(data[k])
                    st.success("Inputs loaded.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

    # RIGHT: reference sketch (with upload fallback)
    with col_right:
        st.markdown("#### Reference Sketch")
        up = st.file_uploader("Upload sketch (PNG/JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if up is not None:
            st.session_state.sketch_bytes = up.read()

        img = None
        if st.session_state.sketch_bytes is not None:
            try:
                img = Image.open(BytesIO(st.session_state.sketch_bytes))
            except Exception:
                st.warning("Uploaded file isn't a valid image. Falling back to local file…")

        if img is None:
            img = find_local_image()

        if img is None:
            st.info("No image found. Add one at `assets/sketch.png` in the repo or upload above.")
        else:
            st.image(img, use_container_width=True)

# ==============================
# EXPLAIN TAB (SHAP)
# ==============================
with tab_explain:
    st.markdown("### Explain (SHAP)")

    bg_src = st.radio("Background source for global SHAP:",
                      ["Uniform (use slider bounds)", "Dataset (upload CSV)"],
                      horizontal=True)
    n_bg = st.slider("Background sample size", 100, 2000, 256, 50,
                     help="Larger = smoother but slower. 256–512 is good for 8 features.")
    seed = st.number_input("Random seed (optional)", min_value=0, max_value=10_000, value=42, step=1)

    try:
        if bg_src.startswith("Uniform"):
            sv_bg, X_bg = shap_background_values_uniform(n=n_bg, rk=ranges_key_tuple(), seed=int(seed))
        else:
            up_bg = st.file_uploader("Upload CSV for SHAP background", type=["csv"], key="bg_csv")
            if up_bg is None:
                st.info("Upload a CSV with columns X1..X8 to compute dataset-based SHAP.")
                st.stop()
            st.session_state.bg_file_bytes = up_bg.read()
            sv_bg, X_bg = shap_background_values_dataset(st.session_state.bg_file_bytes, n=n_bg, seed=int(seed))

        # Global: bar + beeswarm
        colA, colB = st.columns(2)
        with colA:
            st.write("**Mean absolute SHAP (bar)**")
            fig = plt.figure(figsize=(7, 4))
            shap.summary_plot(sv_bg.values, X_bg, plot_type="bar", show=False)
            st.pyplot(fig, clear_figure=True, bbox_inches="tight")
        with colB:
            st.write("**Beeswarm (distribution of impacts)**")
            fig = plt.figure(figsize=(7, 4))
            shap.summary_plot(sv_bg.values, X_bg, show=False)
            st.pyplot(fig, clear_figure=True, bbox_inches="tight")

        # Dependence plots (top features)
        mean_abs = np.mean(np.abs(sv_bg.values), axis=0)
        ordered_cols = list(X_bg.columns)
        order_idx = np.argsort(-mean_abs)
        top_feats = [ordered_cols[i] for i in order_idx[:5]]

        st.write("**Dependence plots**")
        dep1 = st.selectbox("Primary feature", top_feats, index=0, key="dep1")
        dep2_options = ["(auto color)"] + [c for c in ordered_cols if c != dep1]
        dep2 = st.selectbox("Color by (optional)", dep2_options, index=0, key="dep2")
        interaction = dep2 if dep2 != "(auto color)" else "auto"

        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            shap.dependence_plot(
                dep1,
                sv_bg.values,
                X_bg,
                interaction_index=interaction,
                show=False,
                ax=ax
            )
            st.pyplot(fig, clear_figure=True, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            fig, ax = plt.subplots(figsize=(7, 4))
            if dep2 == "(auto color)":
                shap.plots.scatter(sv_bg[:, dep1], ax=ax, show=False)
            else:
                shap.plots.scatter(sv_bg[:, dep1], color=sv_bg[:, dep2], ax=ax, show=False)
            st.pyplot(fig, clear_figure=True, bbox_inches="tight")
            plt.close(fig)

        # Local SHAP for current inputs
        with st.expander("Local explanation for current inputs", expanded=True):
            if st.session_state.current_pred is None:
                st.info("Make a prediction first in the Predict tab to see the local explanation.")
            else:
                values = {k: float(st.session_state.current_inputs[k]) for k in FEATURE_KEYS}
                model, expected = load_model_and_expected()
                explainer = get_explainer()
                X_one = _ordered_df(values, expected)
                sv_one = explainer(X_one)

                st.write("**Waterfall (feature contributions)**")
                try:
                    fig = plt.figure(figsize=(7, 5))
                    shap.plots.waterfall(sv_one[0], max_display=8, show=False)
                    st.pyplot(fig, clear_figure=True, bbox_inches="tight")
                except Exception:
                    fig = plt.figure(figsize=(7, 4))
                    shap.plots.bar(sv_one[0], show=False, max_display=8)
                    st.pyplot(fig, clear_figure=True, bbox_inches="tight")

    except FileNotFoundError:
        st.error(f"Model not found at `{MODEL_PATH}`. Adjust MODEL_PATH or upload the model.")
    except Exception as e:
        st.error(f"SHAP explain error: {e}")

# ==============================
# BATCH TAB
# ==============================
with tab_batch:
    st.markdown("### Batch Predictions (CSV → CSV)")
    st.write("Upload a CSV with columns **X1..X8** in any order; the app will align them.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        try:
            df = pd.read_csv(up)
            model, expected = load_model_and_expected()
            if not set(FEATURE_KEYS).issubset(df.columns):
                missing = [c for c in FEATURE_KEYS if c not in df.columns]
                st.error(f"CSV missing columns: {missing}")
            else:
                ordered_cols = _ordered_df({k: 0.0 for k in FEATURE_KEYS}, expected).columns
                X = df[ordered_cols].astype(np.float32)
                try:
                    model.get_booster().set_param({"predictor":"cpu_predictor","nthread":1})
                except Exception:
                    pass
                try:
                    model.set_params(n_jobs=1, nthread=1, thread_count=1)
                except Exception:
                    pass
                preds = model.predict(X)
                out = df.copy()
                out["Pred_L_over_Lo"] = np.ravel(preds)
                st.success("Batch predictions complete.")
                st.dataframe(out.head(20), use_container_width=True)
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes,
                                   file_name="predictions.csv", mime="text/csv")
        except FileNotFoundError:
            st.error(f"Model not found at `{MODEL_PATH}`. Adjust MODEL_PATH or upload the model.")
        except Exception as e:
            st.error(f"Batch error: {e}")

# ==============================
# HISTORY TAB
# ==============================
with tab_hist:
    st.markdown("### Session History")
    if len(st.session_state.history) == 0:
        st.info("No history yet. Make a prediction in the Predict tab.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download History CSV", data=csv_bytes,
                           file_name="history.csv", mime="text/csv")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ==============================
# ARTICLE INFO TAB
# ==============================
with tab_article:
    st.markdown("### Article & Authors")
    st.markdown(
        """
        <div style="font-size:28px; font-weight:800; line-height:1.25;">
        Machine Learning-Based Modeling of Artificial Recharge and Cutoff Walls for Seawater Intrusion Control in Coastal Aquifers
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:20px; font-weight:700; margin-top:0.5rem;">
        Developers: Mohamed Kamel Elshaarawy & Asaad Mater Armanuos
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="muted" style="font-size:16px; margin-top:.5rem;">
        This app provides a fast, interpretable predictor for the relative seawater intrusion (SWI) wedge length (L/Lo)
        under artificial recharge and cutoff wall configurations.
        </div>
        """, unsafe_allow_html=True,
    )

    citation = (
        "Elshaarawy, M.K., & Armanuos, A.M. (n.d.). Machine Learning-Based Modeling of Artificial Recharge "
        "and Cutoff Walls for Seawater Intrusion Control in Coastal Aquifers. [Software]."
    )
    st.download_button("Download Citation (.txt)", data=citation.encode("utf-8"),
                       file_name="citation.txt", mime="text/plain")
