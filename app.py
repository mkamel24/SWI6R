# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# SWI Wedge Length Ratio (L/Lo) — Smart Predictor (CatBoost)
#   - Model load: repo path models/CGB.joblib OR upload .joblib
#   - Clean, card-style GUI (Predict, Explain (SHAP), Batch, History, Article Info)
#   - Paper title as header (reduced font) + updated authors/affiliations
#   - Deterministic CPU inference where possible
# ------------------------------------------------------------

import io
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

# ==============================
# Page / theme config
# ==============================
st.set_page_config(
    page_title="SWI Wedge Length Ratio – Smart Predictor (L/Lo, CatBoost)",
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
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid var(--ui-border) !important;
        border-radius: 10px;
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
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
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

      .muted { color: var(--ui-text-muted); font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Paper strings (updated)
# ==============================
ARTICLE_TITLE = (
    "Simulating the Effectiveness of Artificial Recharge and Cutoff Walls for "
    "Saltwater Intrusion Control with Explainable ML and GUI Deployment"
)
ARTICLE_AUTHORS_HTML = (
    "Mohamed Kamel Elshaarawy<sup>1,*</sup> &amp; "
    "Asaad M. Armanuos<sup>2,*</sup>"
)
ARTICLE_AFFILS_HTML = (
    "1 Civil Engineering Department, Faculty of Engineering, Horus University-Egypt, "
    "New Damietta 34517, Egypt; <a href='mailto:melshaarawy@horus.edu.eg'>melshaarawy@horus.edu.eg</a> (M.K.E.)<br>"
    "2 Irrigation and Hydraulics Engineering Department, Faculty of Engineering, Tanta University, "
    "Tanta 31733, Egypt; <a href='mailto:asaad.matter@f-eng.tanta.edu.eg'>asaad.matter@f-eng.tanta.edu.eg</a> (A.M.A.)<br>"
    "*Corresponding author"
)
ARTICLE_JOURNAL = "Catena"

# ==============================
# Config / constants
# ==============================
MODEL_PATH_DEFAULT = "models/CGB.joblib"  # repo-relative CatBoost model
IMAGE_CANDIDATES = [
    Path("assets/sketch22.png"),
    Path("assets/sketch.png"),
    Path("sketch22.png"),
    Path("sketch.png"),
]

# Feature names (MUST match training order)
FEATURE_KEYS = [
    "ρs/ρf",       # X1  Relative density
    "K/Ko",        # X2  Relative hydraulic conductivity
    "Qi/(Ko·Lo²)", # X3  Relative recharge rate
    "i",           # X4  Hydraulic gradient
    "Xi/Lo",       # X5  Relative well distance
    "Yi/Lo",       # X6  Relative well depth
    "Xb/Lo",       # X7  Relative barrier wall distance
    "Db/Lo",       # X8  Relative barrier wall depth
]

HELP = {
    "ρs/ρf": "Relative density (saltwater/freshwater).",
    "K/Ko": "Relative hydraulic conductivity.",
    "Qi/(Ko·Lo²)": "Relative recharge rate.",
    "i": "Hydraulic gradient (dimensionless).",
    "Xi/Lo": "Recharge well distance (relative).",
    "Yi/Lo": "Recharge well depth (relative).",
    "Xb/Lo": "Barrier wall distance (relative).",
    "Db/Lo": "Barrier wall depth (relative).",
}

# Defaults (tune for your dataset)
DEFAULTS = {
    "ρs/ρf": 1.025,
    "K/Ko": 1.000,
    "Qi/(Ko·Lo²)": 0.0010,
    "i": 0.0120,
    "Xi/Lo": 1.000,
    "Yi/Lo": 0.417,
    "Xb/Lo": 0.313,
    "Db/Lo": 0.323,
}

# Reasonable global ranges for SHAP uniform background (edit if you have dataset mins/maxes)
FEATURE_RANGES = {
    "ρs/ρf":       (0.995, 1.035, DEFAULTS["ρs/ρf"]),
    "K/Ko":        (0.30,  2.50,  DEFAULTS["K/Ko"]),
    "Qi/(Ko·Lo²)": (1e-5,  5e-3,  DEFAULTS["Qi/(Ko·Lo²)"]),
    "i":           (1e-3,  0.05,  DEFAULTS["i"]),
    "Xi/Lo":       (0.10,  3.00,  DEFAULTS["Xi/Lo"]),
    "Yi/Lo":       (0.05,  1.00,  DEFAULTS["Yi/Lo"]),
    "Xb/Lo":       (0.05,  2.00,  DEFAULTS["Xb/Lo"]),
    "Db/Lo":       (0.05,  1.00,  DEFAULTS["Db/Lo"]),
}

NUM_SPEC = {
    "ρs/ρf":       dict(step=1e-4, fmt="%.6f"),
    "K/Ko":        dict(step=1e-4, fmt="%.6f"),
    "Qi/(Ko·Lo²)": dict(step=1e-5, fmt="%.6f"),
    "i":           dict(step=1e-5, fmt="%.6f"),
    "Xi/Lo":       dict(step=1e-3, fmt="%.6f"),
    "Yi/Lo":       dict(step=1e-3, fmt="%.6f"),
    "Xb/Lo":       dict(step=1e-3, fmt="%.6f"),
    "Db/Lo":       dict(step=1e-3, fmt="%.6f"),
}

PRESETS = {
    "— choose a preset —": None,
    "Baseline (defaults)": DEFAULTS,
    "Higher recharge": {**DEFAULTS, "Qi/(Ko·Lo²)": 0.0015},
    "Deeper barrier":   {**DEFAULTS, "Db/Lo": 0.45},
    "Farther barrier":  {**DEFAULTS, "Xb/Lo": 0.60},
}

# ==============================
# Helpers
# ==============================
def _force_catboost_cpu_single_thread(model_obj):
    """Best-effort: force CatBoost to CPU, 1 thread (safe no-op for other models)."""
    try:
        import catboost
        if isinstance(model_obj, (catboost.CatBoostRegressor, catboost.CatBoostClassifier)):
            try:
                model_obj.set_params(task_type="CPU", thread_count=1, random_seed=42)
            except Exception:
                pass
    except Exception:
        pass
    try:
        if hasattr(model_obj, "set_params"):
            model_obj.set_params(thread_count=1)
    except Exception:
        pass
    return model_obj

def _ordered_df(values: dict, expected_names: list | None):
    """Build 1-row DataFrame in model's expected order when available."""
    if expected_names and set(map(str, expected_names)) == set(FEATURE_KEYS):
        cols = list(map(str, expected_names))
    else:
        cols = FEATURE_KEYS[:]
    row = [values[c] for c in cols]
    return pd.DataFrame([row], columns=cols).astype(np.float32)

def _get_model_feature_names(model_obj):
    """Try to read feature names from model (CatBoost/pandas training)."""
    names = None
    for attr in ("feature_names_in_", "feature_names_", "feature_names"):
        if hasattr(model_obj, attr):
            try:
                candidates = list(getattr(model_obj, attr))
                if candidates and set(candidates) == set(FEATURE_KEYS):
                    names = candidates
                    break
            except Exception:
                pass
    return names

def json_download_bytes(obj):
    buf = BytesIO()
    buf.write(json.dumps(obj, indent=2).encode("utf-8"))
    buf.seek(0)
    return buf

def sample_background_df(ranges: dict, n: int = 256, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(None if seed is None else int(seed))
    data = {}
    for k in FEATURE_KEYS:
        lo, hi, _ = ranges[k]
        data[k] = rng.uniform(float(lo), float(hi), size=n)
    return pd.DataFrame(data).astype(np.float32)

def find_local_image() -> Image.Image | None:
    for p in IMAGE_CANDIDATES:
        if p.exists():
            try:
                return Image.open(p)
            except Exception:
                pass
    return None

def ranges_key_tuple() -> tuple:
    """Hashable key for caching background SHAP when ranges change."""
    return tuple((k, tuple(map(float, FEATURE_RANGES[k]))) for k in FEATURE_KEYS)

def clip_to_bounds(vals: dict) -> dict:
    out = {}
    for k, v in vals.items():
        lo, hi, _ = FEATURE_RANGES[k]
        out[k] = min(max(float(v), float(lo)), float(hi))
    return out

# ==============================
# Session state
# ==============================
if "model" not in st.session_state:
    st.session_state.model = None
if "model_origin" not in st.session_state:
    st.session_state.model_origin = f"path:{MODEL_PATH_DEFAULT}"
if "expected_names" not in st.session_state:
    st.session_state.expected_names = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, each with time + Xs + pred
if "current_pred" not in st.session_state:
    st.session_state.current_pred = None
if "current_inputs" not in st.session_state:
    st.session_state.current_inputs = {k: DEFAULTS[k] for k in FEATURE_KEYS}
if "sketch_bytes" not in st.session_state:
    st.session_state.sketch_bytes = None
if "bg_file_bytes" not in st.session_state:
    st.session_state.bg_file_bytes = None

# ==============================
# Sidebar: model loader (repo path OR upload)
# ==============================
with st.sidebar:
    st.header("Model Loader")
    mode = st.radio("Load CatBoost model from:", ["Repo path", "Upload"], horizontal=True)

    if mode == "Repo path":
        exists = Path(MODEL_PATH_DEFAULT).exists()
        st.caption(f"Default path: `{MODEL_PATH_DEFAULT}` — {'✅ found' if exists else '❗ not found'}")
        mp = st.text_input("Model path (.joblib)", MODEL_PATH_DEFAULT)
        if st.button("Load model", type="primary", use_container_width=True):
            try:
                st.session_state.model = joblib.load(mp)
                st.session_state.model = _force_catboost_cpu_single_thread(st.session_state.model)
                st.session_state.expected_names = _get_model_feature_names(st.session_state.model)
                st.session_state.model_origin = f"path:{mp}"
                st.success("Model loaded from path.")
            except Exception as e:
                st.error(f"Model load failed: {e}")
    else:
        up_model = st.file_uploader("Upload .joblib model", type=["joblib"])
        if up_model and st.button("Load uploaded model", type="primary", use_container_width=True):
            try:
                st.session_state.model = joblib.load(io.BytesIO(up_model.read()))
                st.session_state.model = _force_catboost_cpu_single_thread(st.session_state.model)
                st.session_state.expected_names = _get_model_feature_names(st.session_state.model)
                st.session_state.model_origin = "upload:memory"
                st.success("Uploaded model loaded.")
            except Exception as e:
                st.error(f"Model load failed: {e}")

    if st.button("Clear model", use_container_width=True):
        st.session_state.model = None
        st.session_state.expected_names = None
        st.info("Model cleared.")

# ==============================
# Header (paper title smaller font)
# ==============================
st.markdown(
    f"""
    <div style="font-size:26px; font-weight:800; line-height:1.25; margin-bottom:.25rem;">
      {ARTICLE_TITLE}
    </div>
    <div class="muted" style="margin-bottom:.75rem;">
      Model source: <code>{st.session_state.model_origin}</code>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_predict, tab_explain, tab_batch, tab_hist, tab_article = st.tabs(
    ["Predict", "Explain", "Batch", "History", "Article Info"]
)

# ==============================
# Predict tab
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
            st.session_state.current_inputs = PRESETS[preset].copy()

        # Number inputs (4 per row)
        ordered_keys = FEATURE_KEYS[:]
        for i in range(0, len(ordered_keys), 4):
            cols = st.columns(4)
            for j, k in enumerate(ordered_keys[i:i+4]):
                spec = NUM_SPEC[k]
                default_val = float(st.session_state.current_inputs.get(k, DEFAULTS[k]))
                with cols[j]:
                    st.session_state.current_inputs[k] = st.number_input(
                        k, value=default_val, step=float(spec["step"]),
                        format=spec["fmt"], help=HELP.get(k, "")
                    )

        st.caption("Tip: Use **Save Inputs (JSON)** to download a template; **Load Inputs** to restore.")

        # Bottom row buttons
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
        with c1:
            if st.button("Predict", use_container_width=True, type="primary"):
                if st.session_state.model is None:
                    st.error("Please load the CatBoost model (sidebar).")
                else:
                    try:
                        vals = clip_to_bounds(st.session_state.current_inputs)
                        X = _ordered_df(vals, st.session_state.expected_names)
                        y = st.session_state.model.predict(X.values)
                        y = float(np.ravel(y)[0])
                        st.session_state.current_pred = y
                        st.session_state.last_inputs = st.session_state.current_inputs.copy()
                        rec = {"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **vals, "Prediction": round(y, 6)}
                        st.session_state.history.append(rec)
                        st.success("Prediction complete.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.current_pred = None
                st.session_state.current_inputs = {k: DEFAULTS[k] for k in FEATURE_KEYS}
                st.info("Cleared.")
        with c3:
            disabled = st.session_state.last_inputs is None
            if st.button("Recall Last", use_container_width=True, disabled=disabled):
                if st.session_state.last_inputs is not None:
                    st.session_state.current_inputs = st.session_state.last_inputs.copy()
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
            st.info("No image found. Add one at `assets/sketch22.png` / `assets/sketch.png` in the repo or upload above.")
        else:
            st.image(img, use_container_width=True)

# ==============================
# Explain tab (SHAP)
# ==============================
with tab_explain:
    st.markdown("### Explain (SHAP)")
    if st.session_state.model is None:
        st.info("Load a model first (sidebar) to enable SHAP explanations.")
    else:
        bg_src = st.radio("Background source for global SHAP:",
                          ["Uniform (use bounds below)", "Dataset (upload CSV)"],
                          horizontal=True)
        n_bg = st.slider("Background sample size", 100, 2000, 256, 50,
                         help="Larger = smoother but slower. 256–512 is good for 8 features.")
        seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42, step=1)

        try:
            # Build background data
            if bg_src.startswith("Uniform"):
                df_bg = sample_background_df(FEATURE_RANGES, n_bg, int(seed))
            else:
                up_bg = st.file_uploader("Upload CSV with columns matching the model features", type=["csv"], key="bg_csv")
                if up_bg is None:
                    st.info("Upload a CSV containing the 8 feature columns to compute dataset-based SHAP.")
                    st.stop()
                df = pd.read_csv(up_bg)
                missing = [c for c in FEATURE_KEYS if c not in df.columns]
                if missing:
                    st.error(f"Dataset missing columns: {missing}")
                    st.stop()
                if len(df) > n_bg:
                    df_bg = df.sample(n=n_bg, random_state=int(seed)).reset_index(drop=True)
                else:
                    df_bg = df.copy().reset_index(drop=True)

            # Align column order to model expectation
            expected_names = st.session_state.expected_names
            X_bg = _ordered_df({k: 0.0 for k in FEATURE_KEYS}, expected_names)
            X_bg = df_bg[X_bg.columns].astype(np.float32)

            # Explainer + SHAP values
            explainer = shap.Explainer(st.session_state.model)
            sv_bg = explainer(X_bg)

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

            # Dependence plots
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
                    X_one = _ordered_df(values, expected_names)
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

        except Exception as e:
            st.error(f"SHAP error: {e}")

# ==============================
# Batch tab
# ==============================
with tab_batch:
    st.markdown("### Batch Predictions (CSV → CSV)")
    st.write("Upload a CSV with columns **exactly**:")
    st.code(", ".join(FEATURE_KEYS), language="text")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        if st.session_state.model is None:
            st.error("Load the CatBoost model first (sidebar).")
        else:
            try:
                df = pd.read_csv(up)
                missing = [c for c in FEATURE_KEYS if c not in df.columns]
                if missing:
                    st.error(f"CSV missing columns: {missing}")
                else:
                    expected_names = st.session_state.expected_names
                    ordered_cols = _ordered_df({k: 0.0 for k in FEATURE_KEYS}, expected_names).columns
                    X = df[ordered_cols].astype(np.float32).values
                    preds = st.session_state.model.predict(X)
                    preds = np.ravel(preds).astype(float)
                    out = df.copy()
                    out["Pred_L_over_Lo"] = preds
                    st.success("Batch predictions complete.")
                    st.dataframe(out.head(20), use_container_width=True)
                    st.download_button("Download predictions CSV",
                                       data=out.to_csv(index=False).encode("utf-8"),
                                       file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch error: {e}")

# ==============================
# History tab
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
# Article Info tab
# ==============================
with tab_article:
    st.markdown("### Article & Authors")
    st.markdown(
        f"""
        <div style="font-size:24px; font-weight:800; line-height:1.25;">
        {ARTICLE_TITLE}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="font-size:18px; font-weight:700; margin-top:0.5rem;">
        {ARTICLE_AUTHORS_HTML}
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="font-size:16px; margin-top:0.5rem;">
        {ARTICLE_AFFILS_HTML}
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="font-size:16px; font-style:italic; margin-top:0.6rem;">
        {ARTICLE_JOURNAL}
        </div>
        """, unsafe_allow_html=True,
    )

    citation = (
        "Elshaarawy, M.K., & Armanuos, A.M. (n.d.). "
        "Simulating the Effectiveness of Artificial Recharge and Cutoff Walls for Saltwater Intrusion Control "
        "with Explainable ML and GUI Deployment. Catena."
    )
    st.download_button("Download Citation (.txt)", data=citation.encode("utf-8"),
                       file_name="citation.txt", mime="text/plain")
