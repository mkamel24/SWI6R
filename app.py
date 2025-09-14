# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# SWI Wedge Length Ratio (L/Lo) — Smart Predictor (CatBoost)
#   - Styled to match your first GUI (cards, big-number, tabs)
#   - Defaults to CatBoost model at C:/Users/asus1/Desktop/CGB.joblib
#   - Forces CPU + single thread for deterministic inference (best-effort)
# ------------------------------------------------------------

import io
import json
from io import BytesIO
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

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
        --ui-bg: #ffffff;
        --ui-card: #ffffff;
        --ui-border: #cccccc;
        --ui-text: #000000;
        --ui-text-muted: #666666;
        --ui-accent: #000000;
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

      [data-testid="stFileUploader"] section {
        border: 1px dashed var(--ui-border);
        background: #f9f9f9;
        border-radius: 12px;
        padding: .8rem;
        color: var(--ui-text);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Article strings (keep/edit)
# ==============================
ARTICLE_TITLE   = "Simulating the Effectiveness of Artificial Recharge and Cutoff Walls for Saltwater Intrusion Control with Explainable ML and GUI Deployment"
ARTICLE_AUTHORS = "Mohamed Kamel Elshaarawy¹,*; Asaad M. Armanuos²,*"
ARTICLE_JOURNAL = "Catena"
ARTICLE_AFFILS  = [
    "¹ Affiliation (update here)",
    "² Affiliation (update here)",
]

# ==============================
# App configuration
# ==============================
MODEL_PATH_DEFAULT = r"C:/Users/asus1/Desktop/CGB.joblib"  # <-- CatBoost model here
IMAGE_CANDIDATES = [
    Path(r"C:/Users/asus1/Desktop/sketch.png"),
    Path("assets/sketch.png"),
    Path("assets/sketch22.png"),
    Path("sketch.png"),
    Path("sketch22.png"),
]

FEATURE_ORDER = [
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

PRESETS = {
    "— choose a preset —": None,
    "Baseline (defaults)": DEFAULTS,
    "Higher recharge": {**DEFAULTS, "Qi/(Ko·Lo²)": 0.0015},
    "Deeper barrier":   {**DEFAULTS, "Db/Lo": 0.45},
    "Farther barrier":  {**DEFAULTS, "Xb/Lo": 0.60},
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

# ==============================
# Session state
# ==============================
if "model" not in st.session_state:
    st.session_state.model = None
if "model_path" not in st.session_state:
    st.session_state.model_path = MODEL_PATH_DEFAULT
if "history" not in st.session_state:
    st.session_state.history = []
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "current_pred" not in st.session_state:
    st.session_state.current_pred = None
if "current_inputs" not in st.session_state:
    st.session_state.current_inputs = {k: float(v) for k, v in DEFAULTS.items()}
if "sketch_bytes" not in st.session_state:
    st.session_state.sketch_bytes = None
if "image_path" not in st.session_state:
    st.session_state.image_path = str(IMAGE_CANDIDATES[0]) if IMAGE_CANDIDATES else ""

# ==============================
# Utilities
# ==============================
def load_model(path_or_bytes):
    """Load CatBoost/scikit model from path or uploaded file (.joblib)."""
    if hasattr(path_or_bytes, "read"):
        data = path_or_bytes.read()
        return joblib.load(io.BytesIO(data))
    return joblib.load(path_or_bytes)

def force_deterministic_cpu(model_obj):
    """
    Best-effort: make CatBoost run on CPU with single thread.
    Works when the loaded object is catboost.CatBoostRegressor/Classifier
    or a scikit wrapper with set_params().
    """
    try:
        import catboost
        if isinstance(model_obj, (catboost.CatBoostRegressor, catboost.CatBoostClassifier)):
            try:
                # Most common CatBoost knobs for inference determinism/CPU
                model_obj.set_params(task_type='CPU', thread_count=1, random_seed=42)
            except Exception:
                pass
    except Exception:
        pass  # CatBoost not installed or different estimator
    try:
        # Scikit-style escape hatch
        if hasattr(model_obj, "set_params"):
            model_obj.set_params(thread_count=1)
    except Exception:
        pass
    return model_obj

def predict_single(model_obj, inputs_dict):
    """Return float prediction given inputs ordered per FEATURE_ORDER."""
    x = np.array([[float(inputs_dict[k]) for k in FEATURE_ORDER]], dtype=np.float32)
    y = model_obj.predict(x)
    try:
        return float(np.ravel(y)[0])
    except Exception:
        return float(y)

def push_history_row(inputs_dict, pred_val):
    row = {"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row.update({k: float(inputs_dict[k]) for k in FEATURE_ORDER})
    row["Pred_L_over_Lo"] = round(float(pred_val), 6)
    st.session_state.history.append(row)

def history_dataframe():
    if not st.session_state.history:
        return pd.DataFrame(columns=["Time"] + FEATURE_ORDER + ["Pred_L_over_Lo"])
    return pd.DataFrame(st.session_state.history)

def json_download_bytes(obj):
    buf = BytesIO()
    buf.write(json.dumps(obj, indent=2).encode("utf-8"))
    buf.seek(0)
    return buf

def copy_to_clipboard_js(text):
    st.components.v1.html(
        f"""
        <button onclick="navigator.clipboard.writeText('{text}');"
                style="padding:8px 12px;border:1px solid #ccc;background:#fff;border-radius:10px;cursor:pointer;font-weight:700;">
            Copy Result
        </button>
        """,
        height=48,
    )

def find_local_image() -> Image.Image | None:
    if st.session_state.sketch_bytes is not None:
        try:
            return Image.open(BytesIO(st.session_state.sketch_bytes))
        except Exception:
            pass
    p = Path(st.session_state.image_path) if st.session_state.image_path else None
    if p and p.exists():
        try:
            return Image.open(p)
        except Exception:
            pass
    for cand in IMAGE_CANDIDATES:
        if cand.exists():
            try:
                return Image.open(cand)
            except Exception:
                pass
    return None

# ==============================
# Sidebar: model & reference image
# ==============================
with st.sidebar:
    st.header("Model & Resources")

    st.subheader("Model (CatBoost .joblib)")
    model_source = st.radio("Load model from:", ["Path", "Upload"], horizontal=True)
    if model_source == "Path":
        mp = st.text_input("Model path (.joblib)", st.session_state.model_path)
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Load model", type="primary", use_container_width=True):
                try:
                    st.session_state.model = force_deterministic_cpu(load_model(mp))
                    st.session_state.model_path = mp
                    st.success("CatBoost model loaded.")
                except Exception as e:
                    st.error(f"Model load failed: {e}")
        with c2:
            if st.button("Clear model", use_container_width=True):
                st.session_state.model = None
                st.info("Model cleared.")
    else:
        uploaded_model = st.file_uploader("Upload CatBoost .joblib", type=["joblib"])
        if uploaded_model and st.button("Load uploaded model", type="primary", use_container_width=True):
            try:
                st.session_state.model = force_deterministic_cpu(load_model(uploaded_model))
                st.success("Uploaded CatBoost model loaded.")
            except Exception as e:
                st.error(f"Model load failed: {e}")

    st.subheader("Reference Image (fallback path)")
    st.session_state.image_path = st.text_input("Image path", st.session_state.image_path)

# ==============================
# Header
# ==============================
st.title("SWI Wedge Length Ratio – Smart Predictor (L/Lo, CatBoost)")
st.caption("CatBoost model • quick, reliable, and clean UI.")

# ==============================
# Tabs
# ==============================
tab_predict, tab_batch, tab_hist, tab_article = st.tabs(
    ["Predict", "Batch", "History", "Article Info"]
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

        # Preset selector
        preset = st.selectbox("Preset", list(PRESETS.keys()), index=0)
        if PRESETS.get(preset):
            st.session_state.current_inputs = {k: float(v) for k, v in PRESETS[preset].items()}

        # Inputs grid (4 per row)
        ordered_keys = FEATURE_ORDER[:]
        for i in range(0, len(ordered_keys), 4):
            cols = st.columns(4)
            for j, k in enumerate(ordered_keys[i:i+4]):
                spec = NUM_SPEC.get(k, dict(step=1e-4, fmt="%.6f"))
                default_val = float(st.session_state.current_inputs.get(k, DEFAULTS.get(k, 0.0)))
                with cols[j]:
                    st.session_state.current_inputs[k] = st.number_input(
                        k,
                        value=default_val,
                        step=float(spec["step"]),
                        format=spec["fmt"],
                        help=HELP.get(k, ""),
                    )

        st.caption("Tip: Save/Load your inputs with JSON; use Recall to restore previous run.")

        # Bottom row buttons
        c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
        with c1:
            if st.button("Predict", use_container_width=True, type="primary"):
                if st.session_state.model is None:
                    st.error("Load the CatBoost model first (sidebar).")
                else:
                    try:
                        yhat = predict_single(st.session_state.model, st.session_state.current_inputs)
                        st.session_state.current_pred = float(yhat)
                        st.session_state.last_inputs = st.session_state.current_inputs.copy()
                        push_history_row(st.session_state.current_inputs, yhat)
                        st.success("Prediction complete.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.current_pred = None
                st.session_state.current_inputs = {k: float(v) for k, v in DEFAULTS.items()}
                st.info("Cleared.")

        with c3:
            disabled = st.session_state.last_inputs is None
            if st.button("Recall Last", use_container_width=True, disabled=disabled):
                if st.session_state.last_inputs is not None:
                    st.session_state.current_inputs = {k: float(v) for k, v in st.session_state.last_inputs.items()}
                    st.success("Recalled last inputs.")

        with c4:
            if st.session_state.current_pred is not None:
                copy_to_clipboard_js(f"{st.session_state.current_pred:.6f}")
            else:
                st.button("Copy Result", disabled=True, use_container_width=True)

        with c5:
            buf = json_download_bytes(st.session_state.current_inputs)
            st.download_button("Save Inputs (JSON)", data=buf, file_name="swi_inputs.json",
                               mime="application/json", use_container_width=True)

        with c6:
            up = st.file_uploader("Load Inputs", type=["json"], label_visibility="collapsed", key="upl_json_predict")
            if up is not None:
                try:
                    data = json.loads(up.read().decode("utf-8"))
                    for k in FEATURE_ORDER:
                        if k in data:
                            st.session_state.current_inputs[k] = float(data[k])
                    st.success("Inputs loaded.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

    # RIGHT: reference sketch (with upload fallback)
    with col_right:
        st.markdown("#### Reference Sketch")
        up_img = st.file_uploader("Upload sketch (PNG/JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if up_img is not None:
            st.session_state.sketch_bytes = up_img.read()

        img = find_local_image()
        if img is None:
            st.info("No image found. Set a valid path in the sidebar, add one at `assets/sketch.png`, or upload above.")
        else:
            st.image(img, use_container_width=True)

# ==============================
# Batch tab
# ==============================
with tab_batch:
    st.markdown("### Batch Predictions (CSV → CSV)")
    st.write("Upload a CSV with **columns matching the model order**:")
    st.code(", ".join(FEATURE_ORDER), language="text")

    batch_file = st.file_uploader("Upload CSV", type=["csv"])
    if batch_file is not None:
        try:
            df_in = pd.read_csv(batch_file)
            missing = [c for c in FEATURE_ORDER if c not in df_in.columns]
            if missing:
                st.error(f"CSV missing required columns: {missing}")
            elif st.session_state.model is None:
                st.error("Load the CatBoost model first (sidebar).")
            else:
                X = df_in[FEATURE_ORDER].astype(np.float32).values
                try:
                    preds = st.session_state.model.predict(X)
                    preds = np.ravel(preds).astype(float)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    preds = None
                if preds is not None:
                    df_out = df_in.copy()
                    df_out["Pred_L_over_Lo"] = preds
                    st.success("Batch predictions complete.")
                    st.dataframe(df_out.head(20), use_container_width=True)
                    st.download_button("Download results CSV",
                                       data=df_out.to_csv(index=False).encode("utf-8"),
                                       file_name="swi_predictions.csv",
                                       mime="text/csv")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# ==============================
# History tab
# ==============================
with tab_hist:
    st.markdown("### Session History")
    df_hist = history_dataframe()
    if df_hist.empty:
        st.info("No predictions yet. Make a run in the Predict tab.")
    else:
        st.dataframe(df_hist, use_container_width=True)
        st.download_button("Download history CSV",
                           data=df_hist.to_csv(index=False).encode("utf-8"),
                           file_name="swi_history.csv",
                           mime="text/csv")
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
        <div style="font-size:28px; font-weight:800; line-height:1.25;">
        {ARTICLE_TITLE}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="font-size:20px; font-weight:700; margin-top:0.5rem;">
        {ARTICLE_AUTHORS}
        </div>
        """, unsafe_allow_html=True,
    )
    if ARTICLE_AFFILS:
        st.markdown(
            "<div style='font-size:18px; margin-top:0.5rem;'>"
            + "<br>".join(ARTICLE_AFFILS) +
            "</div>",
            unsafe_allow_html=True,
        )
    if ARTICLE_JOURNAL and ARTICLE_JOURNAL.strip():
        st.markdown(
            f"<div style='font-size:18px; font-style:italic; margin-top:0.6rem;'>"
            f"{ARTICLE_JOURNAL}</div>",
            unsafe_allow_html=True,
        )

    citation = (
        f"{ARTICLE_AUTHORS.replace(';', ',')} (n.d.). {ARTICLE_TITLE}. {ARTICLE_JOURNAL}."
    )
    st.download_button("Download Citation (.txt)", data=citation.encode("utf-8"),
                       file_name="citation.txt", mime="text/plain")
