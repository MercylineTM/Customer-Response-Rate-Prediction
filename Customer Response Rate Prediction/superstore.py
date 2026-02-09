# superstore_dashboard.py
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, precision_recall_curve
)

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Gold Membership Campaign Targeting Dashboard",
    layout="wide"
)

# ---------------- Session state ----------------
if "data" not in st.session_state:
    st.session_state.data = None

# ---------------- Background (blur image) + theme styling ----------------
def set_blur_bg(image_path: str, blur_px: int = 10, overlay_alpha: float = 0.55):
    with open(image_path, "rb") as f:
        bg_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* ===== Blurred full-page background ===== */
        .stApp {{
            background: transparent;
            color: #ffffff;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/jpg;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            filter: blur({blur_px}px);
            transform: scale(1.06);
            opacity: 0.95;
            z-index: -2;
        }}

        /* ===== Dark overlay for readability ===== */
        .stApp::after {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            background: rgba(2, 6, 23, {overlay_alpha});
            z-index: -1;
        }}

        /* ===== Header styling ===== */
        .app-header {{
            padding: 14px 0 16px 0;
            margin-bottom: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.18);
        }}
        .app-header h1 {{
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 3px 10px rgba(0,0,0,0.35);
        }}
        .app-header p {{
            margin: 6px 0 0 0;
            color: rgba(255,255,255,0.85);
        }}

        /* ===== Sidebar glass ===== */
        section[data-testid="stSidebar"] {{
            background-color: rgba(2, 6, 23, 0.75);
            color: #ffffff;
            backdrop-filter: blur(6px);
        }}

        /* ===== KPI Metric cards (glass) ===== */
        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.14) !important;
            padding: 18px !important;
            border-radius: 16px !important;
            border: 1px solid rgba(250, 204, 21, 0.35) !important;
            box-shadow: 0 8px 22px rgba(0,0,0,0.35) !important;
            backdrop-filter: blur(8px);
        }}
        div[data-testid="stMetricLabel"] > div {{
            color: rgba(255,255,255,0.92) !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }}
        div[data-testid="stMetricValue"] > div {{
            color: #ffffff !important;
            font-weight: 800 !important;
            font-size: 1.8rem !important;
        }}

        /* ===== Tables & dataframes ===== */
        .stDataFrame, .stTable {{
            background-color: rgba(255,255,255,0.10) !important;
            border-radius: 14px !important;
            backdrop-filter: blur(8px);
        }}
        .stDataFrame * {{
            color: #ffffff !important;
        }}

        /* ===== Primary buttons ===== */
        button[kind="primary"] {{
            background-color: #facc15 !important;
            color: #020617 !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
        }}

        /* ===== Sliders ===== */
        .stSlider * {{
            color: #ffffff !important;
        }}
        .stSlider .rc-slider-track {{
            background-color: #facc15 !important;
        }}
        .stSlider .rc-slider-handle {{
            border-color: #facc15 !important;
        }}

        /* ===== Tabs ===== */
        button[data-baseweb="tab"] * {{
            color: rgba(255,255,255,0.65) !important;
            font-weight: 650 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] * {{
            color: #ffffff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Change this if your image has a different name
set_blur_bg("bg.png", blur_px=10, overlay_alpha=0.55)

# ---------------- Header (no logo) ----------------
st.markdown(
    """
    <div class="app-header">
        <h1>Gold Membership Campaign Targeting Dashboard</h1>
        <p>Use predicted response probabilities to target calls, measure model value, and explore segments.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar: reset (works in both phases) ----------------
with st.sidebar:
    st.header("Controls")
    if st.button("Reset / Upload new file"):
        st.session_state.data = None
        st.rerun()

# ---------------- Upload phase (uploader visible only here) ----------------
if st.session_state.data is None:
    # Make uploader readable (black text) while it is visible
    st.markdown(
        """
        <style>
        /* Dropdown/select text (black) */
        div[data-baseweb="select"] span { color: #000000 !important; font-weight: 600; }
        div[data-baseweb="menu"] span { color: #000000 !important; }
        div[data-baseweb="select"] > div { background-color: #f8fafc !important; }

        /* File uploader readable */
        div[data-testid="stFileUploader"] {
            background-color: #f8fafc !important;
            border-radius: 14px !important;
        }
        div[data-testid="stFileUploader"] * {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload CSV with predictions (must include Response_Probability; Response optional)",
        type=["csv"],
        key="uploader"
    )

    if uploaded is not None:
        st.session_state.data = pd.read_csv(uploaded)
        st.rerun()

    st.info("Upload your predictions CSV to start. After upload, the uploader will disappear and the analysis dashboard will load.")
    st.stop()

# ---------------- Analysis phase (uploader hidden; df always defined) ----------------
df = st.session_state.data.copy()
df.columns = [c.strip() for c in df.columns]

# --- Detect probability column safely ---
prob_col_candidates = ["Response_Probability", "response_probability", "prob", "proba", "Probability"]
prob_col = next((c for c in prob_col_candidates if c in df.columns), None)

if prob_col is None:
    st.error("Could not find a probability column. Please include a column named 'Response_Probability'.")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Optional label and id columns
label_col = "Response" if "Response" in df.columns else None
id_col = "Id" if "Id" in df.columns else ("ID" if "ID" in df.columns else None)

# Clean probability column
df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
df = df[df[prob_col].notna()].copy()
df[prob_col] = df[prob_col].clip(0, 1)

# ---------------- Sidebar controls (analysis phase) ----------------
with st.sidebar:
    st.subheader("Targeting")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.60, 0.01)
    top_pct = st.slider("Or target top % customers", 1, 100, 20, 1)
    use_top_pct = st.checkbox("Use top % targeting instead of threshold", value=True)

# Sort by probability
df_sorted = df.sort_values(prob_col, ascending=False).reset_index(drop=True)

# Apply targeting rule
if use_top_pct:
    n_target = int(np.ceil(len(df_sorted) * (top_pct / 100)))
    df_sorted["Target"] = 0
    df_sorted.loc[: max(n_target - 1, 0), "Target"] = 1
    active_rule_text = f"Top {top_pct}% (n={n_target})"
else:
    df_sorted["Target"] = (df_sorted[prob_col] >= threshold).astype(int)
    n_target = int(df_sorted["Target"].sum())
    active_rule_text = f"Probability ≥ {threshold:.2f} (n={n_target})"

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total customers", f"{len(df_sorted):,}")
c2.metric("Customers targeted", f"{n_target:,}", help=active_rule_text)
c3.metric("Target rate", f"{(n_target/len(df_sorted))*100:.1f}%")

if label_col:
    c4.metric("Base response rate", f"{df_sorted[label_col].mean()*100:.1f}%")
else:
    c4.metric("Base response rate", "N/A", help="Add Response column (0/1) to compute performance metrics.")

st.divider()

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Value", "Segments", "Call List"])

# ---------- Tab 1: Overview ----------
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Probability distribution")
        fig = plt.figure()
        plt.hist(df_sorted[prob_col], bins=30)
        plt.xlabel("Predicted probability")
        plt.ylabel("Customers")
        st.pyplot(fig)

    with right:
        st.subheader("Targeting summary")
        st.write(f"**Active rule:** {active_rule_text}")

        if label_col:
            y_true = df_sorted[label_col].astype(int).values
            y_pred = df_sorted["Target"].astype(int).values

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0

            st.metric("Precision (of targeted)", f"{precision*100:.1f}%")
            st.metric("Recall (captured responders)", f"{recall*100:.1f}%")
            st.caption("Precision: among those you call, how many accept. Recall: of all acceptors, how many you captured.")

            st.write("Confusion Matrix (Actual vs Targeted)")
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Target 0", "Target 1"]))
        else:
            st.info("Add a `Response` column to compute precision/recall and confusion matrix.")

# ---------- Tab 2: Model Value ----------
with tab2:
    st.subheader("Model value & diagnostics")

    if not label_col:
        st.warning("To compute ROC-AUC, lift, and gains, your CSV must include `Response` (0/1).")
    else:
        y_true = df_sorted[label_col].astype(int).values
        y_score = df_sorted[prob_col].values

        # ROC-AUC
        auc = roc_auc_score(y_true, y_score)
        st.metric("ROC-AUC", f"{auc:.3f}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(fig)

        # Precision-Recall curve
        pr, rc, _ = precision_recall_curve(y_true, y_score)
        fig = plt.figure()
        plt.plot(rc, pr)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        st.pyplot(fig)

        # Lift & Gains by decile
        st.subheader("Lift & Gains (by score deciles)")
        temp = df_sorted.copy()
        temp["decile"] = pd.qcut(temp[prob_col], 10, labels=False, duplicates="drop")
        dec = (
            temp.groupby("decile")
            .agg(customers=("decile", "count"), responders=(label_col, "sum"), avg_score=(prob_col, "mean"))
            .reset_index()
        )
        dec["decile"] = dec["decile"].astype(int)
        dec = dec.sort_values("decile", ascending=False).reset_index(drop=True)

        base_rate = temp[label_col].mean()
        dec["response_rate"] = dec["responders"] / dec["customers"]
        dec["lift"] = dec["response_rate"] / base_rate if base_rate else np.nan

        dec["cum_customers"] = dec["customers"].cumsum()
        dec["cum_responders"] = dec["responders"].cumsum()
        total_resp = dec["responders"].sum()
        dec["cum_gain"] = dec["cum_responders"] / total_resp if total_resp else 0.0
        dec["cum_pop"] = dec["cum_customers"] / len(temp)

        st.dataframe(dec[["decile", "customers", "responders", "response_rate", "lift", "cum_pop", "cum_gain", "avg_score"]])

        fig = plt.figure()
        plt.plot(dec["cum_pop"], dec["cum_gain"], marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Cumulative % of customers contacted")
        plt.ylabel("Cumulative % of responders captured (gain)")
        st.pyplot(fig)

# ---------- Tab 3: Segments ----------
with tab3:
    st.subheader("Segmentation insights")

    possible_features = ["Age", "Income", "Tenure", "TotalSpend", "TotalPurchases", "Recency", "MntWines"]
    present = [c for c in possible_features if c in df_sorted.columns]

    if not present:
        st.info("Add feature columns (Age, Income, TotalSpend, etc.) to your CSV to enable segmentation insights.")
    else:
        feat = st.selectbox("Choose a feature to segment by", present)
        bins = st.slider("Number of bins", 3, 20, 8, 1)

        seg = df_sorted[[feat, prob_col] + ([label_col] if label_col else [])].copy()
        seg[feat] = pd.to_numeric(seg[feat], errors="coerce")
        seg = seg[seg[feat].notna()].copy()

        seg["bucket"] = pd.cut(seg[feat], bins=bins)
        out = seg.groupby("bucket").agg(
            customers=("bucket", "count"),
            avg_prob=(prob_col, "mean"),
            responder_rate=(label_col, "mean") if label_col else (prob_col, "mean"),
        ).reset_index()

        cA, cB = st.columns(2)
        with cA:
            st.write("Segment table")
            st.dataframe(out)

        with cB:
            st.write("Average probability by segment")
            fig = plt.figure()
            plt.plot(range(len(out)), out["avg_prob"], marker="o")
            plt.xticks(range(len(out)), [str(b) for b in out["bucket"]], rotation=45, ha="right")
            plt.ylabel("Avg predicted probability")
            plt.tight_layout()
            st.pyplot(fig)

# ---------- Tab 4: Call List ----------
with tab4:
    st.subheader("Call list (export)")

    cols_to_show = []
    if id_col:
        cols_to_show.append(id_col)
    cols_to_show += [prob_col, "Target"]
    if label_col:
        cols_to_show.append(label_col)

    for c in ["Age", "Income", "TotalSpend", "Recency", "NumWebVisitsMonth"]:
        if c in df_sorted.columns and c not in cols_to_show:
            cols_to_show.append(c)

    call_list = df_sorted[df_sorted["Target"] == 1][cols_to_show].copy()
    st.dataframe(call_list.head(200))

    csv_bytes = call_list.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download targeted customers CSV",
        data=csv_bytes,
        file_name="targeted_customers.csv",
        mime="text/csv"
    )
