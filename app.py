import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backend.pipeline import (
    DEFAULT_RF_PARAMS,
    detect_target_column,
    inspect_dataset,
    load_table,
    map_target_series,
    predict_customer,
    prepare_features,
    train_random_forest,
)

st.set_page_config(
    page_title="ChurnPulse · Random Forest Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)

CHART_FONT = dict(color="#D7E2F2", family="Outfit, sans-serif")
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=CHART_FONT,
    margin=dict(l=8, r=8, t=48, b=8),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
    autosize=True,
)
PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False, "scrollZoom": False}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@330;400;560;700&family=IBM+Plex+Mono:wght@400;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

        html, body, [class*="css"] { font-family: "Outfit", sans-serif; }
        html, body, .stApp { overflow-x: hidden; max-width: 100%; }

        .stApp {
            background:
                radial-gradient(1100px 520px at -8% -10%, rgba(62, 224, 197, 0.16), transparent 46%),
                radial-gradient(900px 480px at 112% 0%, rgba(126, 91, 255, 0.20), transparent 42%),
                linear-gradient(180deg, #070B16 0%, #0A1020 100%);
            color: #E7EDF8;
        }

        #MainMenu, footer { visibility: hidden; }
        header[data-testid="stHeader"] {
            background: transparent !important;
            visibility: visible;
        }
        .material-symbols-rounded {
            font-family: "Material Symbols Rounded", sans-serif !important;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.1rem;
            padding-bottom: 2.2rem;
            padding-left: clamp(0.75rem, 2.2vw, 2rem);
            padding-right: clamp(0.75rem, 2.2vw, 2rem);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0C1224 0%, #080C16 100%);
            border-right: 1px solid rgba(62, 224, 197, 0.14);
        }
        [data-testid="stSidebar"] * { font-family: "Outfit", sans-serif; }

        [data-testid="stFileUploader"] section {
            padding: 0 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            justify-content: center !important;
            gap: 0.35rem !important;
            min-height: 72px !important;
            padding: 0.75rem !important;
            background: rgba(62, 224, 197, 0.08) !important;
            border: 1px dashed rgba(62, 224, 197, 0.38) !important;
            border-radius: 14px !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzone"] svg,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] {
            display: none !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            width: 100% !important;
            position: relative !important;
            z-index: 2 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stFileUploaderDropzone"] button p,
        [data-testid="stFileUploaderDropzone"] button span {
            font-size: 0.9rem !important;
            white-space: nowrap !important;
        }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 0.85rem 1rem;
            align-items: stretch;
        }
        [data-testid="stHorizontalBlock"] > div {
            min-width: 0;
        }
        [data-testid="stVerticalBlock"] { gap: 0.65rem; }
        [data-testid="stDataFrame"], .stPlotlyChart, iframe {
            max-width: 100%;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: clamp(1rem, 2vw, 1.5rem);
            border-radius: 22px;
            border: 1px solid rgba(62, 224, 197, 0.18);
            background: linear-gradient(120deg, rgba(18, 24, 42, 0.92), rgba(12, 18, 34, 0.78));
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
            margin-bottom: 1.1rem;
        }
        .hero::after {
            content: "";
            position: absolute;
            width: min(280px, 50vw);
            height: min(280px, 50vw);
            right: -40px;
            top: -90px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(62, 224, 197, 0.22), transparent 68%);
            pointer-events: none;
        }
        .kicker, .hero h1, .hero p { position: relative; z-index: 1; }
        .kicker {
            font-family: "IBM Plex Mono", monospace;
            letter-spacing: 0.14em;
            font-size: clamp(0.62rem, 1.6vw, 0.72rem);
            color: #3EE0C5;
            margin-bottom: 0.35rem;
        }
        .hero h1 {
            font-size: clamp(1.35rem, 3.4vw, 2.05rem);
            line-height: 1.15;
            margin: 0 0 0.4rem 0;
            font-weight: 700;
        }
        .hero h1 span { color: #3EE0C5; }
        .hero p {
            margin: 0;
            max-width: 720px;
            color: #A9B6CA;
            font-size: clamp(0.88rem, 1.6vw, 0.98rem);
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.2rem 0 1rem;
            width: 100%;
        }
        .kpi {
            min-width: 0;
            padding: 1rem 1.05rem 0.9rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: linear-gradient(180deg, rgba(22, 29, 52, 0.88), rgba(14, 19, 36, 0.72));
        }
        .kpi .label {
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8EA0B8;
            font-family: "IBM Plex Mono", monospace;
        }
        .kpi .value {
            font-size: clamp(1.2rem, 2.4vw, 1.72rem);
            font-weight: 700;
            margin-top: 0.18rem;
            color: #F4F7FB;
            overflow-wrap: anywhere;
        }
        .kpi .hint {
            color: #8EA0B8;
            font-size: 0.78rem;
            margin-top: 0.12rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .kpi.teal .value { color: #3EE0C5; }
        .kpi.violet .value { color: #B59CFF; }
        .kpi.amber .value { color: #F3C15B; }
        .kpi.rose .value { color: #FF7AA2; }

        .panel {
            padding: 1rem 1.05rem 0.85rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(16, 22, 40, 0.62);
            margin-bottom: 0.9rem;
        }
        .risk-chip {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-family: "IBM Plex Mono", monospace;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .risk-Critical { background: rgba(255, 122, 162, 0.18); color: #FF7AA2; }
        .risk-Elevated { background: rgba(243, 193, 91, 0.16); color: #F3C15B; }
        .risk-Watch { background: rgba(181, 156, 255, 0.16); color: #B59CFF; }
        .risk-Stable { background: rgba(62, 224, 197, 0.16); color: #3EE0C5; }
        .verdict {
            padding: 1.2rem;
            border-radius: 20px;
            text-align: center;
            border: 1px solid rgba(62, 224, 197, 0.2);
            background: linear-gradient(180deg, rgba(18, 28, 48, 0.9), rgba(10, 16, 30, 0.8));
        }
        .verdict h2 { margin: 0.15rem 0 0.35rem; font-size: clamp(1.5rem, 3vw, 2rem); }
        .muted { color: #8EA0B8; }

        div[data-testid="stMetric"] {
            background: rgba(16, 22, 40, 0.55);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 0.7rem 0.8rem;
            border-radius: 14px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #3EE0C5, #6BE8C0);
            color: #071018;
            border: 0;
            font-weight: 700;
            border-radius: 12px;
        }
        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(62, 224, 197, 0.3);
            background: transparent;
            color: #3EE0C5;
            width: 100%;
        }
        iframe[height="0"] { display: none !important; height: 0 !important; }

        @media (max-width: 1100px) {
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 900px) {
            [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
            [data-testid="stHorizontalBlock"] > div {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
            section[data-testid="stSidebar"] {
                width: min(300px, 88vw) !important;
                min-width: 0 !important;
            }
            .kpi .hint { white-space: normal; }
        }
        @media (max-width: 560px) {
            .kpi-grid { grid-template-columns: 1fr; }
            .hero { border-radius: 16px; }
            .block-container { padding-top: 0.7rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "df": None,
        "source_name": None,
        "prepared": None,
        "result": None,
        "page": "Pulse",
        "file_id": None,
        "target_name": None,
        "ingest_error": None,
        "uploader_nonce": 0,
        "n_estimators": DEFAULT_RF_PARAMS["n_estimators"],
        "criterion": DEFAULT_RF_PARAMS["criterion"],
        "max_depth": DEFAULT_RF_PARAMS["max_depth"],
        "min_samples_split": DEFAULT_RF_PARAMS["min_samples_split"],
        "min_samples_leaf": DEFAULT_RF_PARAMS["min_samples_leaf"],
        "test_size": 0.20,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def kpi_card(label: str, value: str, hint: str, tone: str) -> str:
    return (
        f'<div class="kpi {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="hint">{hint}</div></div>'
    )


def plotly_defaults(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


def confusion_figure(matrix, labels=("Stay", "Churn")) -> go.Figure:
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=["#12182A", "#1C3B44", "#3EE0C5"],
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=list(labels),
        y=list(labels),
    )
    fig.update_traces(textfont=dict(size=18, color="#F4F7FB"))
    return plotly_defaults(fig, 380)


def importance_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.sort_values("Importance")
    fig = go.Figure(
        go.Bar(
            x=data["Importance"],
            y=data["Feature"],
            orientation="h",
            marker=dict(
                color=data["Importance"],
                colorscale=[[0, "#7E5BFF"], [1, "#3EE0C5"]],
                showscale=False,
            ),
        )
    )
    fig.update_layout(title="Signal strength by feature")
    return plotly_defaults(fig, max(360, 28 * len(data) + 80))


def gauge_figure(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 38, "color": "#F4F7FB"}},
            title={"text": "Churn probability", "font": {"size": 14, "color": "#8EA0B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8EA0B8"},
                "bar": {"color": "#3EE0C5"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(62, 224, 197, 0.18)"},
                    {"range": [25, 45], "color": "rgba(181, 156, 255, 0.18)"},
                    {"range": [45, 70], "color": "rgba(243, 193, 91, 0.20)"},
                    {"range": [70, 100], "color": "rgba(255, 122, 162, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#FF7AA2", "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(**CHART_LAYOUT, height=280)
    return fig


def split_figure(result) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Train", x=["Accuracy"], y=[result.train_accuracy], marker_color="#3EE0C5")
    fig.add_bar(name="Test", x=["Accuracy"], y=[result.test_accuracy], marker_color="#7E5BFF")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]), title="Train vs test accuracy")
    return plotly_defaults(fig, 300)


def cohort_figure(df: pd.DataFrame, target_name: str | None) -> go.Figure | None:
    if not target_name or target_name not in df.columns:
        return None
    mapped = df.copy()
    mapped["_target"] = map_target_series(mapped[target_name])
    cat_cols = [
        col
        for col in mapped.select_dtypes(exclude="number").columns
        if col not in (target_name,) and mapped[col].nunique(dropna=True) <= 12
    ]
    if not cat_cols:
        return None
    preferred = next((col for col in ("Contract", "Gender", "InternetService", "PaymentMethod") if col in cat_cols), cat_cols[0])
    rates = mapped.groupby(preferred)["_target"].mean().reset_index()
    rates["_target"] *= 100
    fig = px.bar(
        rates,
        x=preferred,
        y="_target",
        color="_target",
        color_continuous_scale=["#7E5BFF", "#3EE0C5"],
    )
    fig.update_layout(title=f"{target_name} rate by {preferred}", coloraxis_showscale=False, yaxis_title="Positive %")
    return plotly_defaults(fig, 340)


def numeric_target_figure(df: pd.DataFrame, target_name: str | None) -> go.Figure | None:
    if not target_name or target_name not in df.columns:
        return None
    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col != target_name and df[col].nunique(dropna=True) > 5
    ]
    if not numeric_cols:
        return None
    col = numeric_cols[0]
    plotted = df[[col, target_name]].copy()
    plotted["_class"] = map_target_series(plotted[target_name]).map({0: "Stay", 1: "Churn"})
    fig = px.histogram(
        plotted.dropna(),
        x=col,
        color="_class",
        barmode="overlay",
        opacity=0.72,
        color_discrete_map={"Stay": "#7E5BFF", "Churn": "#3EE0C5"},
    )
    fig.update_layout(title=f"{col} by {target_name}", bargap=0.08)
    return plotly_defaults(fig, 340)


def ingest_file(uploaded, file_id_prefix: str) -> None:
    file_id = f"{file_id_prefix}-{uploaded.name}-{uploaded.size}"
    if st.session_state.get("file_id") == file_id:
        return
    try:
        df = load_table(uploaded)
    except Exception as exc:
        st.session_state.ingest_error = f"Could not read that file: {exc}"
        return
    if df.empty or df.shape[1] < 2:
        st.session_state.ingest_error = "The file needs at least two columns and one row of data."
        return
    guessed = detect_target_column(df)
    st.session_state.df = df
    st.session_state.source_name = uploaded.name
    st.session_state.file_id = file_id
    st.session_state.target_name = guessed or df.columns[-1]
    st.session_state.result = None
    st.session_state.prepared = None
    st.session_state.ingest_error = None


def train_model() -> None:
    df = st.session_state.df
    target_name = st.session_state.target_name
    if df is None:
        st.warning("Upload a CSV or Excel file first.")
        return
    if not target_name or target_name not in df.columns:
        st.session_state.ingest_error = "Choose a target column from the uploaded file."
        return
    params = {
        "n_estimators": st.session_state.n_estimators,
        "criterion": st.session_state.criterion,
        "max_depth": st.session_state.max_depth,
        "min_samples_split": st.session_state.min_samples_split,
        "min_samples_leaf": st.session_state.min_samples_leaf,
        "random_state": 42,
    }
    with st.spinner("Training Random Forest on your dataset..."):
        try:
            prepared = prepare_features(df, target_name)
            result = train_random_forest(prepared, params, test_size=st.session_state.test_size)
        except Exception as exc:
            st.session_state.ingest_error = str(exc)
            st.session_state.result = None
            st.session_state.prepared = None
            return
        st.session_state.prepared = prepared
        st.session_state.result = result
        st.session_state.ingest_error = None


def auto_train_if_needed() -> None:
    if st.session_state.df is not None and st.session_state.result is None and st.session_state.target_name:
        train_model()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### ◈ ChurnPulse")
        st.caption("Random Forest retention studio")
        st.divider()
        st.markdown("**Upload dataset**")
        st.caption("CSV or Excel. Charts stay empty until you upload.")
        uploaded = st.file_uploader(
            "Choose CSV or Excel",
            type=["xlsx", "xls", "csv"],
            key=f"sidebar_upload_{st.session_state.uploader_nonce}",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            ingest_file(uploaded, "sidebar")

        if st.session_state.df is not None:
            columns = st.session_state.df.columns.tolist()
            current = st.session_state.target_name if st.session_state.target_name in columns else columns[-1]
            chosen = st.selectbox("Target column", columns, index=columns.index(current))
            if chosen != st.session_state.target_name:
                st.session_state.target_name = chosen
                st.session_state.result = None
                st.session_state.prepared = None

        if st.button("Clear dataset", width="stretch"):
            st.session_state.df = None
            st.session_state.source_name = None
            st.session_state.file_id = None
            st.session_state.target_name = None
            st.session_state.result = None
            st.session_state.prepared = None
            st.session_state.ingest_error = None
            st.session_state.uploader_nonce = int(st.session_state.uploader_nonce) + 1
            st.rerun()

        st.divider()
        st.markdown("**Forest controls**")
        st.select_slider("Trees", options=[50, 100, 150, 200, 300], key="n_estimators")
        st.selectbox("Criterion", ["gini", "entropy"], key="criterion")
        st.slider("Max depth", 3, 16, key="max_depth")
        st.slider("Min samples split", 2, 30, key="min_samples_split")
        st.slider("Min samples leaf", 1, 20, key="min_samples_leaf")
        st.slider("Test size", 0.10, 0.40, step=0.05, key="test_size")

        if st.button("Retrain Random Forest", type="primary", width="stretch"):
            st.session_state.result = None
            train_model()

        st.divider()
        pages = ["Pulse", "Observatory", "Scoreboard", "Signals", "Oracle"]
        st.radio("Studio", pages, key="page")


def page_pulse() -> None:
    df = st.session_state.df
    result = st.session_state.result
    target_name = st.session_state.target_name

    st.markdown(
        """
        <div class="hero">
            <div class="kicker">RANDOM FOREST · RETENTION INTELLIGENCE</div>
            <h1>See the customers who are about to <span>leave</span>.</h1>
            <p>Upload your own CSV or Excel file. Metrics, charts, and predictions are built from that file only — nothing is preloaded.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.ingest_error:
        st.error(st.session_state.ingest_error)

    if df is None:
        st.markdown(
            '<div class="panel"><p class="muted">No dataset yet. Choose a <b>.csv</b>, <b>.xlsx</b>, or <b>.xls</b> file below or in the sidebar. Visualizations appear after upload.</p></div>',
            unsafe_allow_html=True,
        )
        pulse_file = st.file_uploader(
            "Drop your dataset here",
            type=["xlsx", "xls", "csv"],
            key=f"pulse_upload_{st.session_state.uploader_nonce}",
        )
        if pulse_file is not None:
            ingest_file(pulse_file, "pulse")
            st.rerun()
        return

    info = inspect_dataset(df, target_name)
    rows = info["shape"][0]
    cols = info["shape"][1]
    churn_rate = info["churn_rate"]
    accuracy = result.metrics["accuracy"] if result else None

    cards = "".join(
        [
            kpi_card("Rows", f"{rows:,}", st.session_state.source_name or "uploaded file", "teal"),
            kpi_card("Columns", str(cols), f"target: {target_name or 'not set'}", "violet"),
            kpi_card("Positive rate", f"{churn_rate:.1%}" if churn_rate is not None else "—", target_name or "choose target", "rose"),
            kpi_card("Model accuracy", f"{accuracy:.1%}" if accuracy is not None else "Training…", "hold-out test set", "amber"),
        ]
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)
    st.caption(f"Using **{st.session_state.source_name}** · target **{target_name}**")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        fig = cohort_figure(df, target_name) or numeric_target_figure(df, target_name)
        if fig:
            show_chart(fig)
        else:
            st.dataframe(df.head(10), width="stretch")
    with right:
        if result:
            show_chart(split_figure(result))
            gap = result.accuracy_gap
            verdict = "Stable fit" if abs(gap) < 0.05 else "Possible overfit" if gap > 0.05 else "Test stronger than train"
            st.markdown(f"**Train–test gap:** `{gap:.4f}` · {verdict}")
        else:
            st.markdown('<div class="panel"><p class="muted">Model is not ready yet. Check the target column, then retrain.</p></div>', unsafe_allow_html=True)


def page_observatory() -> None:
    df = st.session_state.df
    if df is None:
        st.warning("Upload a CSV or Excel file to inspect it.")
        return

    info = inspect_dataset(df, st.session_state.target_name)
    st.subheader("Dataset observatory")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{info['shape'][0]:,}")
    c2.metric("Columns", info["shape"][1])
    c3.metric("Missing cells", int(df.isnull().sum().sum()))

    tabs = st.tabs(["Preview", "Schema", "Missing", "Numeric summary"])
    with tabs[0]:
        st.dataframe(info["head"], width="stretch")
    with tabs[1]:
        st.dataframe(info["dtypes"], width="stretch", hide_index=True)
        if st.session_state.prepared:
            dropped = st.session_state.prepared.dropped_columns
            st.caption(f"ID columns dropped before training: {', '.join(dropped) if dropped else 'none'}")
    with tabs[2]:
        st.dataframe(info["missing"], width="stretch")
    with tabs[3]:
        if info["summary"].empty:
            st.info("No numeric columns to summarise.")
        else:
            st.dataframe(info["summary"], width="stretch")


def page_scoreboard() -> None:
    result = st.session_state.result
    if result is None:
        st.warning("Upload a dataset so the Random Forest can train.")
        return

    m = result.metrics
    st.subheader("Random Forest results")
    cards = "".join(
        [
            kpi_card("Accuracy", f"{m['accuracy']:.4f}", f"{m['accuracy'] * 100:.2f}%", "teal"),
            kpi_card("Precision", f"{m['precision']:.4f}", "positive class = Churn", "violet"),
            kpi_card("Recall", f"{m['recall']:.4f}", "caught leavers", "amber"),
            kpi_card("F1-Score", f"{m['f1']:.4f}", "balance of both", "rose"),
        ]
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        show_chart(confusion_figure(result.confusion))
    with right:
        report_df = pd.DataFrame(result.report).T
        keep = [row for row in report_df.index if row in ("Stay", "Churn", "accuracy", "macro avg", "weighted avg")]
        st.markdown("**Classification report**")
        st.dataframe(report_df.loc[keep].round(4), width="stretch")
        st.caption(
            f"Training accuracy {result.train_accuracy:.4f} · Testing accuracy {result.test_accuracy:.4f}"
        )

    pred_frame = pd.DataFrame(
        {
            "Actual": result.y_test.map({0: "Stay", 1: "Churn"}).values,
            "Predicted": pd.Series(result.y_pred).map({0: "Stay", 1: "Churn"}).values,
            "ChurnProbability": result.y_proba.round(4),
        }
    )
    st.markdown("**Hold-out predictions**")
    st.dataframe(pred_frame.head(20), width="stretch")
    st.download_button(
        "Download test predictions CSV",
        pred_frame.to_csv(index=False).encode("utf-8"),
        file_name="churnpulse_test_predictions.csv",
        mime="text/csv",
    )


def page_signals() -> None:
    result = st.session_state.result
    df = st.session_state.df
    if result is None:
        st.warning("Upload a dataset to see feature signals.")
        return

    st.subheader("What the forest listens to")
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        show_chart(importance_figure(result.feature_importance))
    with right:
        st.dataframe(result.feature_importance, width="stretch", hide_index=True)
        top = result.feature_importance.iloc[0]
        st.markdown(f"Strongest signal: **{top['Feature']}** ({top['Importance']:.3f})")

    if df is not None:
        fig = cohort_figure(df, st.session_state.target_name) or numeric_target_figure(df, st.session_state.target_name)
        if fig:
            show_chart(fig)


def page_oracle() -> None:
    prepared = st.session_state.prepared
    result = st.session_state.result
    if prepared is None or result is None:
        st.warning("Upload a dataset before scoring a customer.")
        return

    st.subheader("Live customer oracle")
    st.caption("Adjust the profile. The same trained Random Forest scores Stay vs Churn in real time.")

    payload: dict = {}
    cols = st.columns(3)
    for idx, field in enumerate(prepared.feature_schema):
        with cols[idx % 3]:
            name = field["name"]
            if field["kind"] == "categorical":
                options = field["options"]
                default = field["default"] if field["default"] in options else options[0]
                payload[name] = st.selectbox(name, options, index=options.index(default), key=f"oracle_{name}")
            else:
                lo, hi = float(field["min"]), float(field["max"])
                if lo == hi:
                    hi = lo + 1.0
                payload[name] = st.number_input(
                    name,
                    min_value=lo,
                    max_value=hi,
                    value=float(min(max(field["median"], lo), hi)),
                    key=f"oracle_{name}",
                )

    prediction = predict_customer(result.model, payload, prepared)
    band = prediction["risk_band"]
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="verdict">
                <div class="risk-chip risk-{band}">{band} risk</div>
                <h2>{prediction['label']}</h2>
                <p class="muted">Stay {prediction['stay_probability']:.1%} · Leave {prediction['churn_probability']:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        show_chart(gauge_figure(prediction["churn_probability"]))
    with right:
        st.markdown("**Drivers the forest weighted most**")
        show_chart(importance_figure(prediction["contributions"][["Feature", "Importance"]].head(10)))

    playbook = {
        "Critical": "Offer a retention call in 24 hours, a contract upgrade, and a support credit.",
        "Elevated": "Trigger an in-app save offer and assign a success manager.",
        "Watch": "Send usage tips and monitor ticket spikes.",
        "Stable": "Keep nurture cadence. No intervention needed.",
    }
    st.success(playbook[band])


def collapse_sidebar_on_phone() -> None:
    components.html(
        """
        <script>
        (function () {
          const win = window.parent;
          if (!win || win.innerWidth >= 900) return;
          if (win.sessionStorage.getItem("churnpulse_sidebar") === "1") return;
          win.sessionStorage.setItem("churnpulse_sidebar", "1");
          const doc = win.document;
          const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
          if (btn) btn.click();
        })();
        </script>
        """,
        height=0,
    )


def main() -> None:
    inject_css()
    collapse_sidebar_on_phone()
    init_state()
    sidebar()
    auto_train_if_needed()
    page = st.session_state.page
    if page == "Pulse":
        page_pulse()
    elif page == "Observatory":
        page_observatory()
    elif page == "Scoreboard":
        page_scoreboard()
    elif page == "Signals":
        page_signals()
    else:
        page_oracle()


if __name__ == "__main__":
    main()
