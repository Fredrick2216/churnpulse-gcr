import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backend.pipeline import (
    DEFAULT_RF_PARAMS,
    generate_demo_dataset,
    inspect_dataset,
    load_table,
    predict_customer,
    prepare_features,
    train_random_forest,
)

st.set_page_config(
    page_title="ChurnPulse · Random Forest Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHART_FONT = dict(color="#D7E2F2", family="Outfit, sans-serif")
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=CHART_FONT,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@330;400;560;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] { font-family: "Outfit", sans-serif; }

        .stApp {
            background:
                radial-gradient(1100px 520px at -8% -10%, rgba(62, 224, 197, 0.16), transparent 46%),
                radial-gradient(900px 480px at 112% 0%, rgba(126, 91, 255, 0.20), transparent 42%),
                linear-gradient(180deg, #070B16 0%, #0A1020 100%);
            color: #E7EDF8;
        }

        #MainMenu, footer, header { visibility: hidden; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0C1224 0%, #080C16 100%);
            border-right: 1px solid rgba(62, 224, 197, 0.14);
        }

        [data-testid="stSidebar"] * { font-family: "Outfit", sans-serif; }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.35rem 1.5rem 1.2rem;
            border-radius: 22px;
            border: 1px solid rgba(62, 224, 197, 0.18);
            background:
                linear-gradient(120deg, rgba(18, 24, 42, 0.92), rgba(12, 18, 34, 0.78));
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
            margin-bottom: 1.1rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            right: -40px;
            top: -90px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(62, 224, 197, 0.22), transparent 68%);
        }

        .kicker {
            font-family: "IBM Plex Mono", monospace;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            color: #3EE0C5;
            margin-bottom: 0.35rem;
        }

        .hero h1 {
            font-size: 2.05rem;
            line-height: 1.1;
            margin: 0 0 0.35rem 0;
            font-weight: 700;
        }

        .hero h1 span { color: #3EE0C5; }

        .hero p {
            margin: 0;
            max-width: 720px;
            color: #A9B6CA;
            font-size: 0.98rem;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.85rem;
            margin: 0.2rem 0 1rem;
        }

        .kpi {
            padding: 1rem 1.05rem 0.9rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: linear-gradient(180deg, rgba(22, 29, 52, 0.88), rgba(14, 19, 36, 0.72));
        }

        .kpi .label {
            font-size: 0.74rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8EA0B8;
            font-family: "IBM Plex Mono", monospace;
        }

        .kpi .value {
            font-size: 1.72rem;
            font-weight: 700;
            margin-top: 0.18rem;
            color: #F4F7FB;
        }

        .kpi .hint { color: #8EA0B8; font-size: 0.8rem; margin-top: 0.12rem; }
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

        .verdict h2 { margin: 0.15rem 0 0.35rem; font-size: 2rem; }
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
        }

        [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

        @media (max-width: 1100px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Booting the live Random Forest demo...")
def boot_demo():
    df = generate_demo_dataset()
    prepared = prepare_features(df, "Churn")
    result = train_random_forest(prepared, DEFAULT_RF_PARAMS, test_size=0.20)
    return df, prepared, result


def ensure_live_demo() -> None:
    if st.session_state.df is None:
        df, prepared, result = boot_demo()
        st.session_state.df = df
        st.session_state.prepared = prepared
        st.session_state.result = result
        st.session_state.source_name = "demo_churn.xlsx"
        st.session_state.file_id = "demo"


def init_state() -> None:
    defaults = {
        "df": None,
        "source_name": None,
        "prepared": None,
        "result": None,
        "page": "Pulse",
        "file_id": None,
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


def cohort_figure(df: pd.DataFrame) -> go.Figure | None:
    if "Churn" not in df.columns:
        return None
    mapped = df.copy()
    mapped["_churn"] = (
        mapped["Churn"].astype(str).str.lower().isin(["yes", "y", "1", "true", "churn"])
        | pd.to_numeric(mapped["Churn"], errors="coerce").fillna(0).eq(1)
    ).astype(int)
    cat_cols = [c for c in mapped.select_dtypes(exclude="number").columns if c not in ("Churn", "CustomerID")]
    if not cat_cols:
        return None
    col = "Contract" if "Contract" in cat_cols else cat_cols[0]
    rates = mapped.groupby(col)["_churn"].mean().reset_index()
    rates["_churn"] *= 100
    fig = px.bar(rates, x=col, y="_churn", color="_churn", color_continuous_scale=["#7E5BFF", "#3EE0C5"])
    fig.update_layout(title=f"Churn rate by {col}", coloraxis_showscale=False, yaxis_title="Churn %")
    return plotly_defaults(fig, 340)


def train_model() -> None:
    df = st.session_state.df
    if df is None:
        st.warning("Load a dataset first.")
        return
    params = {
        "n_estimators": st.session_state.n_estimators,
        "criterion": st.session_state.criterion,
        "max_depth": st.session_state.max_depth,
        "min_samples_split": st.session_state.min_samples_split,
        "min_samples_leaf": st.session_state.min_samples_leaf,
        "random_state": 42,
    }
    with st.spinner("Forging the Random Forest..."):
        prepared = prepare_features(df, "Churn")
        result = train_random_forest(prepared, params, test_size=st.session_state.test_size)
        st.session_state.prepared = prepared
        st.session_state.result = result


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### ◈ ChurnPulse")
        st.caption("Random Forest retention studio")
        st.divider()

        uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
        if uploaded is not None:
            file_id = f"{uploaded.name}-{uploaded.size}"
            if st.session_state.get("file_id") != file_id:
                st.session_state.df = load_table(uploaded)
                st.session_state.source_name = uploaded.name
                st.session_state.file_id = file_id
                st.session_state.result = None
                st.session_state.prepared = None

        if st.button("Load demo telecom set", width="stretch"):
            st.session_state.df = generate_demo_dataset()
            st.session_state.source_name = "demo_churn.xlsx"
            st.session_state.file_id = "demo"
            st.session_state.result = None
            st.session_state.prepared = None

        st.divider()
        st.markdown("**Forest controls**")
        st.select_slider("Trees", options=[50, 100, 150, 200, 300], key="n_estimators")
        st.selectbox("Criterion", ["gini", "entropy"], key="criterion")
        st.slider("Max depth", 3, 16, key="max_depth")
        st.slider("Min samples split", 2, 30, key="min_samples_split")
        st.slider("Min samples leaf", 1, 20, key="min_samples_leaf")
        st.slider("Test size", 0.10, 0.40, step=0.05, key="test_size")

        if st.button("Train Random Forest", type="primary", width="stretch"):
            train_model()

        st.divider()
        pages = ["Pulse", "Observatory", "Scoreboard", "Signals", "Oracle"]
        st.radio("Studio", pages, key="page")


def page_pulse() -> None:
    df = st.session_state.df
    result = st.session_state.result
    rows = 0 if df is None else len(df)
    cols = 0 if df is None else df.shape[1]
    churn_rate = inspect_dataset(df)["churn_rate"] if df is not None else None
    accuracy = result.metrics["accuracy"] if result else None

    st.markdown(
        """
        <div class="hero">
            <div class="kicker">RANDOM FOREST · RETENTION INTELLIGENCE</div>
            <h1>See the customers who are about to <span>leave</span>.</h1>
            <p>Upload the same Excel workbook you used in Colab, train the forest with your original hyperparameters, then inspect metrics, feature signals, and live risk scores in one studio.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = "".join(
        [
            kpi_card("Customers", f"{rows:,}" if df is not None else "—", st.session_state.source_name or "No file loaded", "teal"),
            kpi_card("Features", str(cols) if df is not None else "—", "including target column", "violet"),
            kpi_card("Churn rate", f"{churn_rate:.1%}" if churn_rate is not None else "—", "share already gone", "rose"),
            kpi_card("Model accuracy", f"{accuracy:.1%}" if accuracy is not None else "Idle", "hold-out test set", "amber"),
        ]
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        if df is not None:
            fig = cohort_figure(df)
            if fig:
                st.plotly_chart(fig, width="stretch")
            else:
                st.dataframe(df.head(10), width="stretch")
        else:
            st.info("Load your Excel file in the sidebar, or generate the demo telecom set to explore the studio.")
    with right:
        if result:
            st.plotly_chart(split_figure(result), width="stretch")
            gap = result.accuracy_gap
            verdict = "Stable fit" if abs(gap) < 0.05 else "Possible overfit" if gap > 0.05 else "Test stronger than train"
            st.markdown(f"**Train–test gap:** `{gap:.4f}` · {verdict}")
        else:
            st.markdown('<div class="panel"><p class="muted">The forest is waiting. Set depth, trees, and split, then train.</p></div>', unsafe_allow_html=True)


def page_observatory() -> None:
    df = st.session_state.df
    if df is None:
        st.warning("Load a dataset to inspect it.")
        return

    info = inspect_dataset(df)
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
        st.warning("Train the Random Forest to unlock the scoreboard.")
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
        st.plotly_chart(confusion_figure(result.confusion), width="stretch")
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
        st.warning("Train the model to see feature signals.")
        return

    st.subheader("What the forest listens to")
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.plotly_chart(importance_figure(result.feature_importance), width="stretch")
    with right:
        st.dataframe(result.feature_importance, width="stretch", hide_index=True)
        top = result.feature_importance.iloc[0]
        st.markdown(f"Strongest signal: **{top['Feature']}** ({top['Importance']:.3f})")

    if df is not None:
        fig = cohort_figure(df)
        if fig:
            st.plotly_chart(fig, width="stretch")


def page_oracle() -> None:
    prepared = st.session_state.prepared
    result = st.session_state.result
    if prepared is None or result is None:
        st.warning("Train the forest before scoring a customer.")
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
        st.plotly_chart(gauge_figure(prediction["churn_probability"]), width="stretch")
    with right:
        st.markdown("**Drivers the forest weighted most**")
        st.plotly_chart(importance_figure(prediction["contributions"][["Feature", "Importance"]].head(10)), width="stretch")

    playbook = {
        "Critical": "Offer a retention call in 24 hours, a contract upgrade, and a support credit.",
        "Elevated": "Trigger an in-app save offer and assign a success manager.",
        "Watch": "Send usage tips and monitor ticket spikes.",
        "Stable": "Keep nurture cadence. No intervention needed.",
    }
    st.success(playbook[band])


def main() -> None:
    inject_css()
    init_state()
    sidebar()
    ensure_live_demo()
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
