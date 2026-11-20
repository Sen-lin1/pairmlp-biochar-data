# -*- coding: utf-8 -*-
"""Bilingual Streamlit GUI for the final 5859-parameter Pair-MLP model.

The interface follows the layout and interaction pattern of the supplied BFRC
GUI while preserving the Pair-MLP data, model, validation, single-prediction,
and batch-prediction logic.  All components use one pastel colour system.
"""

from datetime import datetime
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ==========================================
# 0. File paths and model contract
# ==========================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# The deployment package is self-contained: GUI, model, data and the compact
# unpickling compatibility module all live in this directory.
MODEL_PATH = BASE_DIR / "selected_model.pkl"
DATA_PATH = BASE_DIR / "test-data.csv"

import optimize_pair_v2  # noqa: E402,F401  Required when unpickling the model.
from optimize_pair_v2 import BASE_FEATURES  # noqa: E402

MODEL_PARAMETERS = 5859

PRIMARY_OPERATING = [
    "Adsorption time",
    "Initial concentration",
    "Adsorbent dosage",
    "Solution pH",
]

ADDITIONAL_OPERATING = [
    "RPM",
    "Volume",
    "Adsorption temperature",
    "Ion concentration",
    "Humic acid",
]

MATERIAL_FEATURES = BASE_FEATURES[:15]


# ==========================================
# 1. Bilingual text
# ==========================================
TRANSLATIONS = {
    "cn": {
        "page_title": "Pair-MLP 生物炭吸附预测系统",
        "main_title": "Pair-MLP 生物炭吸附预测系统",
        "sub_title": (
            "基于含 **5859 个参数**的最终 Pair-MLP，在已记录材料–污染物体系的支持域内，"
            "预测指定接触时间的吸附量。"
        ),
        "profile": "材料与污染物",
        "defaults": "默认值来自该组合的一条训练记录。预测前请核对全部设置。",
        "sec1_title": "### 1. 主要操作条件",
        "sec2_title": "### 2. 其他操作条件",
        "sec3_title": "### 3. 材料与体系信息",
        "water_type": "水体类型",
        "adsorption_mode": "吸附模式",
        "material_profile": "已记录材料性质",
        "material_input": "材料输入",
        "recorded_value": "记录值",
        "predict": "开始预测",
        "predicted": "预测吸附量（给定接触时间）",
        "upper": "质量守恒上限",
        "outside": "下列输入超出该组合的记录范围：",
        "estimate_note": "该结果为模型估计值，不是实测值或不确定性区间。",
        "download": "导出本次预测",
        "input_summary": "查看本次完整输入",
        "feature_order": "查看模型输入顺序",
        "order": "序号",
        "feature": "特征",
        "batch": "批量 CSV 预测",
        "batch_help": "请提供模板所示的全部 27 项输入；无需最终浓度和实测吸附量。",
        "template": "下载输入模板",
        "upload": "上传输入 CSV",
        "batch_download": "下载批量预测结果",
        "missing": "缺少输入列：",
        "load_fail": "模型或数据加载失败：",
        "prediction_fail": "预测失败：",
        "batch_fail": "批量文件无法处理：",
        "contact_time": "接触时间 (min)",
        "initial_concentration": "初始浓度 (mg/L)",
        "dosage": "投加量 (g/L)",
        "solution_ph": "溶液 pH",
        "rpm": "转速 (rpm)",
        "volume": "溶液体积 (L)",
        "temperature": "吸附温度 (°C)",
        "ions": "离子浓度 (mol/L)",
        "humic": "腐殖酸 (mg/L)",
    },
    "en": {
        "page_title": "Pair-MLP biochar adsorption predictor",
        "main_title": "Pair-MLP biochar adsorption predictor",
        "sub_title": (
            "Prediction of adsorption capacity at a specified contact time with the final "
            "**5859-parameter Pair-MLP**, within the support of recorded material–pollutant systems."
        ),
        "profile": "Material and pollutant",
        "defaults": (
            "Defaults come from one recorded training condition. "
            "Confirm all settings before prediction."
        ),
        "sec1_title": "### 1. Primary operating conditions",
        "sec2_title": "### 2. Additional operating conditions",
        "sec3_title": "### 3. Material and system information",
        "water_type": "Water type",
        "adsorption_mode": "Adsorption mode",
        "material_profile": "Recorded material properties",
        "material_input": "Material input",
        "recorded_value": "Recorded value",
        "predict": "Run prediction",
        "predicted": "Predicted amount at recorded time",
        "upper": "Mass-balance upper bound",
        "outside": "Outside the recorded range for this profile: ",
        "estimate_note": "This is a model estimate, not a measured result or uncertainty interval.",
        "download": "Download prediction",
        "input_summary": "View complete input",
        "feature_order": "View model input order",
        "order": "Order",
        "feature": "Feature",
        "batch": "Batch prediction from CSV",
        "batch_help": (
            "Provide all 27 input columns shown in the template. "
            "Final concentration and measured capacity are not required."
        ),
        "template": "Download input template",
        "upload": "Upload input CSV",
        "batch_download": "Download batch results",
        "missing": "Missing inputs: ",
        "load_fail": "Model or dataset loading failed: ",
        "prediction_fail": "Prediction failed: ",
        "batch_fail": "Input could not be processed: ",
        "contact_time": "Contact time (min)",
        "initial_concentration": "Initial concentration (mg/L)",
        "dosage": "Dosage (g/L)",
        "solution_ph": "Solution pH",
        "rpm": "Rotation speed (rpm)",
        "volume": "Solution volume (L)",
        "temperature": "Adsorption temperature (°C)",
        "ions": "Ion concentration (mol/L)",
        "humic": "Humic acid (mg/L)",
    },
}

FEATURE_LABEL_KEYS = {
    "Adsorption time": "contact_time",
    "Initial concentration": "initial_concentration",
    "Adsorbent dosage": "dosage",
    "Solution pH": "solution_ph",
    "RPM": "rpm",
    "Volume": "volume",
    "Adsorption temperature": "temperature",
    "Ion concentration": "ions",
    "Humic acid": "humic",
}

MATERIAL_LABELS_CN = {
    "Adsorbent": "吸附剂",
    "Pollutant": "污染物",
    "Pyrolysis temperature": "热解温度",
    "Pyrolysis time": "热解时间",
    "Ash": "灰分",
    "Surface area": "比表面积",
    "Pore volume": "孔容",
    "Average pore size": "平均孔径",
}

INPUT_CONFIG = {
    "Adsorption time": {"min_value": 0.0, "step": 1.0, "format": "%.6f"},
    "Initial concentration": {"min_value": 0.0, "step": 0.1, "format": "%.6f"},
    "Adsorbent dosage": {"min_value": 0.000001, "step": 0.01, "format": "%.6f"},
    "Solution pH": {"min_value": 0.0, "step": 0.1, "format": "%.6f"},
    "RPM": {"min_value": 0.0, "step": 1.0, "format": "%.6f"},
    "Volume": {"min_value": 0.0, "step": 0.001, "format": "%.6f"},
    "Adsorption temperature": {"min_value": 0.0, "step": 1.0, "format": "%.6f"},
    "Ion concentration": {"min_value": 0.0, "step": 0.01, "format": "%.6f"},
    "Humic acid": {"min_value": 0.0, "step": 0.1, "format": "%.6f"},
}


# ==========================================
# 2. Page configuration
# ==========================================
st.set_page_config(
    page_title="Pair-MLP biochar adsorption predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==========================================
# 3. One consistent visual system
# ==========================================
st.markdown(
    """
<style>
:root {
    --salmon: #DF9E9B;
    --blue: #99BADF;
    --green: #D8E7CA;
    --teal: #99CDCE;
    --purple: #999ACD;
    --pink: #FFD0E9;
    --ink: #28333D;
    --muted: #66727C;
    --paper: #FBFCFD;
    --title: #52659B;
    --heading: #4E6576;
    --border: #C9D2DA;
}

html, body, .stApp, .stMarkdown, .stText, .stCaption, label, p, input,
textarea, button, select, [data-testid="stWidgetLabel"],
[data-testid="stMarkdownContainer"], [data-testid="stDataFrame"],
[data-testid="stTable"], [data-testid="stRadio"] label,
[data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label,
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    font-family: Arial, "Microsoft YaHei", "SimHei", sans-serif !important;
    font-weight: 700 !important;
}

span[data-testid="stIconMaterial"], [data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"], [data-testid="stExpanderToggleIcon"] *,
.material-icons, .material-icons-outlined, .material-icons-round,
.material-symbols-outlined, .material-symbols-rounded, .material-symbols-sharp {
    font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 1.25rem !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    direction: ltr !important;
    -webkit-font-feature-settings: "liga" !important;
    -webkit-font-smoothing: antialiased !important;
    font-feature-settings: "liga" !important;
}

.stApp { background: var(--paper); color: var(--ink); }
header[data-testid="stHeader"] {
    border-top: 4px solid transparent;
    border-image: linear-gradient(90deg, var(--salmon), var(--blue), var(--green),
        var(--teal), var(--purple), var(--pink)) 1;
}

.block-container, [data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1120px !important;
    padding: 1rem 1.5rem 2rem !important;
}

h1 {
    color: var(--title) !important;
    font-size: 1.55rem !important;
    font-weight: 900 !important;
    line-height: 1.35 !important;
}

.stMarkdown h3 {
    color: var(--heading) !important;
    border-bottom: 2px solid var(--teal);
    padding-bottom: .3rem;
    margin: 1rem 0 .5rem;
    font-size: 1.18rem !important;
    font-weight: 800 !important;
}

[data-testid="stForm"] {
    width: 100% !important;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    background: #FFFFFF;
}

[data-testid="stSelectbox"], [data-testid="stNumberInput"] { width: 100% !important; }
[data-testid="stNumberInput"] input { font-family: Arial, sans-serif !important; }
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"],
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p,
[data-testid="stNumberInput"] label, [data-testid="stNumberInput"] label p {
    white-space: nowrap !important;
    word-break: keep-all !important;
    overflow-wrap: normal !important;
    font-size: .88rem !important;
    line-height: 1.25 !important;
}
[data-testid="stHorizontalBlock"] > div { min-width: 0 !important; }

[data-testid="stAlert"] {
    background: var(--green) !important;
    border-left: 4px solid var(--teal) !important;
    color: var(--ink) !important;
}
[data-testid="stAlert"] p { color: var(--ink) !important; }

.stButton > button, [data-testid="stFormSubmitButton"] > button {
    width: 100%;
    margin-top: 1rem;
    padding: .65rem 2rem;
    border-radius: 8px;
    background: var(--purple) !important;
    border-color: var(--purple) !important;
    color: #17232C !important;
    font-size: 1.05rem;
    font-weight: 800 !important;
    transition: all .25s;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover,
.stDownloadButton > button:hover {
    background: var(--blue) !important;
    border-color: var(--blue) !important;
    color: #17232C !important;
}
.stButton > button:focus, .stDownloadButton > button:focus,
[data-testid="stFormSubmitButton"] > button:focus {
    outline: none !important;
    box-shadow: none !important;
}

.stDownloadButton > button {
    width: 100%;
    margin-top: .8rem;
    border-radius: 8px;
    background: var(--teal) !important;
    border-color: var(--teal) !important;
    color: #17232C !important;
    font-weight: 800 !important;
}

.result-box {
    margin: 1.5rem 0 .5rem;
    padding: 1.35rem;
    border: 1px solid var(--teal);
    border-radius: 10px;
    background: var(--green);
    text-align: center;
}
.result-box.upper { background: #EEF3FA; border-color: var(--blue); }
.result-label { margin-bottom: .4rem; color: #4E5B65; font-size: 1.02rem; }
.result-value { color: var(--title); font-size: 2.05rem; font-weight: 900 !important; line-height: 1.25; }
.result-unit { margin-left: .15rem; color: var(--muted); font-size: 1rem; }

div[data-testid="stRadio"] { min-width: 190px !important; }
div[data-testid="stRadio"] > label {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    visibility: hidden !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-flow: row nowrap !important;
    justify-content: flex-end !important;
    align-items: center !important;
    gap: .75rem !important;
    width: 100% !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    display: inline-flex !important;
    width: auto !important;
    min-width: max-content !important;
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label p {
    margin: 0 !important;
    white-space: nowrap !important;
    line-height: 1.2 !important;
}

[data-testid="stExpander"] summary { min-height: 2.2rem !important; align-items: center !important; }
[data-testid="stExpander"] summary p { margin: 0 !important; color: var(--ink) !important; }
[data-testid="stDataFrame"] { background: #FFFFFF; }

@media screen and (max-width: 900px) {
    [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p,
    [data-testid="stNumberInput"] label p { font-size: .80rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 4. Data and prediction helpers
# ==========================================
@st.cache_resource(show_spinner=False)
def resources():
    """Load the training data and final Pair-MLP, then verify their contract."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.name}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Predictor not found: {MODEL_PATH.name}")

    data = pd.read_csv(DATA_PATH)
    missing = sorted(set(BASE_FEATURES + ["partition"]) - set(data.columns))
    if missing:
        raise ValueError("Dataset columns missing: " + ", ".join(missing))

    model = pickle.loads(MODEL_PATH.read_bytes())
    if int(getattr(model, "parameters", -1)) != MODEL_PARAMETERS:
        raise RuntimeError("Unexpected Pair-MLP artifact")
    return data, model


def make_input(label: str, default_value: float, feature_name: str, key_prefix: str) -> float:
    """Create one consistently configured numeric field."""
    cfg = INPUT_CONFIG[feature_name]
    return st.number_input(
        label=label,
        min_value=float(cfg["min_value"]),
        value=float(default_value),
        step=float(cfg["step"]),
        format=cfg["format"],
        key=f"{key_prefix}::{feature_name}",
    )


def finite_prediction(predictor, frame: pd.DataFrame) -> np.ndarray:
    """Predict and reject incomplete or non-finite model output."""
    prediction = np.asarray(
        predictor.predict(frame[BASE_FEATURES]), dtype=float
    ).reshape(-1)
    if prediction.size != len(frame) or not np.isfinite(prediction).all():
        raise ValueError("The predictor returned a non-finite or incomplete result.")
    return prediction


def material_table(row: pd.Series, language: str, t: dict[str, str]) -> pd.DataFrame:
    """Build the read-only material-properties table."""
    labels = [
        MATERIAL_LABELS_CN.get(feature, feature) if language == "cn" else feature
        for feature in MATERIAL_FEATURES
    ]
    return pd.DataFrame(
        {
            t["material_input"]: labels,
            t["recorded_value"]: [str(row[feature]) for feature in MATERIAL_FEATURES],
        }
    )


def result_box(label: str, value: float, extra_class: str = "") -> str:
    """Render one result card."""
    return (
        f'<div class="result-box {extra_class}">'
        f'<div class="result-label">{label}</div>'
        f'<div class="result-value">{value:.1f} '
        f'<span class="result-unit">mg/g</span></div></div>'
    )


# ==========================================
# 5. Language selector and heading
# ==========================================
_, language_col = st.columns([2.2, 1.8])
with language_col:
    language_option = st.radio(
        "Language",
        ["中文", "English"],
        index=0,
        horizontal=True,
        key="language_switch",
        label_visibility="collapsed",
    )

language = "cn" if language_option == "中文" else "en"
t = TRANSLATIONS[language]

st.title(t["main_title"])


# ==========================================
# 6. Load data and model
# ==========================================
try:
    data, model = resources()
except Exception as error:
    st.error(f"{t['load_fail']}{error}")
    st.stop()

train = data[data["partition"].eq("train")].copy()
if train.empty:
    st.error(f"{t['load_fail']}No training rows were found in the dataset.")
    st.stop()

profiles = (
    train[["Adsorbent", "Pollutant"]]
    .drop_duplicates()
    .sort_values(["Adsorbent", "Pollutant"])
)
options = [
    f"{adsorbent} | {pollutant}"
    for adsorbent, pollutant in profiles.itertuples(index=False, name=None)
]


# ==========================================
# 7. Fixed supported system for the single-record prototype
# ==========================================
material, pollutant = "AMCB", "IBU"
choice = f"{material} | {pollutant}"
pool = train[
    train["Adsorbent"].eq(material) & train["Pollutant"].eq(pollutant)
]
positive = pool[
    pool["Adsorption time"].gt(0) & pool["Initial concentration"].gt(0)
]
row = (positive if not positive.empty else pool).iloc[0][BASE_FEATURES].copy()
st.caption(
    "Fixed supported system: AMCB–IBU | known-pair interpolation only"
    if language == "en"
    else "固定支持体系：AMCB–IBU｜仅用于已知组合内插值"
)


# ==========================================
# 8. Input form
# ==========================================
with st.form("prediction_form"):
    values: dict[str, float] = {}

    st.markdown(t["sec1_title"])
    primary_columns = st.columns([1.00, 1.25, 1.00, 0.85], gap="medium")
    for column, feature in zip(primary_columns, PRIMARY_OPERATING):
        with column:
            values[feature] = make_input(
                t[FEATURE_LABEL_KEYS[feature]], float(row[feature]), feature, choice
            )

    st.markdown(t["sec2_title"])
    first_additional_row = st.columns([0.85, 0.95, 1.25], gap="medium")
    for column, feature in zip(first_additional_row, ADDITIONAL_OPERATING[:3]):
        with column:
            values[feature] = make_input(
                t[FEATURE_LABEL_KEYS[feature]], float(row[feature]), feature, choice
            )

    second_additional_row = st.columns(3, gap="medium")
    for column, feature in zip(second_additional_row, ADDITIONAL_OPERATING[3:]):
        with column:
            values[feature] = make_input(
                t[FEATURE_LABEL_KEYS[feature]], float(row[feature]), feature, choice
            )
    with second_additional_row[2]:
        st.empty()

    submit_btn = st.form_submit_button(t["predict"], type="primary")


# ==========================================
# 9. Single prediction and export
# ==========================================
if submit_btn:
    try:
        prediction_row = row.copy()
        for feature, value in values.items():
            prediction_row[feature] = value

        frame = pd.DataFrame([prediction_row])[BASE_FEATURES]
        prediction = float(finite_prediction(model, frame)[0])
        upper = float(
            frame["Initial concentration"].iloc[0]
            / frame["Adsorbent dosage"].iloc[0]
        )

        result_col, upper_col = st.columns(2, gap="medium")
        with result_col:
            st.markdown(
                result_box(t["predicted"], prediction), unsafe_allow_html=True
            )
        with upper_col:
            st.markdown(
                result_box(t["upper"], upper, "upper"), unsafe_allow_html=True
            )

        outside = [
            feature
            for feature in values
            if float(frame[feature].iloc[0]) < float(pool[feature].min())
            or float(frame[feature].iloc[0]) > float(pool[feature].max())
        ]
        if outside:
            shown = [
                t.get(FEATURE_LABEL_KEYS.get(feature, ""), feature)
                for feature in outside
            ]
            st.warning(t["outside"] + ", ".join(shown))

        st.caption(t["estimate_note"])

        export = frame.copy()
        export["predicted_q_mg_g"] = prediction
        export["mass_balance_upper_bound_mg_g"] = upper
        export["prediction_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        export["predictor_file"] = MODEL_PATH.name

        st.download_button(
            label=t["download"],
            data=export.to_csv(index=False).encode("utf-8-sig"),
            file_name="pair_mlp_adsorption_prediction.csv",
            mime="text/csv",
        )

        with st.expander(t["input_summary"]):
            st.dataframe(export, hide_index=True, use_container_width=True)

        with st.expander(t["feature_order"]):
            order_table = pd.DataFrame(
                {
                    t["order"]: range(1, len(BASE_FEATURES) + 1),
                    t["feature"]: BASE_FEATURES,
                }
            )
            st.dataframe(order_table, hide_index=True, use_container_width=True)

    except Exception as error:
        st.error(f"{t['prediction_fail']}{error}")


# ==========================================
# 10. Batch CSV prediction
# ==========================================
with st.expander(t["batch"]):
    st.write(t["batch_help"])

    st.download_button(
        label=t["template"],
        data=train[BASE_FEATURES].head(5).to_csv(index=False).encode("utf-8-sig"),
        file_name="pair_mlp_adsorption_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(t["upload"], type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = sorted(set(BASE_FEATURES) - set(batch.columns))
            if missing:
                st.error(t["missing"] + ", ".join(missing))
            else:
                batch = batch.copy()
                batch["predicted_q_mg_g"] = finite_prediction(model, batch)
                st.dataframe(batch, hide_index=True, use_container_width=True)
                st.download_button(
                    label=t["batch_download"],
                    data=batch.to_csv(index=False).encode("utf-8-sig"),
                    file_name="pair_mlp_adsorption_batch_predictions.csv",
                    mime="text/csv",
                )
        except Exception as error:
            st.error(f"{t['batch_fail']}{error}")
