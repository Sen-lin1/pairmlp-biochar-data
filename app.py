# -*- coding: utf-8 -*-
"""Bilingual Pair-MLP adsorption prediction app."""

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
MODEL_PATH = BASE_DIR / "Pair-MLP.pkl"
DATA_PATH = BASE_DIR / "test-data.csv"

import optimize_pair_v2  # noqa: E402,F401  Required when unpickling the model.
from optimize_pair_v2 import BASE_FEATURES  # noqa: E402

MODEL_PARAMETERS = 5859

OPERATING_GROUPS = (
    ("🧪", "Adsorption inputs", "吸附条件", ("Adsorption time", "Initial concentration", "Adsorbent dosage")),
    ("💧", "Water chemistry", "水化学条件", ("Solution pH", "Ion concentration", "Humic acid")),
    ("⚙️", "Experimental settings", "实验设置", ("RPM", "Volume", "Adsorption temperature")),
)


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
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================
# 3. One consistent visual system
# ==========================================
st.markdown('\n<style>\n:root { --paper: #edf6fc; --ink: #172b36; --muted: #506d7f; }\n.stApp, [data-testid="stHeader"] { background: var(--paper); color: var(--ink); }\n.stApp :is(h1,h2,h3,p,label,input,button,summary) { font-family: Arial, "Microsoft YaHei", sans-serif; }\n[data-testid="stMainBlockContainer"] { max-width: 1260px; padding-top: 1.75rem; }\n.st-key-language_switch [role="radiogroup"] { justify-content: end; }\n.stApp h1 { text-align: center; color: var(--ink); font-size: clamp(1.9rem,3.5vw,3.2rem); }\n.intro, .system { text-align: center; color: var(--muted); }\n.intro { font-size: 1.25rem; margin: 0; }\n.system { margin: 0 0 .65rem; }\n.system strong { color: #26796d; }\n.stApp h3 { color: var(--ink); font-size: clamp(1.04rem,1.7vw,1.48rem); margin-bottom: .55rem; }\n[data-testid="stForm"] { background: transparent; border: 0; padding: 0; }\n[data-testid="stNumberInput"] [data-baseweb="input"] { background: #f1f2f5; border-color: #dce7ed; }\n[data-testid="stNumberInput"] button { background: #f1f2f5; color: #294958; }\n[data-testid="stFormSubmitButton"] > button { background: #30a79d; border-color: #278c85; color: #092e29; font-weight: 700; margin-top: .35rem; }\n[data-testid="stFormSubmitButton"] > button:hover { background: #247a73; color: white; }\n.result-box { background: #e0f0d8; border: 1px solid #a9cd9c; border-radius: 10px; padding: 1.1rem 1.4rem; display: flex; align-items: center; justify-content: space-between; gap: .8rem; flex-wrap: wrap; }\n.result-box.upper { background: #e1f0fa; border-color: #abcfe4; }\n.result-value { font-size: 2.35rem; font-weight: 750; line-height: 1.15; color: #386b30; }\n.result-box.upper .result-value { color: #316786; }\n.result-unit { font-size: 1rem; font-weight: 400; }\n[data-testid="stCaptionContainer"] { color: var(--muted); }\n</style>\n', unsafe_allow_html=True)

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
        index=1,
        horizontal=True,
        key="language_switch",
        label_visibility="collapsed",
    )

language = "cn" if language_option == "中文" else "en"
t = TRANSLATIONS[language]

st.title("💧 Pair-MLP 生物炭吸附预测" if language == "cn" else "💧 Pair-MLP Biochar Adsorption Prediction", anchor=False)
intro = "根据实验条件预测指定接触时间的吸附量（mg/g）" if language == "cn" else "Predict uptake (mg/g) at a specified contact time from experimental conditions"
st.markdown(f'<p class="intro">{intro}</p>', unsafe_allow_html=True)


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
system = "固定支持体系" if language == "cn" else "Fixed supported system"
scope = "仅用于已知组合内插值" if language == "cn" else "Known-pair interpolation only"
st.markdown(f'<p class="system">{system} · <strong>AMCB–IBU</strong> · {scope}</p>', unsafe_allow_html=True)


# ==========================================
# 8. Input form
# ==========================================
with st.form("prediction_form"):
    values: dict[str, float] = {}
    for column, (icon, en, cn, features) in zip(st.columns(3, gap="large"), OPERATING_GROUPS):
        with column:
            st.markdown(f"### {icon} {cn if language == 'cn' else en}")
            for feature in features:
                values[feature] = make_input(
                    t[FEATURE_LABEL_KEYS[feature]], float(row[feature]), feature, choice
                )
    submit_btn = st.form_submit_button("开始预测" if language == "cn" else "Predict adsorption", type="primary")


# ==========================================
# 9. Single prediction and export
# ==========================================
if submit_btn:
    st.session_state["prediction_values"] = values.copy()

if "prediction_values" in st.session_state:
    try:
        prediction_row = row.copy()
        for feature, value in st.session_state["prediction_values"].items():
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
