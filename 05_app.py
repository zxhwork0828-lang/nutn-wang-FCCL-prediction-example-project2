"""銅箔資料探索與 Random Forest 推論。啟動：streamlit run 05_app.py。"""

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import hashlib
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# 依程式所在目錄定位，支援從其他工作目錄或部署平台啟動。
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/prepared/copper_foil_cleaned.csv"
MODEL_DIR = ROOT / "models"
PROCESS = ["Line_Speed_mpm", "Current_ASD", "Bath_Temp_C", "Cu_Conc_gL",
           "Additive_ppm", "Jet_Pressure_bar", "Edge_Current_Bias"]
POSITIVE = ["Line_Speed_mpm", "Current_ASD", "Cu_Conc_gL", "Additive_ppm", "Jet_Pressure_bar"]
LABELS = {
    "Line_Speed_mpm": "線速 (m/min)", "Current_ASD": "電流密度 (ASD)",
    "Bath_Temp_C": "槽液溫度 (°C)", "Cu_Conc_gL": "銅離子濃度 (g/L)",
    "Additive_ppm": "添加劑濃度 (ppm)", "Jet_Pressure_bar": "噴流壓力 (bar)",
    "Edge_Current_Bias": "邊緣電流偏差", "Thick_Mean_um": "平均厚度 (μm)",
    "Thick_Range_um": "厚度極差 (μm)",
}
COLORS = alt.Scale(domain=["正常", "異常"], range=["#238b8d", "#e76f51"])


@st.cache_data
def load_data(path: str, modified: int) -> pd.DataFrame:
    """快取資料；檔案更新時間變動時重新讀取。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = PROCESS + ["Timestamp", "Sample_ID", "Split", "Is_Anomaly", "Shift",
                          "Thick_Mean_um", "Thick_Range_um"]
    missing = sorted(set(required) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"資料為空或缺少欄位：{missing}")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="raise")
    if frame["Timestamp"].isna().any() or not frame["Is_Anomaly"].isin([0, 1]).all():
        raise ValueError("時間或異常標籤無效")
    frame["品質"] = frame["Is_Anomaly"].map({0: "正常", 1: "異常"})
    return frame.sort_values("Timestamp")


@st.cache_resource
def load_model(files_signature: tuple):
    """只載入專案內可信模型；檔案更新時重新載入。"""
    metadata = json.loads((MODEL_DIR / "random_forest_metadata.json").read_text(encoding="utf-8"))
    params = json.loads((MODEL_DIR / "random_forest_preprocessing.json").read_text(encoding="utf-8"))
    model_path = MODEL_DIR / "random_forest_model.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("模型與設定檔不一致，請重新匯出整組模型檔案。")
    model = joblib.load(model_path)
    if not (metadata["feature_columns"] == params["feature_columns"] == model.feature_names_in_.tolist()):
        raise ValueError("模型與前處理特徵順序不一致。")
    if params["process_columns"] != PROCESS or not 0 <= metadata["threshold"] <= 1:
        raise ValueError("製程欄位或門檻設定不相容。")
    return model, metadata, params


def transform_input(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    """沿用 02/04 的前處理；新資料不重新 fit。"""
    timestamp = pd.to_datetime(raw["Timestamp"], errors="raise")
    if timestamp.isna().any():
        raise ValueError("時間不可為空")
    numeric = raw[PROCESS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in POSITIVE:
        numeric.loc[numeric[column].le(0), column] = np.nan
    features = pd.DataFrame(index=raw.index)
    for column in PROCESS:
        # 先保留缺值旗標，再使用訓練中位數補值與訓練參數標準化。
        features[f"{column}_was_missing"] = numeric[column].isna().astype(int)
        scale = params["scaler_scale"][column]
        if scale <= 0:
            raise ValueError(f"{column} 標準化參數無效")
        filled = numeric[column].fillna(params["imputer_median"][column])
        features[column] = (filled - params["scaler_mean"][column]) / scale
    hours = timestamp.dt.hour
    shift = pd.Series(np.select([hours.between(8, 15), hours.between(16, 23)],
                                ["早班", "中班"], default="晚班"), index=raw.index)
    for category in params["shift_categories"]:
        features[f"Shift_{category}"] = shift.eq(category).astype(float)
    fractional_hour = hours + timestamp.dt.minute / 60
    features["Hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24)
    features["Hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24)
    features = features.loc[:, params["feature_columns"]].astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("輸入轉換後含無效數值")
    return features


def visualization_page():
    st.title("銅箔製程資料總覽")
    st.caption("探索製程條件與品質表現。圖表使用清理後的原始單位資料。")
    try:
        frame = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"無法讀取圖表資料：{exc}")
        st.info("請將 data/prepared/copper_foil_cleaned.csv 一併放入部署專案。")
        return
    left, right = st.columns(2)
    splits = left.multiselect("資料集合", ["train", "validation", "test"],
                              default=["train", "validation", "test"])
    dates = right.date_input("日期範圍", value=(frame["Timestamp"].min().date(), frame["Timestamp"].max().date()),
                             min_value=frame["Timestamp"].min().date(), max_value=frame["Timestamp"].max().date())
    if len(dates) != 2:
        st.info("請選擇完整的開始與結束日期。")
        return
    filtered = frame.loc[frame["Split"].isin(splits) &
                         frame["Timestamp"].dt.date.between(dates[0], dates[1])].copy()
    if filtered.empty:
        st.info("目前篩選沒有資料，請調整集合或日期。")
        return
    count, rate, thick, spread = st.columns(4)
    count.metric("樣本數", f"{len(filtered):,}")
    rate.metric("異常比例", f"{filtered['Is_Anomaly'].mean():.1%}")
    thick.metric("平均厚度", f"{filtered['Thick_Mean_um'].mean():.2f} μm")
    spread.metric("平均厚度極差", f"{filtered['Thick_Range_um'].mean():.2f} μm")
    st.caption("教學合成資料；異常比例與厚度僅供描述分析，不代表實際產線規格。")
    trend_tab, distribution_tab, relation_tab, data_tab = st.tabs(["時間趨勢", "分布與品質", "特徵關係", "資料明細"])
    with trend_tab:
        selected = st.selectbox("趨勢特徵", list(LABELS), format_func=LABELS.get)
        base = alt.Chart(filtered).encode(
            x=alt.X("Timestamp:T", title="時間"), y=alt.Y(f"{selected}:Q", title=LABELS[selected], scale=alt.Scale(zero=False)),
            tooltip=["Sample_ID:N", "Timestamp:T", alt.Tooltip(f"{selected}:Q", title=LABELS[selected]), "品質:N"])
        # 按時間連線、用點色區分品質，避免按品質分組連線造成誤讀。
        st.altair_chart(base.mark_line(color="#9ba9b5") + base.mark_circle(size=25).encode(
            color=alt.Color("品質:N", scale=COLORS)), width="stretch")
    with distribution_tab:
        feature = st.selectbox("分布特徵", PROCESS, format_func=LABELS.get)
        first, second = st.columns(2)
        first.altair_chart(alt.Chart(filtered).mark_bar(opacity=0.8).encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=25), title=LABELS[feature]),
            y=alt.Y("count():Q", title="樣本數"), color=alt.Color("品質:N", scale=COLORS),
            tooltip=["品質:N", alt.Tooltip("count():Q", title="樣本數")]), width="stretch")
        second.altair_chart(alt.Chart(filtered).mark_boxplot().encode(
            x=alt.X("品質:N", title="品質"), y=alt.Y(f"{feature}:Q", title=LABELS[feature], scale=alt.Scale(zero=False)),
            color=alt.Color("品質:N", scale=COLORS)), width="stretch")
        shift = filtered.groupby("Shift", as_index=False).agg(異常比例=("Is_Anomaly", "mean"), 樣本數=("Sample_ID", "size"))
        st.altair_chart(alt.Chart(shift).mark_bar(color="#238b8d").encode(
            x=alt.X("Shift:N", title="班別", sort=["早班", "中班", "晚班"]),
            y=alt.Y("異常比例:Q", axis=alt.Axis(format=".0%")), tooltip=["Shift:N", "樣本數:Q", alt.Tooltip("異常比例:Q", format=".1%")]),
            width="stretch")
    with relation_tab:
        first, second = st.columns(2)
        x = first.selectbox("X 軸特徵", PROCESS, format_func=LABELS.get)
        y = second.selectbox("Y 軸特徵", PROCESS, index=1, format_func=LABELS.get)
        st.altair_chart(alt.Chart(filtered).mark_circle(size=45, opacity=0.65).encode(
            x=alt.X(f"{x}:Q", title=LABELS[x], scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y}:Q", title=LABELS[y], scale=alt.Scale(zero=False)),
            color=alt.Color("品質:N", scale=COLORS), tooltip=["Sample_ID:N", f"{x}:Q", f"{y}:Q", "品質:N"]),
            width="stretch")
        correlation = filtered[PROCESS].rename(columns=LABELS).corr()
        correlation.index.name = "特徵一"
        correlation.columns.name = "特徵二"
        long = correlation.stack().rename("相關係數").reset_index()
        st.altair_chart(alt.Chart(long).mark_rect().encode(
            x="特徵一:N", y="特徵二:N", color=alt.Color("相關係數:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
            tooltip=["特徵一:N", "特徵二:N", alt.Tooltip("相關係數:Q", format=".2f")]), width="stretch")
        st.caption("Pearson 相關性不代表因果關係；清理資料中的補值也可能影響分布與相關性。")
    with data_tab:
        st.dataframe(filtered, hide_index=True, width="stretch")
        st.download_button("下載篩選資料 CSV", filtered.to_csv(index=False).encode("utf-8-sig"),
                           "filtered_copper_foil.csv", "text/csv")


def prediction_page():
    st.title("製程品質異常預測")
    st.caption("輸入製程條件與採樣時間，預測是否為品質異常。班別由時間自動計算。")
    try:
        paths = [MODEL_DIR / filename for filename in ["random_forest_model.joblib",
                 "random_forest_metadata.json", "random_forest_preprocessing.json"]]
        model, metadata, params = load_model(tuple(path.stat().st_mtime_ns for path in paths))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        st.error(f"無法載入模型：{exc}")
        st.info("請確認 models/ 內的模型與兩份 JSON 設定檔完整，且使用 requirements.txt 的套件版本。")
        return
    threshold = metadata["threshold"]
    st.info(f"異常判定門檻：{threshold:.2f}。模型分數大於或等於此門檻，判定為異常。")
    with st.form("prediction_form"):
        date_col, time_col = st.columns(2)
        sample_date = date_col.date_input("採樣日期", value=datetime.now(timezone(timedelta(hours=8))).date())
        sample_time = time_col.time_input("採樣時間", value=time(8, 0), step=900)
        values = {}
        columns = st.columns(2)
        for i, feature in enumerate(PROCESS):
            # 預設為訓練中位數；正值欄位設下界，邊緣電流偏差可輸入負值。
            values[feature] = columns[i % 2].number_input(
                LABELS[feature], value=float(params["imputer_median"][feature]),
                min_value=0.001 if feature in POSITIVE else None,
                step=0.01, format="%.3f", key=f"input_{feature}")
        submitted = st.form_submit_button("進行異常預測", type="primary")
    if submitted:
        try:
            if not np.isfinite(list(values.values())).all():
                raise ValueError("請輸入有限數值")
            raw = pd.DataFrame([{**values, "Timestamp": datetime.combine(sample_date, sample_time)}])
            features = transform_input(raw, params)
            positive = np.flatnonzero(model.classes_ == metadata["positive_class"])
            if len(positive) != 1:
                raise ValueError("模型缺少異常類別")
            probability = float(model.predict_proba(features)[0, int(positive[0])])
            is_anomaly = probability >= threshold
            # 儲存提交快照，避免將尚未提交的新輸入當作已預測資料。
            hour = sample_time.hour
            raw["Shift"] = "早班" if 8 <= hour < 16 else "中班" if 16 <= hour < 24 else "晚班"
            raw["anomaly_probability"] = probability
            raw["threshold"] = threshold
            raw["predicted_anomaly"] = int(is_anomaly)
            raw["prediction_label"] = "異常" if is_anomaly else "正常"
            st.session_state["last_prediction"] = raw
        except (ValueError, KeyError, TypeError) as exc:
            st.session_state.pop("last_prediction", None)
            st.error(f"預測失敗：{exc}")
    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        record = result.iloc[0]
        st.subheader("最近一次提交的預測結果")
        if record["predicted_anomaly"]:
            st.error("預測結果：異常，建議確認製程條件與實際品質量測。")
        else:
            st.success("預測結果：正常。")
        score_col, threshold_col, shift_col = st.columns(3)
        score_col.metric("異常模型分數", f"{record['anomaly_probability']:.2%}")
        threshold_col.metric("判定門檻", f"{record['threshold']:.2f}")
        shift_col.metric("採樣班別", record["Shift"])
        with st.expander("查看本次提交資料"):
            st.dataframe(result, hide_index=True, width="stretch")
        st.download_button("下載本次預測 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                           "copper_foil_prediction.csv", "text/csv")
    st.caption("模型分數未經機率校準。模型使用教學合成資料訓練，且測試集未含異常樣本，尚無法驗證未來異常召回能力。")


def main():
    st.set_page_config(page_title="銅箔製程品質分析", page_icon="📊", layout="wide")
    with st.sidebar:
        st.title("銅箔品質分析")
        page = st.radio("功能分頁", ["資料視覺化", "異常預測"])
        st.caption("Random Forest · 製程品質探索與預測")
    if page == "資料視覺化":
        visualization_page()
    else:
        prediction_page()


if __name__ == "__main__":
    main()
