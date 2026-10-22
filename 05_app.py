"""銅箔資料探索與 Random Forest 推論。啟動：streamlit run 05_app.py。"""

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import hashlib
import json
from time import monotonic

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


def calculate_spc(values, baseline_count: int):
    """Freeze I-MR limits from the initial reference window (MR span = 2)."""
    series = pd.Series(values, dtype=float).reset_index(drop=True)
    if not np.isfinite(series).all():
        raise ValueError("SPC 資料必須是完整的有限數值，請先處理缺失值。")
    if not 2 <= baseline_count < len(series):
        raise ValueError("基準筆數至少為 2，且必須保留後續監測資料。")
    baseline = series.iloc[:baseline_count]
    center = float(baseline.mean())
    mr_mean = float(baseline.diff().abs().mean())
    if mr_mean <= 0:
        raise ValueError("基準資料沒有變異，無法估計管制界限，請更換基準期間。")
    sigma = mr_mean / 1.128
    limits = {"CL": center, "LCL": center - 3 * sigma, "UCL": center + 3 * sigma,
              "MR_CL": mr_mean, "MR_LCL": 0.0, "MR_UCL": 3.267 * mr_mean}
    result = pd.DataFrame({"Value": series, "MR": series.diff().abs()})
    result["I_signal"] = (series < limits["LCL"]) | (series > limits["UCL"])
    result["MR_signal"] = result.MR > limits["MR_UCL"]
    # Report the endpoint of each run, including runs crossing the reference boundary.
    above = series.gt(center).rolling(8).sum().eq(8)
    below = series.lt(center).rolling(8).sum().eq(8)
    result["Run8_signal"] = above | below
    result["Phase"] = np.where(result.index < baseline_count, "基準", "監測")
    result["Signal"] = result[["I_signal", "MR_signal", "Run8_signal"]].any(axis=1)
    return result, limits


def r2r_step(measured, current, previous_offset, *, target, nominal_thickness,
             nominal_current, gain, weight, lower, upper, max_step, deadband):
    """EWMA intercept adaptation for y = y0 + gain * (u - u0) + offset."""
    numbers = [measured, current, previous_offset, target, nominal_thickness,
               nominal_current, gain, weight, lower, upper, max_step, deadband]
    if not np.isfinite(numbers).all():
        raise ValueError("R2R 參數必須為有限數值。")
    if gain <= 0 or not 0 < weight <= 1 or lower <= 0 or upper <= lower or max_step <= 0 or deadband < 0:
        raise ValueError("請確認正增益、EWMA 權重、電流上下限、單批調整上限與死區。")
    if not lower <= current <= upper or not lower <= nominal_current <= upper:
        raise ValueError("目前電流與名目電流必須位於設定上下限內。")
    residual = measured - nominal_thickness - gain * (current - nominal_current)
    offset = weight * residual + (1 - weight) * previous_offset
    desired = nominal_current + (target - nominal_thickness - offset) / gain
    hold = abs(measured - target) <= deadband
    proposed = current if hold else desired
    next_current = float(np.clip(np.clip(proposed, current - max_step, current + max_step), lower, upper))
    return {"offset": offset, "desired_current": desired, "next_current": next_current,
            "delta": next_current - current, "held": hold,
            "limited": not np.isclose(next_current, proposed)}


def simulate_r2r(*, runs, seed, drift, noise, disturbance_step, plant_gain, **settings):
    if runs < 2 or not np.isfinite([drift, noise, disturbance_step, plant_gain]).all() or noise < 0 or plant_gain <= 0:
        raise ValueError("請確認模擬批數、擾動與實際製程正增益。")
    rng = np.random.default_rng(seed)
    disturbances = drift * np.arange(runs) + rng.normal(0, noise, runs)
    disturbances[runs // 2:] += disturbance_step
    current, offset = settings["nominal_current"], 0.0
    records = []
    for index, disturbance in enumerate(disturbances):
        open_output = settings["nominal_thickness"] + disturbance
        measured = open_output + plant_gain * (current - settings["nominal_current"])
        decision = r2r_step(measured, current, offset, **settings)
        records.append({"Run": index + 1, "Open_loop_um": open_output, "R2R_um": measured,
                        "Current_ASD": current, "Next_Current_ASD": decision["next_current"],
                        "Offset_um": decision["offset"], "Limited": decision["limited"],
                        "Held": decision["held"], "Target_um": settings["target"]})
        current, offset = decision["next_current"], decision["offset"]
    return pd.DataFrame(records)


def spc_page():
    st.title("SPC 統計製程管制")
    st.caption("以時間排序的單筆量測建立 I-MR 管制圖；前段建立界限，後段使用固定界限監測。")
    try:
        frame = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns).reset_index(drop=True)
        if len(frame) < 3:
            raise ValueError("至少需要 3 筆時間序列資料。")
        left, right = st.columns(2)
        feature = left.selectbox("監控量測", list(LABELS), index=7, format_func=LABELS.get)
        baseline_count = int(right.number_input("前段基準筆數", min_value=2, max_value=len(frame) - 1,
                                               value=min(100, len(frame) - 1), step=1))
        result, limits = calculate_spc(frame[feature], baseline_count)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"無法建立 SPC：{exc}")
        return
    result.insert(0, "Timestamp", frame.Timestamp)
    result.insert(0, "Sample_ID", frame.Sample_ID)
    monitoring = result.iloc[baseline_count:]
    a, b, c, d = st.columns(4)
    a.metric("監測筆數", len(monitoring))
    b.metric("I 圖超界", int(monitoring.I_signal.sum()))
    c.metric("MR 圖超界", int(monitoring.MR_signal.sum()))
    d.metric("連續 8 點同側訊號", int(monitoring.Run8_signal.sum()))
    if result.iloc[:baseline_count].Signal.any():
        st.warning("基準期間也有管制訊號；請調查特殊原因並確認基準穩定性，再解讀後續警示。")
    for value, signal, title, keys in [
        ("Value", "I_signal", f"I 圖：{LABELS[feature]}", ["LCL", "CL", "UCL"]),
        ("MR", "MR_signal", "MR 圖：相鄰兩筆量測差的絕對值", ["MR_LCL", "MR_CL", "MR_UCL"]),
    ]:
        st.subheader(title)
        base = alt.Chart(result).encode(x=alt.X("Timestamp:T", title="時間"),
                                        y=alt.Y(f"{value}:Q", scale=alt.Scale(zero=False), title=title))
        points = base.mark_circle(size=30).encode(
            color=alt.condition(alt.datum[signal], alt.value("#e76f51"), alt.value("#238b8d")),
            tooltip=["Sample_ID:N", "Timestamp:T", f"{value}:Q", "Phase:N", "Run8_signal:N"])
        rules = alt.Chart(pd.DataFrame({"Limit": keys, "Value": [limits[k] for k in keys]})).mark_rule(
            strokeDash=[5, 4]).encode(y="Value:Q", color="Limit:N", tooltip=["Limit:N", "Value:Q"])
        boundary = alt.Chart(pd.DataFrame({"Timestamp": [frame.Timestamp.iloc[baseline_count]]})).mark_rule(
            color="#777777", strokeDash=[2, 2]).encode(x="Timestamp:T")
        st.altair_chart(base.mark_line(color="#9ba9b5") + points + rules + boundary, width="stretch")
    st.caption(f"基準結束：{frame.Timestamp.iloc[baseline_count - 1]}。灰色垂直線為監測起點。"
               "連續 8 點同側訊號標記在第 8 點及其後持續同側的各點，可跨越基準邊界。"
               "管制界限描述製程變異，不等同產品規格；合成資料的訊號僅供示範。")
    st.dataframe(pd.DataFrame([limits]), hide_index=True, width="stretch")
    only_signals = st.checkbox("僅顯示監測期間有訊號的資料", value=True)
    st.dataframe(monitoring.loc[monitoring.Signal] if only_signals else result, hide_index=True, width="stretch")
    st.download_button("下載 SPC 完整結果", result.assign(**limits).to_csv(index=False).encode("utf-8-sig"),
                       "spc_results.csv", "text/csv")
    st.markdown("公式來源：[NIST Individuals Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm)。")


def r2r_page():
    st.title("R2R 批次間回饋控制")
    st.info("可調增益的教學模擬：依實測平均厚度建議下一批電流密度。預設增益是假設值，需以製程實驗辨識；此頁不連接設備。")
    with st.expander("控制器參數", expanded=True):
        a, b, c = st.columns(3)
        target = a.number_input("目標厚度 (μm)", value=18.0, min_value=0.001, key="r2r_target")
        nominal_thickness = b.number_input("名目厚度 (μm)", value=18.0, min_value=0.001)
        nominal_current = c.number_input("名目電流 (ASD)", value=58.0, min_value=0.001)
        gain = a.number_input("控制器增益 (μm/ASD，假設值)", value=0.30, min_value=0.001, step=0.01, format="%.3f")
        weight = b.slider("EWMA 權重", min_value=0.01, max_value=1.0, value=0.3, step=0.01)
        deadband = c.number_input("厚度死區 (μm)", value=0.02, min_value=0.0, step=0.01)
        lower = a.number_input("電流下限 (ASD)", value=45.0, min_value=0.001)
        upper = b.number_input("電流上限 (ASD)", value=70.0, min_value=0.001)
        max_step = c.number_input("單批最大電流調整 (ASD)", value=1.0, min_value=0.001, step=0.1)
    settings = dict(target=target, nominal_thickness=nominal_thickness, nominal_current=nominal_current,
                    gain=gain, weight=weight, lower=lower, upper=upper, max_step=max_step, deadband=deadband)
    manual, simulation = st.tabs(["下一批設定建議", "多批次控制模擬"])
    with manual:
        a, b, c = st.columns(3)
        measured = a.number_input("本批實測平均厚度 (μm)", value=18.5, min_value=0.001)
        current = b.number_input("本批實際電流 (ASD)", value=58.0, min_value=0.001)
        previous_offset = c.number_input("前次 EWMA 偏移估計 (μm)", value=0.0, step=0.01)
        try:
            decision = r2r_step(measured, current, previous_offset, **settings)
            a.metric("建議下一批電流", f"{decision['next_current']:.3f} ASD", f"{decision['delta']:+.3f} ASD")
            b.metric("更新後偏移估計", f"{decision['offset']:.4f} μm")
            c.metric("調整狀態", "死區內維持" if decision["held"] else "已限幅" if decision["limited"] else "一般調整")
            st.caption("跨批使用時，將更新後偏移估計填入下一次的前次估計；此區每次依目前輸入計算，不會自行累積狀態。")
            export = {**settings, "measured_um": measured, "current_asd": current,
                      "previous_offset_um": previous_offset, **decision}
            st.download_button("下載本次 R2R 建議", pd.DataFrame([export]).to_csv(index=False).encode("utf-8-sig"),
                               "r2r_recommendation.csv", "text/csv")
        except ValueError as exc:
            st.error(str(exc))
    with simulation:
        a, b, c = st.columns(3)
        runs = int(a.number_input("模擬批數", min_value=10, max_value=1000, value=100, step=10))
        seed = int(b.number_input("隨機種子", min_value=0, value=42, step=1))
        plant_gain = c.number_input("模擬製程實際增益 (μm/ASD)", value=0.30, min_value=0.001, step=0.01)
        drift = a.number_input("每批厚度漂移 (μm)", value=0.005, step=0.001, format="%.3f")
        noise = b.number_input("量測雜訊標準差 (μm)", value=0.05, min_value=0.0, step=0.01)
        disturbance_step = c.number_input("中途階躍擾動 (μm)", value=0.3, step=0.1)
        try:
            simulated = simulate_r2r(runs=runs, seed=seed, drift=drift, noise=noise,
                                     disturbance_step=disturbance_step, plant_gain=plant_gain, **settings)
        except ValueError as exc:
            st.error(str(exc))
            return
        a.metric("無控制 RMSE", f"{np.sqrt(np.mean((simulated.Open_loop_um - target) ** 2)):.4f} μm")
        b.metric("R2R RMSE", f"{np.sqrt(np.mean((simulated.R2R_um - target) ** 2)):.4f} μm")
        c.metric("調整限幅次數", int(simulated.Limited.sum()))
        long = simulated.melt(id_vars=["Run"], value_vars=["Open_loop_um", "R2R_um", "Target_um"],
                              var_name="Series", value_name="Thickness_um")
        st.altair_chart(alt.Chart(long).mark_line().encode(x=alt.X("Run:Q", title="批次"),
            y=alt.Y("Thickness_um:Q", title="平均厚度 (μm)", scale=alt.Scale(zero=False)),
            color="Series:N", tooltip=["Run:Q", "Series:N", "Thickness_um:Q"]), width="stretch")
        st.altair_chart(alt.Chart(simulated).mark_line(color="#238b8d").encode(x="Run:Q",
            y=alt.Y("Current_ASD:Q", title="本批使用電流 (ASD)", scale=alt.Scale(zero=False)),
            tooltip=["Run:Q", "Current_ASD:Q", "Next_Current_ASD:Q", "Limited:N"]), width="stretch")
        st.caption("無控制與 R2R 使用相同擾動及雜訊。第 k 批量測完成後才更新第 k+1 批電流；"
                   "兩組增益可設不同值，觀察模型誤差的影響。此為線性假設的模擬，並非歷史數據控制成效。")
        st.dataframe(simulated, hide_index=True, width="stretch")
        st.download_button("下載 R2R 模擬結果", simulated.assign(Seed=seed, Drift=drift, Noise=noise,
            Disturbance_step=disturbance_step, Plant_gain=plant_gain, **settings).to_csv(index=False).encode("utf-8-sig"),
            "r2r_simulation.csv", "text/csv")
    with st.expander("計算方式"):
        st.code("y = y₀ + G × (u − u₀) + d\n"
                "d̂ₖ = λ × [yₖ − y₀ − G × (uₖ − u₀)] + (1 − λ) × d̂ₖ₋₁\n"
                "uₖ₊₁ = u₀ + (target − y₀ − d̂ₖ) / G\n"
                "死區內維持電流；其他情況套用單批調整上限與電流上下限。")


def create_live_simulation(frame, *, seed, sample_limit, baseline_count, source, settings, now):
    """Create an independent replay session; reference rows never enter the draw pool."""
    if sample_limit < 1:
        raise ValueError("模擬筆數至少為 1。")
    reference = frame.iloc[:baseline_count].Thick_Mean_um.tolist()
    # Validate the reference without using a future monitored measurement.
    reference_result, limits = calculate_spc(reference + [reference[-1]] if reference else [], baseline_count)
    r2r_step(settings["nominal_thickness"], settings["nominal_current"], 0., **settings)
    pool = frame.iloc[baseline_count:].copy()
    if source == "僅異常樣本":
        pool = pool.loc[pool.Is_Anomaly.eq(1)]
    elif source == "僅正常樣本":
        pool = pool.loc[pool.Is_Anomaly.eq(0)]
    elif source != "全部樣本":
        raise ValueError("無效的抽樣來源。")
    if pool.empty:
        raise ValueError("基準期間以外沒有符合條件的抽樣資料，請減少基準筆數或切換來源。")
    if not np.isfinite(pool[PROCESS + ["Thick_Mean_um", "anomaly_probability"]].to_numpy(dtype=float)).all():
        raise ValueError("抽樣資料含無效數值。")
    return {"running": True, "rng": np.random.default_rng(seed), "seed": seed,
            "sample_limit": sample_limit, "source": source, "pool": pool.reset_index(drop=True),
            "reference": reference, "limits": limits, "settings": settings.copy(), "offset": 0.,
            "baseline_warning": bool(reference_result.iloc[:baseline_count].Signal.any()),
            "next_due": now + 1., "records": []}


def advance_live_simulation(state, *, now, emitted_at):
    """Emit about once per second, tolerating 30 ms of timer jitter without bursts."""
    if not state["running"] or now + .03 < state["next_due"]:
        return False
    if len(state["records"]) >= state["sample_limit"]:
        state["running"] = False
        return False
    row = state["pool"].iloc[int(state["rng"].integers(len(state["pool"])))]
    record = {"Sequence": len(state["records"]) + 1, "Received_at": emitted_at,
              "Source_Timestamp": row.Timestamp, "Sample_ID": row.Sample_ID,
              **{c: float(row[c]) for c in PROCESS},
              "Thick_Mean_um": float(row.Thick_Mean_um), "Thick_Range_um": float(row.Thick_Range_um),
              "Is_Anomaly": int(row.Is_Anomaly), "anomaly_probability": float(row.anomaly_probability),
              "threshold": float(row.threshold)}
    record["Predicted_anomaly"] = record["anomaly_probability"] >= record["threshold"]
    values = state["reference"] + [r["Thick_Mean_um"] for r in state["records"]] + [record["Thick_Mean_um"]]
    spc, _ = calculate_spc(values, len(state["reference"]))
    for column in ["MR", "I_signal", "MR_signal", "Run8_signal", "Signal"]:
        record[column] = spc.iloc[-1][column]
    # Estimates carry forward; historical input currents remain the measured recipe.
    record["Previous_offset_um"] = state["offset"]
    try:
        decision = r2r_step(record["Thick_Mean_um"], record["Current_ASD"], state["offset"], **state["settings"])
        state["offset"] = decision["offset"]
        record.update(Next_Current_ASD=decision["next_current"], Delta_ASD=decision["delta"],
                      R2R_status="死區內維持" if decision["held"] else "已限幅" if decision["limited"] else "一般調整")
    except ValueError as exc:
        record.update(Next_Current_ASD=np.nan, Delta_ASD=np.nan, R2R_status=f"暫停建議：{exc}")
    record["Offset_um"] = state["offset"]
    state["records"].append(record)
    state["next_due"] = now + 1.
    if len(state["records"]) >= state["sample_limit"]:
        state["running"] = False
    return True


def live_simulation_page():
    st.title("即時資料輸入模擬")
    st.caption("每秒隨機抽取一筆既有製程數據，更新異常預測、厚度 SPC 與下一批電流建議。")
    st.info("隨機重播會打亂原始時間順序；SPC 訊號僅供流程示範。R2R 建議不套用到下一筆歷史量測，"
            "若要觀察調整後的厚度反應，請使用「R2R 控制模擬」。")
    state = st.session_state.get("live_simulation")
    running = bool(state and state["running"])
    try:
        frame = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns).reset_index(drop=True)
        if len(frame) < 3:
            raise ValueError("至少需要 3 筆資料。")
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"無法載入模擬資料：{exc}")
        return
    with st.expander("抽樣與控制參數（開始新模擬時套用）", expanded=state is None):
        with st.form("live_settings"):
            a, b, c = st.columns(3)
            source = a.selectbox("隨機抽樣來源", ["全部樣本", "僅異常樣本", "僅正常樣本"], disabled=running)
            seed = int(b.number_input("重播隨機種子", min_value=0, value=42, step=1, disabled=running))
            sample_limit = int(c.number_input("本次輸送筆數", min_value=1, max_value=1000, value=120, step=1, disabled=running))
            baseline_count = int(a.number_input("SPC 前段基準筆數", min_value=2, max_value=len(frame) - 1,
                                                value=min(100, len(frame) - 1), step=1, disabled=running))
            target = b.number_input("動態目標厚度 (μm)", value=18., min_value=0.001, disabled=running)
            gain = c.number_input("動態控制增益 (μm/ASD，假設值)", value=.3, min_value=.001, step=.01, disabled=running)
            nominal_thickness = a.number_input("動態名目厚度 (μm)", value=18., min_value=.001, disabled=running)
            nominal_current = b.number_input("動態名目電流 (ASD)", value=58., min_value=.001, disabled=running)
            weight = c.slider("動態 EWMA 權重", min_value=.01, max_value=1., value=.3, step=.01, disabled=running)
            lower = a.number_input("動態電流下限 (ASD)", value=45., min_value=.001, disabled=running)
            upper = b.number_input("動態電流上限 (ASD)", value=70., min_value=.001, disabled=running)
            max_step = c.number_input("動態單筆調整上限 (ASD)", value=1., min_value=.001, disabled=running)
            deadband = a.number_input("動態厚度死區 (μm)", value=.02, min_value=0., step=.01, disabled=running)
            start = st.form_submit_button("開始新模擬", type="primary", disabled=running)
    if start:
        try:
            paths = [MODEL_DIR / name for name in ["random_forest_model.joblib", "random_forest_metadata.json",
                                                  "random_forest_preprocessing.json"]]
            model, metadata, params = load_model(tuple(path.stat().st_mtime_ns for path in paths))
            raw = frame.copy()
            # Restore missing indicators before inference on already imputed source data.
            for column in PROCESS:
                flag = f"{column}_was_missing"
                if flag in raw:
                    raw.loc[raw[flag].eq(1), column] = np.nan
            positive = np.flatnonzero(model.classes_ == metadata["positive_class"])
            if len(positive) != 1:
                raise ValueError("模型缺少異常類別。")
            frame["anomaly_probability"] = model.predict_proba(transform_input(raw, params))[:, int(positive[0])]
            frame["threshold"] = metadata["threshold"]
            settings = dict(target=target, nominal_thickness=nominal_thickness, nominal_current=nominal_current,
                            gain=gain, weight=weight, lower=lower, upper=upper, max_step=max_step, deadband=deadband)
            st.session_state["live_simulation"] = create_live_simulation(frame, seed=seed, sample_limit=sample_limit,
                baseline_count=baseline_count, source=source, settings=settings, now=monotonic())
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            st.error(f"無法開始模擬：{exc}")
        else:
            st.rerun()
    state = st.session_state.get("live_simulation")
    if state is None:
        st.info("設定參數後按「開始新模擬」，第一筆資料約在一秒後送入。")
        return
    a, b, c = st.columns(3)
    if a.button("暫停輸送", disabled=not state["running"]):
        state["running"] = False
        st.rerun()
    if b.button("繼續輸送", disabled=state["running"] or len(state["records"]) >= state["sample_limit"]):
        state["running"], state["next_due"] = True, monotonic() + 1.
        st.rerun()
    if c.button("清除模擬"):
        del st.session_state["live_simulation"]
        st.rerun()
    st.caption(f"本次設定：{state['source']}｜種子 {state['seed']}｜每秒 1 筆（允許重複抽樣）｜"
               f"目標 {state['settings']['target']:.3f} μm｜增益 {state['settings']['gain']:.3f} μm/ASD。"
               "離開此分頁會暫停；網頁需保持連線，延遲時不會補送多筆。")
    if state["baseline_warning"]:
        st.warning("SPC 基準本身有管制訊號，請先確認基準穩定性；此處使用固定界限展示監測流程。")

    @st.fragment(run_every=1.0 if state["running"] else None)
    def live_dashboard():
        emitted = advance_live_simulation(state, now=monotonic(),
            emitted_at=datetime.now(timezone(timedelta(hours=8))))
        if emitted and not state["running"]:
            st.rerun()  # Remove the timer and update the outer controls on completion.
        count = len(state["records"])
        status = "輸送中" if state["running"] else "已完成" if count >= state["sample_limit"] else "已暫停"
        st.subheader(f"{status} · {count} / {state['sample_limit']} 筆")
        st.progress(count / state["sample_limit"])
        if not count:
            st.info("等待下一筆資料…")
            return
        history = pd.DataFrame(state["records"])
        latest = history.iloc[-1]
        a, b, c, d = st.columns(4)
        a.metric("最新平均厚度", f"{latest.Thick_Mean_um:.3f} μm",
                 f"{latest.Thick_Mean_um - state['settings']['target']:+.3f} μm")
        b.metric("異常模型分數", f"{latest.anomaly_probability:.2%}")
        c.metric("SPC 訊號筆數", int(history.Signal.sum()))
        d.metric("模型判異常筆數", int(history.Predicted_anomaly.sum()))
        a, b, c = st.columns(3)
        a.metric("本筆實際電流", f"{latest.Current_ASD:.3f} ASD")
        b.metric("建議下一批電流", "無建議" if pd.isna(latest.Next_Current_ASD) else f"{latest.Next_Current_ASD:.3f} ASD",
                 None if pd.isna(latest.Delta_ASD) else f"{latest.Delta_ASD:+.3f} ASD")
        c.metric("EWMA 偏移估計", f"{latest.Offset_um:.4f} μm")
        st.write(f"來源樣本：{latest.Sample_ID}｜原始採樣：{latest.Source_Timestamp}｜接收：{latest.Received_at}")
        signals = [label for column, label in [("I_signal", "I 圖超界"), ("MR_signal", "MR 圖超界"),
                   ("Run8_signal", "連續 8 點同側")] if latest[column]]
        if latest.Predicted_anomaly or signals:
            st.warning(f"模型：{'異常' if latest.Predicted_anomaly else '正常'}｜SPC：{'、'.join(signals) or '無訊號'}")
        else:
            st.success("本筆模型判定正常，SPC 無訊號。")
        st.write(f"R2R：{latest.R2R_status}")
        view = history.tail(100)
        chart = alt.Chart(view).encode(x=alt.X("Sequence:Q", title="輸送序號"),
            y=alt.Y("Thick_Mean_um:Q", title="平均厚度 (μm)", scale=alt.Scale(zero=False)))
        points = chart.mark_circle(size=45).encode(
            color=alt.condition(alt.datum.Signal, alt.value("#e76f51"), alt.value("#238b8d")),
            tooltip=["Sequence:Q", "Sample_ID:N", "Thick_Mean_um:Q", "anomaly_probability:Q"])
        lines = pd.DataFrame({"Line": ["LCL", "CL", "UCL", "目標"],
                              "Thickness": [state["limits"][k] for k in ["LCL", "CL", "UCL"]] + [state["settings"]["target"]]})
        rules = alt.Chart(lines).mark_rule(strokeDash=[5, 4]).encode(y="Thickness:Q", color="Line:N")
        st.altair_chart(chart.mark_line(color="#9ba9b5") + points + rules, width="stretch")
        currents = view.melt(id_vars=["Sequence"], value_vars=["Current_ASD", "Next_Current_ASD"],
                             var_name="Series", value_name="Current")
        st.altair_chart(alt.Chart(currents).mark_line(point=True).encode(
            x=alt.X("Sequence:Q", title="輸送序號"), y=alt.Y("Current:Q", title="本筆電流與下一批建議 (ASD)",
            scale=alt.Scale(zero=False)), color="Series:N", tooltip=["Sequence:Q", "Series:N", "Current:Q"]), width="stretch")
        st.caption("圖表顯示最近 100 筆；下表顯示最近 20 筆。完整輸送紀錄與本次參數可下載。模型使用來源採樣時間計算特徵。")
        st.dataframe(history.tail(20).iloc[::-1], hide_index=True, width="stretch")
        exported = history.assign(Seed=state["seed"], Source=state["source"], **state["settings"], **state["limits"])
        st.download_button("下載動態模擬紀錄 CSV", exported.to_csv(index=False).encode("utf-8-sig"),
                           "live_simulation.csv", "text/csv", on_click="ignore")
    live_dashboard()


def main():
    st.set_page_config(page_title="銅箔製程品質分析", page_icon="📊", layout="wide")
    with st.sidebar:
        st.title("銅箔品質分析")
        page = st.radio("功能分頁", ["資料視覺化", "異常預測", "SPC 管制圖", "R2R 控制模擬", "即時資料模擬"])
        st.caption("Random Forest · SPC · R2R")
    if page != "即時資料模擬" and "live_simulation" in st.session_state:
        st.session_state["live_simulation"]["running"] = False
    if page == "資料視覺化":
        visualization_page()
    elif page == "異常預測":
        prediction_page()
    elif page == "SPC 管制圖":
        spc_page()
    elif page == "R2R 控制模擬":
        r2r_page()
    else:
        live_simulation_page()


if __name__ == "__main__":
    main()
