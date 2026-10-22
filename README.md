# 銅箔製程品質分析

`05_app.py` 是 Streamlit 主程式，左側提供「資料視覺化」與「異常預測」兩頁。
視覺化包含時間趨勢、直方圖、箱型圖、班別異常比例、散佈圖與相關性熱圖。
預測頁接收七個原始單位製程數值與採樣時間，自動套用訓練時的前處理與模型門檻，並提供 CSV 下載。

## 本機執行

建議 Python 3.10（訓練版本為 3.10.21）。在專案根目錄執行：

```powershell
python -m pip install -r requirements.txt
python -m streamlit run 05_app.py
```

若使用現有 Windows 虛擬環境：

```powershell
.\.venv\Scripts\python.exe -m streamlit run 05_app.py
```

## 部署

將下列檔案連同目錄結構放入部署儲存庫；不需要上傳 `.venv` 或 notebook：

```text
05_app.py
requirements.txt
data/prepared/copper_foil_cleaned.csv
models/random_forest_model.joblib
models/random_forest_metadata.json
models/random_forest_preprocessing.json
```

部署服務安裝 `requirements.txt`，入口指定 `05_app.py`；如需自訂啟動命令：

```bash
python -m streamlit run 05_app.py --server.address=0.0.0.0 --server.port=8501
```

連接埠依平台設定調整。三個模型檔案必須來自同一次訓練匯出，僅使用可信來源的 joblib。
更新模型時重新執行 `03_random_forest_training.ipynb`，一起部署模型與設定檔。

此專案使用合成資料，模型門檻從設定檔載入（目前 0.71）。模型分數未經校準；
原測試集只有正常資料，無法據此評估實際異常召回率。
