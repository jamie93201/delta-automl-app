#AutoML 應用程式

這個專案提供一個 **Streamlit** 構建的網頁介面，可讓使用者透過幾個簡單步驟完成資料上傳、資料探索、模型訓練、模型下載與預測。內部使用 [PyCaret](https://pycaret.org/) 進行自動化機器學習，並使用 ydata‑profiling 生成資料報告。

## 功能簡介

本介面採用側邊選單形式，包含以下五個功能頁面：

1. **Upload**：上傳你的原始資料 (CSV 檔)，系統會顯示 DataFrame，並將檔案保存為 `database.csv` 以供後續使用。
2. **Profiling**：若已上傳資料，可利用 ydata‑profiling 快速生成資料結構分析報告，檢視各欄位的統計摘要與分佈情況。
3. **ML**：選擇目標特徵後，由 PyCaret 自動完成前處理、模型比較與最佳模型訓練，並將最佳模型儲存為 `best_model.pkl`。
4. **Download**：當已經訓練出模型時，可在此頁面下載 `best_model.pkl` 以便日後離線使用。
5. **Predict**：上傳一份新的 CSV 測試資料，系統會載入已訓練的模型進行預測，並顯示預測結果與分類評估指標（若資料中包含真實標籤）。

## 安裝與執行

1. 確認 Python 版本為 **3.8** 或以上。
2. 建議建立虛擬環境 (virtualenv 或 conda)。
3. 安裝必要套件：
   ```bash
   pip install streamlit pandas ydata-profiling streamlit-ydata-profiling scikit-learn pycaret[full] plotly
   ```
   > **注意：** PyCaret 安裝時可能會需要較多時間與依賴套件。
4. 將本專案的程式碼下載或 clone 至本地端，並確定 `app.py` 與 `README.md` 位於同一資料夾。
5. 在終端機執行下列指令啟動應用程式：
   ```bash
   streamlit run app.py
   ```
6. 開啟瀏覽器並依提示操作。

## 程式碼架構詳細說明

程式碼主要由以下幾個部分組成：

### 1. 套件載入

```python
import streamlit as st
import pandas as pd
import os
from operator import index
import ydata_profiling
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tempfile
import numpy as np
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model
import plotly.express as px
```

* `streamlit`：負責建構網頁介面。
* `pandas`：處理資料表格式。
* `os`：檢查檔案是否存在。
* `ydata_profiling` 與 `streamlit_ydata_profiling`：產生資料分析報告並嵌入於 Streamlit 頁面。
* `sklearn.metrics`：用來計算分類模型的評估指標。
* `numpy`：處理數值型資料與缺失值。
* `pycaret.classification`：提供自動化機器學習流程，包括資料前處理、模型比較與預測。
* `plotly.express`：在此程式碼中尚未使用，可用於製作互動式圖表。

### 2. 資料讀取

```python
if os.path.exists('database.csv'):
    df = pd.read_csv('database.csv', index_col=None)
else:
    df = None
```

啟動應用程式時，若專案目錄下已有 `database.csv`，則自動載入資料供後續分析；否則將 `df` 設為 `None`。

### 3. 介面佈局

```python
with st.sidebar:
    st.image("logo_light.png")
    st.title("Delta Auto Machine Learning Center")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download", "Predict"])
    st.info("這個介面將協助你建構\n\n自動化機器學習系統\n\n來完成資料分析")
```

使用 Streamlit 的 `sidebar` 區塊建立側邊欄，顯示 logo、標題及選單。使用者可透過 `st.radio` 選擇不同功能頁面。

### 4. Upload 頁面

```python
if choice == "Upload":
    st.title("上傳你的數據，開始建模")
    file = st.file_uploader("上傳你的數據", type=['csv'])
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('database.csv', index=None)
        st.dataframe(df)
```

* 利用 `st.file_uploader` 上傳 CSV 檔。
* 讀取後顯示在介面中，並存為 `database.csv`，方便後續功能重複使用同一份資料。

### 5. Profiling 頁面

```python
elif choice == "Profiling":
    st.title("自動分析資料結構")
    if df is not None:
        profile_report = ProfileReport(df, minimal=True, tsmode=True, sensitive=True, explorative=True)
        st_profile_report(profile_report)
    else:
        st.warning("請先上傳數據或確保 'database.csv' 文件存在。")
```

當已經有資料 (`df` 不為 `None`) 時，使用 ydata‑profiling 生成報告。`minimal`、`tsmode` 等參數用以簡化報告內容、支援時間序列及避開敏感資料等。報告透過 `st_profile_report` 嵌入在頁面中。

### 6. ML 頁面

```python
elif choice == "ML":
    if df is not None:
        chosen_target = st.selectbox('選擇目標特徵', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target, verbose=False)
            setup_df = pull()
            st.dataframe(setup_df)

            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)

            save_model(best_model, 'best_model')
            st.success("模型訓練完成並已儲存為 best_model.pkl")
    else:
        st.warning("請先上傳數據或確保 'database.csv' 文件存在。")
```

* 透過 `st.selectbox` 讓使用者指定目標欄位 (label)。
* `setup` 用來自動完成前處理，包括分割資料、類別編碼等；`pull()` 可將內部產生的設定表格提取出來顯示。
* `compare_models()` 會比較多種分類模型，並返回最佳模型；再次呼叫 `pull()` 取得比較結果。
* `save_model` 將最佳模型儲存為 `best_model.pkl`，便於後續預測或下載。

### 7. Download 頁面

```python
elif choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button(
                label='Download Model',
                data=f,
                file_name="best_model.pkl",
                mime="application/octet-stream"
            )
    else:
        st.warning("尚未找到 best_model.pkl，請先到 ML 頁面建立模型。")
```

檢查 `best_model.pkl` 是否存在，如果有則顯示下載按鈕。使用者可以將訓練好的模型下載到本地端備份。

### 8. Predict 頁面

```python
elif choice == "Predict":
    st.title("使用模型來預測新數據")
    uploaded_model = 'best_model'
    uploaded_data = st.file_uploader("上傳要預測的 .csv 檔案", type='csv')

    if os.path.exists(uploaded_model + '.pkl') and uploaded_data is not None:
        model = load_model(uploaded_model)
        test_df = pd.read_csv(uploaded_data)
        st.write("Uploaded data columns:", test_df.columns.tolist())
        st.write("Required columns:", [...略...])
        # 檢查無效或極端值
        if not test_df.replace([np.inf, -np.inf], np.nan).dropna().empty:
            st.warning("上傳的數據包含無效或極端值，請檢查並修正後再試。")
        else:
            prediction = predict_model(model, data=test_df)
            st.dataframe(prediction)
        # 若包含真實標籤，計算 accuracy、precision、recall、F1、ROC AUC
        ...
    else:
        st.warning("預測結果裡沒有找到 'Test_Result' 或 'prediction_label' 欄位，無法計算評估指標。")
```

* 先上傳要預測的資料，再確認模型檔案存在。
* 使用 `load_model` 載入模型，`predict_model` 執行預測並返回包含 `prediction_label` 與 `prediction_score` 的 DataFrame。
* 若使用者的資料中包含真實標籤 `Test_Result`，即會計算各種分類指標 (accuracy、precision、recall、F1、ROC AUC)。否則只顯示預測結果。

## 新手使用流程

1. **啟動應用程式**：執行 `streamlit run app.py`，瀏覽器會自動開啟應用頁面。
2. **上傳資料**：選擇左側 `Upload`，點擊「上傳你的數據」按鈕，選擇本地的 CSV 檔案。檔案上傳後，資料會顯示在畫面上。
3. **檢視資料報告**：切換至 `Profiling`，系統將自動產生詳細的資料分析報告，協助了解資料概況與缺失情形。
4. **建構模型**：前往 `ML`，從下拉選單中選擇要預測的目標欄位，按下 **Run Modelling**。系統將自動嘗試多種演算法，展示各模型的比較結果並保存最佳模型。
5. **下載模型**：若需要將模型另存，至 `Download` 點擊 **Download Model**，即可下載 `best_model.pkl`。
6. **執行預測**：切換到 `Predict`，先確認 `best_model.pkl` 是否存在，然後上傳新的 CSV 測試資料。系統會載入模型並輸出預測結果，若同時提供真實標籤會額外顯示各項評估指標。

## 注意事項

* 請確保訓練資料與預測資料使用相同的欄位名稱與資料格式，尤其是預測頁面所列出的必需欄位。
* 若資料中包含缺失值或極端值，建議先進行清理或填補，以避免影響模型效能。
* 預測資料的欄位請勿包含系統欄位名稱 (`prediction_label`、`prediction_score` 等)，以免衝突。

希望這份說明能幫助您快速上手自動化機器學習介面！如果遇到問題，歡迎在 GitHub 提出 Issue 以取得協助。
