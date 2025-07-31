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

# 如果有 database.csv，就讀取
if os.path.exists('database.csv'):
    df = pd.read_csv('database.csv', index_col=None)
else:
    df = None

# 建立側邊欄
with st.sidebar:
    st.image("logo_light.png")
    st.title("Delta Auto Machine Learning Center")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download", "Predict"])
    st.info("這個介面將協助你建構\n\n自動化機器學習系統\n\n來完成資料分析")

# Upload
if choice == "Upload":
    st.title("上傳你的數據，開始建模")
    file = st.file_uploader("上傳你的數據", type=['csv'])
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('database.csv', index=None)  # 寫成本地檔案，供下次使用
        st.dataframe(df)

# Profiling
elif choice == "Profiling":
    st.title("自動分析資料結構")
    if df is not None:
        profile_report = ProfileReport(df, minimal=True, tsmode=True, sensitive=True, explorative=True)
        st_profile_report(profile_report)
    else:
        st.warning("請先上傳數據或確保 'database.csv' 文件存在。")

# ML
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

# Download
elif choice == "Download":
    # 若 'best_model.pkl' 存在，就提供下載
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

# Predict
elif choice == "Predict":
    st.title("使用模型來預測新數據")
    uploaded_model = 'best_model'
    uploaded_data = st.file_uploader("上傳要預測的 .csv 檔案", type='csv')

    if os.path.exists(uploaded_model + '.pkl') and uploaded_data is not None:

        model = load_model(uploaded_model)

        # 讀取上傳 CSV
        test_df = pd.read_csv(uploaded_data)

        st.write("Uploaded data columns:", test_df.columns.tolist())
        st.write("Required columns:", ['Date', 'Equipment No', 'NTC1_Res', 'NTC1_Temp', 'WR_UV_Res', 'WR_VW_Res', 'WR_WU_Res', 'Offset_WR_UV_Res', 'Offset_WR_VW_Res', 'Offset_WR_WU_Res', 'Un_UV', 'Un_VW', 'Un_WU', 'Surge_U-VW_AreaDiff', 'Surge_V-WU_AreaDiff', 'Surge_W-UV_AreaDiff', 'Surge_U-VW_L-Vale', 'Surge_V-WU_L-Vale', 'Surge_W-UV_L-Vale'])

        # 檢查並處理無效或極端值
        if not test_df.replace([np.inf, -np.inf], np.nan).dropna().empty:
            st.warning("上傳的數據包含無效或極端值，請檢查並修正後再試。")
        else:
            prediction = predict_model(model, data=test_df)
            st.dataframe(prediction)

        # 假設真實標籤是 'Test_Result'，且可能是 'Pass'/'Fail'
        # 若資料中沒有真實標籤(測試階段只想看預測)，可以不算分數
        if 'Test_Result' in prediction.columns and 'prediction_label' in prediction.columns:
            accuracy = accuracy_score(prediction['Test_Result'], prediction['prediction_label'])
            # 指定正樣本
            precision = precision_score(prediction['Test_Result'], prediction['prediction_label'], pos_label='Pass')
            recall = recall_score(prediction['Test_Result'], prediction['prediction_label'], pos_label='Pass')
            f1 = f1_score(prediction['Test_Result'], prediction['prediction_label'], pos_label='Pass')
            # ROC AUC 需要 0/1 整數或機率
            y_true_binary = (prediction['Test_Result'] == 'Pass').astype(int)
            y_pred_binary = (prediction['prediction_label'] == 'Pass').astype(int)
            roc_auc = roc_auc_score(y_true_binary, y_pred_binary)

            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")
            st.write(f"ROC AUC: {roc_auc:.2f}")
        else:
            st.warning("預測結果裡沒有找到 'Test_Result' 或 'prediction_label' 欄位，無法計算評估指標。")