import streamlit as st
import pandas as pd
from model import load_data, train_models, get_shap_values
from shap_utils import plot_summary_figures

# 1. 页面标题
st.title("Unplanned Reoperation Risk Prediction")

# 2. 上传数据
uploaded_file = st.file_uploader("Upload your Excel data", type=["xls", "xlsx"])

if uploaded_file is not None:
    # 3. 加载数据
    df = load_data(uploaded_file)
    st.write(df.head())

    # 4. 训练模型
    results, (X_train, X_test, y_train, y_test), model = train_models(df)
    st.write(results)

    # 5. 获取SHAP值
    shap_values = get_shap_values(model, X_train, X_test)

    # 6. 绘制SHAP图
    plot_summary_figures(shap_values, X_test)

    st.success("SHAP plots generated successfully!")