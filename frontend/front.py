import streamlit as st
import requests
import pandas as pd
import os

# Determine API URL: when running in Docker use service name `backend`
API_URL = os.getenv("API_URL", "http://backend:8000" if os.path.exists("/.dockerenv") else "http://127.0.0.1:8000")

st.set_page_config(page_title="Sentiment Classification", layout="wide")

st.title("Tunisian Arabic Sentiment — Demo")

with st.sidebar:
    st.header("Options")
    page = st.radio("Page", ["Home", "Single Text", "Batch CSV"])

if page == "Home":
    st.markdown("This demo calls the backend API to get sentiment predictions for Tunisian Arabic text.")

if page == "Single Text":
    text = st.text_area("Enter text", value="مثال على نص تونسي")
    if st.button("Predict"):
        with st.spinner("Requesting prediction..."):
            try:
                r = requests.post(f"{API_URL}/predict", json={"text": text})
                r.raise_for_status()
                out = r.json()
                st.json(out)
            except Exception as e:
                st.error(f"Error: {e}")

if page == "Batch CSV":
    uploaded = st.file_uploader("Upload CSV with a `text` column", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Predict Batch"):
            with st.spinner("Sending CSV..."):
                try:
                    files = {"file": (uploaded.name, uploaded, "text/csv")}
                    r = requests.post(f"{API_URL}/predict_csv", files=files)
                    r.raise_for_status()
                    st.success("Done — download the results below")
                    st.download_button("Download results", r.content, file_name=f"predictions_{uploaded.name}", mime="text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")
