import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import tempfile
from largebert import prediction  # Assumes 'prediction' is exposed in largebert.py

st.set_page_config(page_title="CUI File Inference", layout="centered")
st.title("📄 Upload a PDF or CSV for CUI Prediction")

uploaded_file = st.file_uploader("Choose a .csv or .pdf file", type=["csv", "pdf"])

if uploaded_file:
    filetype = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filetype}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Run Inference"):
        if filetype == "csv":
            df = pd.read_csv(tmp_path)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                results = []
                for row in df["text"]:
                    _, label, conf = prediction(row)
                    results.append({"text": row[:100], "prediction": label, "confidence": f"{conf:.2f}%"})
                st.dataframe(pd.DataFrame(results))
        elif filetype == "pdf":
            reader = PdfReader(tmp_path)
            full_text = "\n".join([p.extract_text() for p in reader if p.extract_text()])
            _, label, conf = prediction(full_text)
            st.write("### PDF Prediction")
            st.write(f"**Label:** {label}")
            st.write(f"**Confidence:** {conf:.2f}%")