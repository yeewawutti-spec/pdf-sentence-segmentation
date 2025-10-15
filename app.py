import streamlit as st
import fitz  # PyMuPDF
import re, unicodedata
import nltk
import spacy
import pandas as pd
from io import BytesIO

# โหลดโมดูล NLP
nltk.download('punkt', quiet=True)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

# ฟังก์ชันดึงข้อความจาก PDF (ตัด header/footer)
def extract_text_from_pdf(file_bytes, header_margin=60, footer_margin=60):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = []
    for page in doc:
        top = page.rect.y0 + header_margin
        bottom = page.rect.y1 - footer_margin
        blocks = page.get_text("blocks")
        page_text = "\n".join([b[4] for b in blocks if b[1] >= top and b[3] <= bottom])
        texts.append(page_text)
    return "\n".join(texts)

def tokenize_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def clean_sentences(sentences):
    cleaned = []
    for s in sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            cleaned.append(s)
    return cleaned

def process_pdf(uploaded_file):
    bytes_data = uploaded_file.read()
    text = extract_text_from_pdf(bytes_data)
    sentences = tokenize_sentences(text)
    sentences = clean_sentences(sentences)
    df = pd.DataFrame({"Sentence Index": range(1, len(sentences)+1), "Sentence": sentences})
    return df

# ส่วนติดต่อผู้ใช้
st.set_page_config(page_title="PDF Sentence Segmenter", layout="wide")
st.title("📄 PDF Sentence Segmentation Web App")

uploaded_file = st.file_uploader("อัปโหลดไฟล์ PDF", type=["pdf"])
if uploaded_file:
    st.success("อัปโหลดสำเร็จ ✅ กำลังประมวลผล...")
    df = process_pdf(uploaded_file)
    st.dataframe(df.head(20))
    
    # ปุ่มดาวน์โหลด
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 ดาวน์โหลดไฟล์ CSV",
        data=csv,
        file_name="sentences_output.csv",
        mime="text/csv",
    )
