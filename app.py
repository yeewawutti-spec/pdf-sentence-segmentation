import streamlit as st
import os
import re
import csv
import fitz  # PyMuPDF
import nltk
import spacy
import pandas as pd
import unicodedata

# ==================================================
# 🔧 โหลดโมดูลภาษา (แก้ปัญหา LookupError ของ NLTK)
# ==================================================
nltk.download('punkt')
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ==================================================
# ⚙️ โหลดโมเดล spaCy
# ==================================================
_SPACY_OK = True
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_OK = False
    nlp = None


# ==================================================
# 🧠 Utility Functions (ฟังก์ชันหลักจาก pipeline ของคุณ)
# ==================================================

def extract_text_from_pdf_positional_auto(file_path, header_margin=60.0, footer_margin=60.0):
    """Extract text from PDF and remove header/footer by margin."""
    pages_text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            top = page.rect.y0 + header_margin
            bottom = page.rect.y1 - footer_margin
            blocks = page.get_text("blocks")
            kept_blocks = []
            for b in blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                if (y1 <= bottom and y0 >= top):
                    kept_blocks.append(text)
            pages_text.append("\n".join(kept_blocks))
    return "\n".join(pages_text)


def tokenize_sentences(text):
    """Split text into sentences using NLTK."""
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)


def remove_extra_whitespace(sentences):
    return [re.sub(r"\s+", " ", s).strip() for s in sentences]


def remove_URLs(sentences):
    pat = re.compile(r'https?://\S+|www\.\S+')
    return [pat.sub("", s) for s in sentences]


def remove_special_chars(sentences):
    table = str.maketrans("", "", "•*#$+|")
    return [s.translate(table) for s in sentences]


def remove_long_short_sentence(sentences, min_words=6):
    kept, removed = [], []
    for s in sentences:
        wc = len(s.split())
        if wc < min_words:
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed


# ==================================================
# 🚀 Pipeline หลัก
# ==================================================
def pdf_to_clean_sentences(pdf_path, out_prefix="output"):
    """Main cleaning pipeline"""
    raw = extract_text_from_pdf_positional_auto(pdf_path)
    sents = tokenize_sentences(raw)
    kept, rem = remove_long_short_sentence(sents)
    kept = remove_special_chars(remove_URLs(remove_extra_whitespace(kept)))
    removed = remove_special_chars(remove_URLs(remove_extra_whitespace(rem)))

    out_dir = "output_data"
    os.makedirs(out_dir, exist_ok=True)
    out_keep = os.path.join(out_dir, f"{out_prefix}_sentences_output.csv")
    pd.DataFrame(kept, columns=["Sentence"]).to_csv(out_keep, index=False, encoding="utf-8")
    return kept, removed, out_keep


# ==================================================
# 🌐 Streamlit UI
# ==================================================
st.set_page_config(page_title="PDF Sentence Pipeline", page_icon="🧩", layout="wide")
st.title("🧩 PDF Sentence Pipeline (by Punyawee)")
st.write("Upload a PDF → Extract & Clean Sentences automatically 🧠")

uploaded_file = st.file_uploader("📄 Upload PDF file", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("⏳ Processing your PDF file, please wait...")
    kept, removed, output_path = pdf_to_clean_sentences("temp.pdf", out_prefix="uploaded")

    st.success(f"✅ Done! Extracted {len(kept)} cleaned sentences.")
    st.dataframe(pd.DataFrame(kept[:50], columns=["Preview (first 50 sentences)"]))

    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Clean Sentences (CSV)", f, file_name="clean_sentences.csv")

else:
    st.warning("📂 Please upload a PDF file to start.")
