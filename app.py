# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import csv
import math
import fitz  # PyMuPDF
import nltk
import spacy
import pandas as pd
import unicodedata

# =========================================
# ✅ ดาวน์โหลด resource ที่จำเป็น
# =========================================
nltk.download('punkt')
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# =========================================
# ⚙️ โหลด spaCy model
# =========================================
_SPACY_OK = True
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_OK = False
    nlp = None

# =========================================
# 🔧 ฟังก์ชันหลัก (เหมือนใน draft(no_3).py)
# =========================================

def extract_text_from_pdf_positional_auto(
    file_path: str,
    header_margin: float = 60.0,
    footer_margin: float = 60.0,
    left_margin: float = 0.0,
    right_margin: float = 0.0,
    granularity: str = "blocks",
):
    pages_text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            top = page.rect.y0 + header_margin
            bottom = page.rect.y1 - footer_margin
            left = page.rect.x0 + left_margin
            right = page.rect.x1 - right_margin

            if granularity == "words":
                words = page.get_text("words")
                kept_words = [
                    w for w in words if (w[1] >= top and w[3] <= bottom and w[0] >= left and w[2] <= right)
                ]
                kept_words.sort(key=lambda w: (round(w[1], 1), w[0]))
                lines, current, last_y = [], [], None
                for w in kept_words:
                    y = round(w[1], 1)
                    if last_y is None or abs(y - last_y) <= 1.0:
                        current.append(w[4])
                    else:
                        if current:
                            lines.append(" ".join(current))
                        current = [w[4]]
                    last_y = y
                if current:
                    lines.append(" ".join(current))
                page_text = "\n".join(lines)
            else:
                blocks = page.get_text("blocks")
                kept_blocks = []
                for b in blocks:
                    x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                    if (y1 <= bottom and y0 >= top and x0 >= left and x1 <= right):
                        kept_blocks.append(text)
                page_text = "\n".join(kept_blocks)
            pages_text.append(page_text)
    return "\n".join(pages_text)


def help_ie(s: str) -> str:
    s = re.sub(r'\bi\.e\.(?=\s*\w)(?!\s*,)', 'i.e.,', s, flags=re.IGNORECASE)
    s = re.sub(r'\be\.g\.(?=\s*\w)(?!\s*,)', 'e.g.,', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNo\.(?=\s*\w)(?!\s*,)', 'No.,', s, flags=re.IGNORECASE)
    return s


def tokenize_sentences(text: str):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)


def remove_extra_whitespace(sentences):
    return [re.sub(r"\s+", " ", s).strip() for s in sentences]


def remove_URLs(sentences):
    pat = re.compile(r'(https?://\S+|www\.\S+)')
    processed = []
    for s in sentences:
        match = pat.search(s)
        if match and match.end() == len(s):
            s = pat.sub("URL.", s)
        else:
            s = pat.sub("URL", s)
        processed.append(s)
    return processed


def remove_special_chars(sentences):
    table = str.maketrans("", "", "•*#+|")
    return [s.translate(table) for s in sentences]


def remove_table_of_contents(sentences):
    removed, kept = [], []
    for s in sentences:
        ss = s.strip()
        if (ss.lower().startswith("table of contents")
            or ss.lower().startswith("contents")
            or re.search(r"\.{6,}", ss)
            or re.search(r"-\s*-\s*-", ss)):
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed


def _normalize_line(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_allcaps_token(tok: str) -> bool:
    t = tok.strip(",.;:()[]{}&/-")
    if not t or t == "I":
        return False
    return t.isdigit() or (t.isupper() and any(c.isalpha() for c in t))


def _is_allcaps_line(s: str) -> bool:
    s = _normalize_line(s)
    tokens = [w for w in s.split() if w.strip(",.;:()[]{}&/-")]
    if len(tokens) <= 2:
        return False
    return all(_is_allcaps_token(w) for w in tokens)


def remove_head(sentences):
    out, removed = [], []
    for s in sentences:
        s = _normalize_line(s)
        orig = s
        if _is_allcaps_line(s):
            removed.append(orig)
            continue
        m = re.match(r'^([A-Z0-9 ,/&\-\(\)]+[:\,])\s*(.*)$', s)
        if m:
            head, rest = m.group(1).rstrip(",: "), m.group(2).strip()
            if _is_allcaps_line(head):
                removed.append(head)
                s = rest
        if s and _is_allcaps_line(s):
            removed.append(orig)
            continue
        s = re.sub(r'\b(?:[A-Z0-9&/\-]{3,}(?: [A-Z0-9&/\-]{2,})+)\b', '', s)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        if s and _is_allcaps_line(s):
            removed.append(orig)
            continue
        if s:
            out.append(s)
        else:
            removed.append(orig)
    return out, removed


def calculate_digit_percentage(s: str) -> float:
    if not s:
        return 0.0
    total = len(s)
    digits = sum(ch.isdigit() for ch in s)
    return 100.0 * digits / max(1, total)


def remove_too_much_digit(sentences, threshold=30.0):
    kept, removed = [], []
    for s in sentences:
        if calculate_digit_percentage(s) >= threshold:
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed


def is_full_sentence_spacy(s: str) -> bool:
    if _SPACY_OK and nlp is not None:
        doc = nlp(s)
        has_subj = any(tok.dep_ in ("nsubj", "nsubjpass", "csubj") for tok in doc)
        has_pred = any(tok.pos_ in ("VERB", "AUX") or tok.dep_ == "ROOT" for tok in doc)
        return has_subj and has_pred
    has_verb = bool(re.search(r"\b(is|are|was|were|be|been|being|has|have|had|will|shall|would|should|can|could|may|might|must|[a-zA-Z]+ed|[a-zA-Z]+ing)\b", s, flags=re.I))
    has_noun = bool(re.search(r"\b(I|you|he|she|it|we|they|the|a|an|this|that|[A-Z][a-z]+)\b", s))
    return has_verb and has_noun


def remove_phrase(sentences):
    kept, removed = [], []
    for s in sentences:
        if is_full_sentence_spacy(s):
            kept.append(s)
        else:
            removed.append(s)
    return kept, removed


def remove_long_short_sentence(sentences, min_words=6):
    kept, removed = [], []
    for s in sentences:
        wc = len(s.split())
        if wc < min_words:
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed


def pdf_to_clean_sentences(pdf_path: str, out_prefix: str = "result"):
    removed_all = []
    raw = extract_text_from_pdf_positional_auto(pdf_path)
    raw = help_ie(raw)
    sents = tokenize_sentences(raw)

    kept, rem = remove_table_of_contents(sents)
    removed_all += rem
    kept, rem = remove_head(kept)
    removed_all += rem
    kept, rem = remove_long_short_sentence(kept)
    removed_all += rem
    kept, rem = remove_phrase(kept)
    removed_all += rem
    kept, rem = remove_too_much_digit(kept)
    removed_all += rem
    kept = remove_special_chars(remove_URLs(remove_extra_whitespace(kept)))

    out_dir = "output_data"
    os.makedirs(out_dir, exist_ok=True)
    out_keep = os.path.join(out_dir, f"{out_prefix}_sentences_output.csv")
    pd.DataFrame(kept, columns=["Sentence"]).to_csv(out_keep, index=False, encoding="utf-8")

    return kept, removed_all, out_keep

# =========================================
# 🌐 Streamlit UI
# =========================================
st.set_page_config(page_title="PDF Sentence Pipeline", page_icon="🧩", layout="wide")
st.title("🧩 PDF Sentence Pipeline (Advanced Version)")
st.write("Upload a PDF → Extract & Clean Sentences (Header/Footer/ALLCAPS/Bold/TOC filtering)")

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

