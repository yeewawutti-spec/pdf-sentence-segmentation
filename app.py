# -*- coding: utf-8 -*-
"""
PDF Sentence Cleaner — Streamlit App
เวอร์ชันตรงกับโค้ดต้นฉบับของคุณ (draft_no_3) 
เพิ่มระบบ:
✅ auto rename ตามชื่อไฟล์ PDF ที่อัปโหลด
✅ ป้องกัน StreamlitDuplicateElementId
✅ รองรับ header/footer, bold remover, strict ALLCAPS, และ export xlsx
"""

import os
import io
import re
import csv
import math
import fitz  # PyMuPDF
import nltk
import spacy
import pandas as pd
import unicodedata
import streamlit as st

# =========================
# 🔧 Setup
# =========================
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_SPACY_OK = True
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_OK = False
    nlp = None

# =========================
# 🧠 Core Functions
# =========================

def extract_text_from_pdf_positional_auto(
    file_path: str,
    header_margin: float = 60.0,
    footer_margin: float = 60.0,
    left_margin: float = 0.0,
    right_margin: float = 0.0,
    granularity: str = "spans",
    remove_bold_all: bool = True,
    remove_bold_lines: bool = False,
):
    """Extract text by removing header/footer and bold spans automatically."""
    def _is_inside(bbox, left, top, right, bottom):
        x0, y0, x1, y1 = bbox
        return (y1 <= bottom and y0 >= top and x0 >= left and x1 <= right)

    def _span_is_bold(span) -> bool:
        font = span.get("font", "") or ""
        flags = int(span.get("flags", 0) or 0)
        by_name = bool(re.search(r"(?i)(bold|black|heavy|semibold|demi)", font))
        by_flags = bool(flags & 2)
        return by_name or by_flags

    pages_text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            top = page.rect.y0 + header_margin
            bottom = page.rect.y1 - footer_margin
            left = page.rect.x0 + left_margin
            right = page.rect.x1 - right_margin

            pdict = page.get_text("dict")
            line_buf = []

            for block in pdict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    spans_in_line = []
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        bbox = span.get("bbox", None)
                        if not bbox or not text:
                            continue
                        if not _is_inside(bbox, left, top, right, bottom):
                            continue
                        if remove_bold_all and _span_is_bold(span):
                            continue
                        spans_in_line.append(text)
                    if spans_in_line:
                        merged = re.sub(r"[ \t]+", " ", "".join(spans_in_line)).strip()
                        if merged:
                            line_buf.append(merged)
            page_text = "\n".join(line_buf)
            pages_text.append(page_text)
    return "\n".join(pages_text)

# --- Utilities ---
def help_ie(s: str) -> str:
    s = re.sub(r"\bi\.e\.(?=\s*\w)(?!\s*,)", "i.e.,", s, flags=re.IGNORECASE)
    s = re.sub(r"\be\.g\.(?=\s*\w)(?!\s*,)", "e.g.,", s, flags=re.IGNORECASE)
    return s

def tokenize_sentences(text: str):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def remove_extra_whitespace(sents): return [re.sub(r"\s+", " ", s).strip() for s in sents]
def remove_URLs(sents): return [re.sub(r"https?://\S+|www\.\S+", "URL", s) for s in sents]
def remove_special_chars(sents): return [s.translate(str.maketrans("", "", "•*#+|")) for s in sents]
def split_bullet(sents):
    out = []
    for s in sents:
        parts = [p.strip() for p in re.split(r"[•]", s) if p.strip()]
        out.extend(parts)
    return out
def split_number_bullet(sents):
    out = []
    for s in sents:
        toks = s.split()
        idxs = [i for i, w in enumerate(toks) if re.fullmatch(r"\(\d+\)", w)]
        if len(idxs) >= 2:
            start = 0
            for i in range(1, len(idxs)):
                if idxs[i] - idxs[i - 1] >= 6:
                    out.append(" ".join(toks[start:idxs[i]]).strip())
                    start = idxs[i]
            out.append(" ".join(toks[start:]).strip())
        else:
            out.append(s)
    return [s for s in out if s]

def remove_table_of_contents(sents):
    kept, removed = [], []
    for s in sents:
        ss = s.strip().lower()
        if (
            ss.startswith("table of contents")
            or ss.startswith("contents")
            or re.search(r"\.{6,}", ss)
            or re.search(r"-\s*-\s*-", ss)
        ):
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed

# --- Strict ALLCAPS remover ---
def _normalize_line(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()
def _is_allcaps_token(t):
    t = t.strip(",.;:()[]{}&/-")
    if not t or t == "I": return False
    return t.isdigit() or (t.isupper() and any(c.isalpha() for c in t))
def _is_allcaps_line(s):
    toks = [w for w in _normalize_line(s).split() if w.strip(",.;:()[]{}&/-")]
    return len(toks) > 2 and all(_is_allcaps_token(w) for w in toks)
def remove_head(sents):
    out, removed = [], []
    for s in sents:
        s = _normalize_line(s)
        orig = s
        if _is_allcaps_line(s):
            removed.append(orig); continue
        m = re.match(r"^([A-Z0-9 ,/&\-\(\)]+[:\,])\s*(.*)$", s)
        if m:
            head, rest = m.group(1).rstrip(",: "), m.group(2).strip()
            if _is_allcaps_line(head):
                removed.append(head); s = rest
        s = re.sub(r"\b(?:[A-Z0-9&/\-]{3,}(?: [A-Z0-9&/\-]{2,})+)\b", "", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s and not _is_allcaps_line(s):
            out.append(s)
        else:
            removed.append(orig)
    return out, removed

# --- Digit %, Sentence Length, Grammar ---
def calc_digit_pct(s): return 100 * sum(c.isdigit() for c in s) / max(1, len(s))
def remove_too_much_digit(sents, th=30):
    kept, rem = [], []
    for s in sents:
        (rem if calc_digit_pct(s) >= th else kept).append(s)
    return kept, rem
def remove_long_short_sentence(sents, minw=6, maxw=None):
    kept, rem = [], []
    for s in sents:
        wc = len(s.split())
        (rem if wc < minw or (maxw and wc > maxw) else kept).append(s)
    return kept, rem
def is_full_sentence_spacy(s):
    if _SPACY_OK and nlp is not None:
        doc = nlp(s)
        has_subj = any(tok.dep_ in ("nsubj", "nsubjpass", "csubj") for tok in doc)
        has_pred = any(tok.pos_ in ("VERB", "AUX") or tok.dep_ == "ROOT" for tok in doc)
        return has_subj and has_pred
    return bool(re.search(r"\b(is|are|was|were|has|have|had|will|should|can|may|[a-zA-Z]+ed|[a-zA-Z]+ing)\b", s, re.I))
def remove_phrase(sents):
    kept, rem = [], []
    for s in sents:
        (kept if is_full_sentence_spacy(s) else rem).append(s)
    return kept, rem

# --- Main Pipeline ---
def pdf_to_clean_sentences(pdf_path: str, out_prefix: str):
    removed_all = []
    raw = extract_text_from_pdf_positional_auto(pdf_path)
    raw = help_ie(raw)
    sents = tokenize_sentences(raw)

    kept, rem = remove_table_of_contents(sents); removed_all += rem
    kept, rem = remove_head(kept); removed_all += rem
    kept = split_bullet(kept); kept = split_number_bullet(kept)
    kept, rem = remove_long_short_sentence(kept); removed_all += rem
    kept, rem = remove_phrase(kept); removed_all += rem
    kept, rem = remove_too_much_digit(kept); removed_all += rem

    kept = remove_URLs(kept); kept = remove_special_chars(kept); kept = remove_extra_whitespace(kept)
    kept = [s for s in kept if s.strip()]
    removed_all = [s for s in removed_all if s.strip()]

    df = pd.DataFrame({"Sentence Index": range(1, len(kept)+1), "Sentence": kept})
    return df

# =========================
# 🌐 Streamlit UI
# =========================
st.set_page_config(page_title="PDF Sentence Cleaner", page_icon="📘", layout="wide")
st.title("📘 PDF Sentence Cleaner — By 802 Squad")

uploaded_file = st.file_uploader("อัปโหลดไฟล์ PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("กำลังประมวลผล..."):
        base_name = os.path.splitext(uploaded_file.name)[0]
        pdf_path = f"/tmp/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        df_output = pdf_to_clean_sentences(pdf_path, out_prefix=base_name)

        # ✅ Export to Excel
        output = io.BytesIO()
        df_output.to_excel(output, index=False)
        output.seek(0)

        xlsx_name = f"{base_name}_cleaned.xlsx"

        st.success(f"✅ เสร็จสิ้น — ตัดตามโค้ดของคุณทุกขั้นตอน ({base_name})")

        st.download_button(
            f"⬇️ ดาวน์โหลดไฟล์ผลลัพธ์ ({xlsx_name})",
            data=output.getvalue(),
            file_name=xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_{xlsx_name}"
        )

        with st.expander("🔍 ดูตัวอย่าง (20 บรรทัดแรก)"):
            st.dataframe(df_output.head(20))







