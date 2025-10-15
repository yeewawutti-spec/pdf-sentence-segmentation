# -*- coding: utf-8 -*-
"""
PDF Sentence Cleaner — Orange Theme Edition
เวอร์ชันตรงกับโค้ดของคุณ (draft_no_3) + ตกแต่งสีส้มอบอุ่น
"""

import os
import io
import re
import fitz
import nltk
import spacy
import pandas as pd
import unicodedata
import streamlit as st
import time

# =============================
# 🔧 Setup
# =============================
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except Exception:
    nlp = None
    _SPACY_OK = False

# =============================
# 🧠 Functions
# =============================

def extract_text_from_pdf_positional_auto(file_path, header_margin=60.0, footer_margin=60.0, left_margin=0.0, right_margin=0.0,
                                          remove_bold_all=True):
    """Extract text by removing header/footer and bold spans automatically."""
    def _is_inside(bbox, left, top, right, bottom):
        x0, y0, x1, y1 = bbox
        return (y1 <= bottom and y0 >= top and x0 >= left and x1 <= right)

    def _span_is_bold(span):
        font = span.get("font", "") or ""
        flags = int(span.get("flags", 0) or 0)
        return bool(re.search(r"(?i)(bold|black|heavy|semibold|demi)", font)) or bool(flags & 2)

    pages_text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            top, bottom = page.rect.y0 + header_margin, page.rect.y1 - footer_margin
            left, right = page.rect.x0 + left_margin, page.rect.x1 - right_margin
            pdict = page.get_text("dict")
            line_buf = []
            for block in pdict.get("blocks", []):
                if block.get("type", 0) != 0: continue
                for line in block.get("lines", []):
                    spans_in_line = []
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        bbox = span.get("bbox", None)
                        if not bbox or not text: continue
                        if not _is_inside(bbox, left, top, right, bottom): continue
                        if remove_bold_all and _span_is_bold(span): continue
                        spans_in_line.append(text)
                    if spans_in_line:
                        merged = re.sub(r"[ \t]+", " ", "".join(spans_in_line)).strip()
                        if merged: line_buf.append(merged)
            pages_text.append("\n".join(line_buf))
    return "\n".join(pages_text)


def help_ie(s: str) -> str:
    s = re.sub(r"\bi\.e\.(?=\s*\w)(?!\s*,)", "i.e.,", s, flags=re.I)
    s = re.sub(r"\be\.g\.(?=\s*\w)(?!\s*,)", "e.g.,", s, flags=re.I)
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
        out.extend([p.strip() for p in re.split(r"[•]", s) if p.strip()])
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
        if ss.startswith(("table of contents", "contents")) or re.search(r"\.{6,}", ss) or re.search(r"-\s*-\s*-", ss):
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed

def _normalize_line(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()
def _is_allcaps_token(t):
    t = t.strip(",.;:()[]{}&/-")
    return bool(t) and (t.isdigit() or (t.isupper() and any(c.isalpha() for c in t)))
def _is_allcaps_line(s):
    toks = [w for w in _normalize_line(s).split() if w.strip(",.;:()[]{}&/-")]
    return len(toks) > 2 and all(_is_allcaps_token(w) for w in toks)
def remove_head(sents):
    out, removed = [], []
    for s in sents:
        s = _normalize_line(s)
        orig = s
        if _is_allcaps_line(s): removed.append(orig); continue
        m = re.match(r"^([A-Z0-9 ,/&\-\(\)]+[:\,])\s*(.*)$", s)
        if m:
            head, rest = m.group(1).rstrip(",: "), m.group(2).strip()
            if _is_allcaps_line(head): removed.append(head); s = rest
        s = re.sub(r"\b(?:[A-Z0-9&/\-]{3,}(?: [A-Z0-9&/\-]{2,})+)\b", "", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s and not _is_allcaps_line(s): out.append(s)
        else: removed.append(orig)
    return out, removed

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
    df = pd.DataFrame({"Sentence Index": range(1, len(kept)+1), "Sentence": kept})
    return df

# =============================
# 🎨 Streamlit UI
# =============================
st.set_page_config(page_title="PDF Sentence Cleaner", page_icon="📘", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#E67E22;'>📘 PDF Sentence Cleaner — By 802 Squad</h1>
<p style='text-align:center; color:#B9770E; font-size:17px;'>
✨ อัปโหลดไฟล์ PDF เพื่อทดลองระบบตัดประโยคอัตโนมัติแปลงข้อความเป็น Excel ภายในไม่กี่วินาที
</p>
<hr style='border:1px solid #FAD7A0'>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์ PDF", type=["pdf"])

if uploaded_file is not None:
    base_name = os.path.splitext(uploaded_file.name)[0]
    pdf_path = f"/tmp/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    with st.spinner("⏳ กำลังประมวลผลโปรดรอสักครู่..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        df_output = pdf_to_clean_sentences(pdf_path, out_prefix=base_name)

    output = io.BytesIO()
    df_output.to_excel(output, index=False)
    output.seek(0)
    xlsx_name = f"{base_name}_cleaned.xlsx"

    st.success(f"✅ ประมวลผลเสร็จสิ้น! ({base_name})")
    st.download_button(
        f"📤 ดาวน์โหลดไฟล์ Excel ({xlsx_name})",
        data=output.getvalue(),
        file_name=xlsx_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{xlsx_name}"
    )

    with st.expander("🔍 ดูตัวอย่าง (20 บรรทัดแรก)"):
        st.dataframe(df_output.head(20), use_container_width=True, hide_index=True)

st.sidebar.markdown("### 🧾 System Description")
st.sidebar.info("ระบบนี้ทำหน้าที่ประมวลผลและแยกประโยคจากไฟล์ PDF โดยอัตโนมัติ พร้อมดำเนินการทำความสะอาดข้อความ เช่น การลบส่วนหัวและส่วนท้ายของเอกสาร (Header/Footer) การลบข้อความตัวหนา รวมถึงการตัดประโยคอย่างละเอียดตามโค้ดต้นฉบับที่กำหนดไว้")

st.sidebar.markdown("---")
st.sidebar.markdown("""
🎓 ACCBA’65 <br>
📅 2025 | Chiang Mai University <br>
👩🏻‍💻 <b>Developer:</b> 802 Squad
💼 Faculty of Accountancy and Business Administration <br>
""", unsafe_allow_html=True)


st.markdown("<hr><p style='text-align:center; color:#BA4A00;'>© 2025  | Developed by 802 Squad</p>", unsafe_allow_html=True)
st.markdown("""
<style>
/* ===== 🎨 พื้นหลัง Mocha Brown ===== */
.stApp {
    background: linear-gradient(180deg, #EDE0D4 0%, #E6CCB2 40%, #F3EDE4 100%);
    color: #3B2F2F; /* สีตัวอักษรหลัก */
    font-family: "Segoe UI", sans-serif;
}

/* ===== หัวข้อหลัก ===== */
h1, h2, h3 {
    color: #1E1E1E !important;  /* สีดำ */
    font-weight: 800;
}

/* ===== คำอธิบายใต้หัวข้อ ===== */
.subtitle {
    color: #2E8B57; /* สีเขียวเข้ม */
    font-size: 1.05rem;
    font-weight: 500;
    margin-top: -10px;
    margin-bottom: 25px;
}

/* ===== เส้นคั่น ===== */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #A0522D 0%, #C19A6B 100%);
    border-radius: 2px;
}

/* ===== Upload Zone ===== */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.85);
    border: 2px dashed #B8860B;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0px 2px 10px rgba(139, 69, 19, 0.15);
}

/* ===== ปุ่ม Browse ===== */
button[kind="secondary"] {
    background: linear-gradient(90deg, #E67300, #FF944D);
    color: white !important;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    transition: all 0.3s ease;
}
button[kind="secondary"]:hover {
    background: linear-gradient(90deg, #FF944D, #FFB066);
    transform: scale(1.03);
}

/* ===== Footer ===== */
footer, .footer {
    color: #5A3E2B;
    text-align: center;
    font-weight: 600;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ===== ส่วนหัว + คำอธิบาย =====
st.markdown("<h1>📘 PDF Sentence Cleaner — By 802 Squad</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>✨ อัปโหลดไฟล์ PDF เพื่อทดลองระบบตัดประโยคอัตโนมัติและแปลงข้อความเป็น Excel </p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


















