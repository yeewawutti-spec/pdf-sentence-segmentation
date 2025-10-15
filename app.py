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
from io import BytesIO

# ========================= SETUP =========================
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

_SPACY_OK = True
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_OK = False
    nlp = None


# ========================= EXTRACT =========================
def extract_text_from_pdf_positional_auto(
    file_path: str,
    header_margin: float = 60.0,
    footer_margin: float = 60.0,
    left_margin: float = 0.0,
    right_margin: float = 0.0,
    granularity: str = "spans",
    remove_bold_all: bool = True,
    remove_bold_lines: bool = False,
    bold_regex: str = r"(?i)\b(bold|black|heavy|semibold|demi)\b",
    heading_bold_ratio: float = 0.7,
    heading_size_multiplier: float = 1.15,
) -> str:

    def _is_inside(bbox, left, top, right, bottom):
        x0, y0, x1, y1 = bbox
        return (y1 <= bottom and y0 >= top and x0 >= left and x1 <= right)

    def _span_is_bold(span):
        font = span.get("font", "") or ""
        flags = int(span.get("flags", 0) or 0)
        by_name = bool(re.search(bold_regex, font))
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
                        if not text or not bbox:
                            continue
                        if not _is_inside(bbox, left, top, right, bottom):
                            continue
                        if remove_bold_all and _span_is_bold(span):
                            continue
                        spans_in_line.append(text)
                    if spans_in_line:
                        merged = " ".join(spans_in_line)
                        merged = re.sub(r"\s+", " ", merged).strip()
                        if merged:
                            line_buf.append(merged)
            pages_text.append("\n".join(line_buf))

    return "\n".join(pages_text)


# ========================= UTILITIES =========================
def help_ie(s: str) -> str:
    s = re.sub(r'\bi\.e\.(?=\s*\w)(?!\s*,)', 'i.e.,', s, flags=re.IGNORECASE)
    s = re.sub(r'\be\.g\.(?=\s*\w)(?!\s*,)', 'e.g.,', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNo\.(?=\s*\w)(?!\s*,)',  'No.,',  s, flags=re.IGNORECASE)
    return s

def tokenize_sentences(text: str):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def remove_extra_whitespace(sentences):
    return [re.sub(r"\s+", " ", s).strip() for s in sentences]

def remove_URLs(sentences):
    pat = re.compile(r'(https?://\S+|www\.\S+)')
    processed_sentences = []
    for s in sentences:
        match = pat.search(s)
        if match and match.end() == len(s):
            processed_s = pat.sub("URL.", s)
        else:
            processed_s = pat.sub("URL", s)
        processed_sentences.append(processed_s)
    return processed_sentences

def remove_special_chars(sentences):
    table = str.maketrans("", "", "•*#+|")
    return [s.translate(table) for s in sentences]

def split_bullet(sentences):
    out = []
    for s in sentences:
        parts = [p.strip() for p in re.split(r"[•]", s) if p.strip()]
        out.extend(parts)
    return out

def split_number_bullet(sentences):
    out = []
    for s in sentences:
        tokens = s.split()
        idxs = [i for i, w in enumerate(tokens) if re.fullmatch(r"\(\d+\)", w)]
        if len(idxs) >= 2:
            start = 0
            for i in range(1, len(idxs)):
                if idxs[i] - idxs[i-1] >= 6:
                    out.append(" ".join(tokens[start:idxs[i]]).strip())
                    start = idxs[i]
            out.append(" ".join(tokens[start:]).strip())
        else:
            out.append(s)
    return [s for s in out if s]


# ========================= CLEANING =========================
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
    if not t:
        return False
    if t == "I":
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
        original = s
        if _is_allcaps_line(s):
            removed.append(original)
            continue
        m = re.match(r'^([A-Z0-9 ,/&\-\(\)]+[:\,])\s*(.*)$', s)
        if m:
            head, rest = m.group(1).rstrip(",: "), m.group(2).strip()
            if _is_allcaps_line(head):
                removed.append(head)
                s = rest
            else:
                s = original
        if s and _is_allcaps_line(s):
            removed.append(original)
            continue
        s = re.sub(r'\b(?:[A-Z0-9&/\-]{3,}(?: [A-Z0-9&/\-]{2,})+)\b', '', s)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        if s and _is_allcaps_line(s):
            removed.append(original)
            continue
        if s:
            out.append(s)
        else:
            removed.append(original)
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

def remove_long_short_sentence(sentences, min_words=6, max_words=None):
    kept, removed = [], []
    for s in sentences:
        wc = len(s.split())
        if wc < min_words or (max_words is not None and wc > max_words):
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed

def is_full_sentence_spacy(s: str) -> bool:
    if _SPACY_OK and nlp is not None:
        doc = nlp(s)
        has_subj = any(tok.dep_ in ("nsubj","nsubjpass","csubj","csubjpass") for tok in doc)
        has_pred = any(tok.pos_ in ("VERB","AUX") or tok.dep_=="ROOT" for tok in doc)
        return has_subj and has_pred
    has_verb = bool(re.search(r"\b(am|is|are|was|were|be|been|being|do|does|did|has|have|had|will|shall|would|should|can|could|may|might|must|[a-zA-Z]+ed|[a-zA-Z]+ing)\b", s, flags=re.I))
    has_noun = bool(re.search(r"\b(I|you|he|she|it|we|they|the|a|an|this|that|these|those|[A-Z][a-z]+)\b", s))
    return has_verb and has_noun

def remove_phrase(sentences):
    kept, removed = [], []
    for s in sentences:
        if is_full_sentence_spacy(s):
            kept.append(s)
        else:
            removed.append(s)
    return kept, removed


# ========================= PIPELINE =========================
def pdf_to_clean_sentences(pdf_path: str, out_prefix: str = "Sabina2024"):
    removed_all = []
    raw = extract_text_from_pdf_positional_auto(pdf_path)
    raw = help_ie(raw)
    sents = tokenize_sentences(raw)
    kept, rem = remove_table_of_contents(sents)
    removed_all += rem
    kept, rem = remove_head(kept)
    removed_all += rem
    kept = split_bullet(kept)
    kept = split_number_bullet(kept)
    kept, rem = remove_long_short_sentence(kept, min_words=6)
    removed_all += rem
    kept, rem = remove_phrase(kept)
    removed_all += rem
    kept, rem = remove_too_much_digit(kept, threshold=30.0)
    removed_all += rem
    kept = remove_URLs(kept)
    kept = remove_special_chars(kept)
    kept = remove_extra_whitespace(kept)
    kept = [s for s in kept if s.strip()]
    removed_all = [s for s in removed_all if s.strip()]
    return kept, removed_all


# ========================= STREAMLIT UI =========================
st.set_page_config(page_title="PDF Sentence Cleaner (เหมือน draft(no_3))", layout="wide")
st.title("📘 PDF Sentence Cleaner — เวอร์ชันตรงกับโค้ดของคุณ")

uploaded_file = st.file_uploader("📄 อัปโหลดไฟล์ PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("⏳ กำลังประมวลผล..."):
        temp_path = "temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        kept, removed = pdf_to_clean_sentences(temp_path)
        df_kept = pd.DataFrame({"Sentence Index": range(1, len(kept)+1), "Sentence": kept})
        df_removed = pd.DataFrame({"Sentence Index": range(1, len(removed)+1), "Sentence": removed})

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_kept.to_excel(writer, sheet_name="Kept", index=False)
            df_removed.to_excel(writer, sheet_name="Removed", index=False)

    st.success("✅ เสร็จสิ้น — ตัดตามโค้ดของคุณทุกขั้นตอน!")
    st.download_button(
        "⬇️ ดาวน์โหลดไฟล์ผลลัพธ์ (.xlsx)",
        data=output.getvalue(),
        file_name="Sabina2024_cleaned.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    with st.expander("🔍 ดูตัวอย่าง (20 บรรทัดแรก)"):
        st.dataframe(df_kept.head(20))

