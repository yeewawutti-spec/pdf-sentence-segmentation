#แบบที่ยังไม่ตัด header, footer
# -*- coding: utf-8 -*-
"""
Step 1-2 only: PDF -> sentences (clean) for /mnt/data/ADVANCE_2018.PDF
Outputs:
  - /mnt/data/data_pdf/ADVANCE_2018_sentences_output.csv
  - /mnt/data/data_pdf/ADVANCE_2018_sentences_removed.csv
"""

import os
import re
import csv
import math
import fitz  # PyMuPDF
import nltk
import spacy
import pandas as pd
import unicodedata  # <<< NEW

# --------- Setup ---------
PDF_PATH = "/content/ADVANC_2018.PDF"
OUT_DIR = "/mnt/data/data_pdf"
os.makedirs(OUT_DIR, exist_ok=True)

# NLTK tokenizers (safe if already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # ใช้เฉพาะถ้าจำเป็น

# Try load spaCy English; if not available, fallback to a lightweight predicate/subject heuristic
_SPACY_OK = True
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_OK = False
    nlp = None

# --------- Step 1: Extract text from PDF ---------
def extract_text_from_pdf(file_path: str) -> str:
    text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "\n".join(text)

# --------- Utilities / Cleaning helpers ---------
def help_ie(s: str) -> str:
    # ensure "i.e." followed by a comma to help sentence tokenizer
    return re.sub(r'i\.e\.(?!,)', 'i.e.,', s)

def tokenize_sentences(text: str):
    # Simple English sentence tokenizer (works well for รายงานภาษาอังกฤษ)
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def remove_extra_whitespace(sentences):
    return [re.sub(r"\s+", " ", s).strip() for s in sentences]

def remove_URLs(sentences):
    pat = re.compile(r'https?://\S+|www\.\S+')
    return [pat.sub("", s) for s in sentences]

def remove_special_chars(sentences):
    # remove common bullets/symbols that often leak from PDF
    table = str.maketrans("", "", "•*#$+|")
    return [s.translate(table) for s in sentences]

def split_bullet(sentences):
    out = []
    for s in sentences:
        parts = [p.strip() for p in re.split(r"[•]", s) if p.strip()]
        out.extend(parts)
    return out

def split_number_bullet(sentences):
    """
    Split patterns like '(1) ... (2) ...' if numbered bullets are far apart.
    Conservative to avoid over-splitting.
    """
    out = []
    for s in sentences:
        tokens = s.split()
        idxs = [i for i, w in enumerate(tokens) if re.fullmatch(r"\(\d+\)", w)]
        if len(idxs) >= 2:
            # split when gaps are large enough (>=6 tokens) to likely be distinct items
            start = 0
            for i in range(1, len(idxs)):
                if idxs[i] - idxs[i-1] >= 6:
                    out.append(" ".join(tokens[start:idxs[i]]).strip())
                    start = idxs[i]
            out.append(" ".join(tokens[start:]).strip())
        else:
            out.append(s)
    return [s for s in out if s]

def remove_table_of_contents(sentences):
    """
    Drop TOC-like lines: start with 'Table of Contents'/'Contents', or dotted leaders, or --- separators
    """
    removed, kept = [], []
    for s in sentences:
        ss = s.strip()
        if (ss.lower().startswith("table of contents")
            or ss.lower().startswith("table of content")
            or ss.lower().startswith("contents")
            or ss.lower().startswith("content")
            or re.search(r"\.{6,}", ss)            # dotted leaders
            or re.search(r"-\s*-\s*-", ss)):      # --- separators
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed

# ========= STRICT ALLCAPS REMOVER (แทนที่ฟังก์ชันเดิม) =========
def _normalize_line(s: str) -> str:  # <<< NEW
    # ป้องกัน Unicode แปลก ๆ และ normalize space
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _is_allcaps_token(tok: str) -> bool:  # <<< NEW
    t = tok.strip(",.;:()[]{}&/-")
    if not t:
        return False
    if t == "I":  # กัน false positive ของ "I"
        return False
    return t.isdigit() or (t.isupper() and any(c.isalpha() for c in t))

def _is_allcaps_line(s: str) -> bool:  # <<< NEW
    s = _normalize_line(s)
    tokens = [w for w in s.split() if w.strip(",.;:()[]{}&/-")]
    if len(tokens) <= 2:  # กันกรณีสั้น ๆ
        return False
    return all(_is_allcaps_token(w) for w in tokens)

def remove_head(sentences):  # <<< NEW (แทนของเดิมทั้งหมด)
    """
    Strict ALLCAPS remover:
    - ลบทั้งบรรทัดถ้าทั้งประโยคเป็น ALLCAPS (เช่น 'AIS HAS BEEN ...')
    - ลบหัวเรื่อง ALLCAPS ต้นบรรทัดจนถึง comma/colon
    - ลบบล็อก ALLCAPS หลายคำที่อยู่กลาง/ท้ายบรรทัด
    - หลังตัดแล้ว ถ้าเหลือเป็น ALLCAPS ทั้งบรรทัดอีก ให้ลบทิ้ง
    """
    out, removed = [], []

    for s in sentences:
        s = _normalize_line(s)
        original = s

        # (A) ถ้าทั้งบรรทัดเป็น ALLCAPS → ลบทันที
        if _is_allcaps_line(s):
            removed.append(original)
            continue

        # (B) ตัดหัวเรื่อง ALLCAPS จนถึง comma/colon หากพบ
        m = re.match(r'^([A-Z0-9 ,/&\-\(\)]+[:\,])\s*(.*)$', s)
        if m:
            head, rest = m.group(1).rstrip(",: "), m.group(2).strip()
            if _is_allcaps_line(head):
                removed.append(head)
                s = rest
            else:
                s = original  # ไม่ใช่หัว ALLCAPS ก็คืนค่าเดิม

        # (C) ถ้าส่วนที่เหลือยังเป็น ALLCAPS ทั้งบรรทัด → ลบทันที
        if s and _is_allcaps_line(s):
            removed.append(original)  # จะเก็บเป็น original หรือ s ก็ได้ตามที่อยากตรวจย้อนหลัง
            continue

        # (D) ลบบล็อก ALLCAPS หลายคำที่อยู่กลาง/ท้ายบรรทัด
        s = re.sub(r'\b(?:[A-Z0-9&/\-]{3,}(?: [A-Z0-9&/\-]{2,})+)\b', '', s)

        # (E) เก็บกวาดช่องว่าง
        s = re.sub(r'\s{2,}', ' ', s).strip()

        # (F) ถ้าหลังลบยังเป็น ALLCAPS ทั้งบรรทัด → ลบทิ้ง
        if s and _is_allcaps_line(s):
            removed.append(original)
            continue

        if s:
            out.append(s)
        else:
            removed.append(original)

    return out, removed
# ========= END STRICT ALLCAPS REMOVER =========

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
    """
    Use spaCy (en_core_web_sm) to check presence of subject & predicate.
    If spaCy unavailable, fallback to simple heuristic (contains a verb and at least one noun/pronoun).
    """
    if _SPACY_OK and nlp is not None:
        doc = nlp(s)
        has_subj = any(tok.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass") for tok in doc)
        has_pred = any(tok.pos_ in ("VERB", "AUX") or tok.dep_ == "ROOT" for tok in doc)
        return has_subj and has_pred
    # Fallback heuristic
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

# --------- Step 2: Pipeline (tokenize + clean) ---------
def pdf_to_clean_sentences(pdf_path: str, out_prefix: str = "ADVANCE_2018"):
    removed_all = []

    # 1) extract
    raw = extract_text_from_pdf(pdf_path)
    raw = help_ie(raw)

    # 2) tokenize
    sents = tokenize_sentences(raw)

    # 3) cleaning steps
    kept, rem = remove_table_of_contents(sents)
    removed_all += rem

    # <<< ใช้ STRICT ALLCAPS REMOVER แทนของเดิม
    kept, rem = remove_head(kept)
    removed_all += rem

    kept = split_bullet(kept)
    kept = split_number_bullet(kept)

    kept, rem = remove_long_short_sentence(kept, min_words=6, max_words=None)
    removed_all += rem

    kept, rem = remove_phrase(kept)
    removed_all += rem

    kept, rem = remove_too_much_digit(kept, threshold=30.0)
    removed_all += rem

    # late normalizations
    kept = remove_URLs(kept)
    kept = remove_special_chars(kept)
    kept = remove_extra_whitespace(kept)

    # Drop empties (safety)
    kept = [s for s in kept if s.strip()]
    removed_all = [s for s in removed_all if s and s.strip()]

    # Save outputs
    out_keep = os.path.join(OUT_DIR, f"{out_prefix}_sentences_output.csv")
    out_drop = os.path.join(OUT_DIR, f"{out_prefix}_sentences_removed.csv")

    with open(out_keep, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sentence Index", "Sentence"])
        for i, s in enumerate(kept, 1):
            w.writerow([i, s])

    with open(out_drop, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sentence Index", "Sentence"])
        for i, s in enumerate(removed_all, 1):
            w.writerow([i, s])

    # Console summary
    print(f"Total raw sentences: {len(sents)}")
    print(f"Kept sentences     : {len(kept)}")
    print(f"Removed sentences  : {len(removed_all)}")
    print(f"Saved kept    -> {out_keep}")
    print(f"Saved removed -> {out_drop}")

    return out_keep, out_drop


if __name__ == "__main__":
    pdf_to_clean_sentences(PDF_PATH, out_prefix="ADVANCE_2018")
    #ลองแบบตัด header, footer ใช้ granularity: str = "blocks"
# -*- coding: utf-8 -*-
"""
Step 1-2 only: PDF -> sentences (clean) for /mnt/data/ADVANCE_2018.PDF
Outputs:
  - /mnt/data/data_pdf/ADVANCE_2018_sentences_output.csv
  - /mnt/data/data_pdf/ADVANCE_2018_sentences_removed.csv
"""
!pip install PyMuPDF
import os #จัดการไฟล์และโฟลเดอร์ เช่น ตรวจสอบและสร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์
import re #ใช้ Regular Expression (regex) เพื่อค้นหา แยก หรือแทนที่ข้อความตามรูปแบบ เช่น ลบหมายเลขหน้า, รวมคำที่ถูกตัดบรรทัด, ตัดคำพิเศษ
import csv #ใช้บันทึกข้อมูลผลลัพธ์ออกเป็นไฟล์ .csv ทั้ง “ประโยคที่เก็บไว้” และ “ประโยคที่ถูกลบออก” เพื่อวิเคราะห์คุณภาพ
import math #ใช้ฟังก์ชันคณิตศาสตร์ (เช่น ปัดเศษ, ตรวจ NaN ฯลฯ) — ในโค้ดนี้เตรียมไว้สำหรับขั้นตอนคำนวณสถิติ เช่น นับจำนวนประโยคหรือคำนวณเปอร์เซ็นต์
import fitz #PyMuPDF #ไลบรารีหลักสำหรับ อ่านและดึงข้อความจากไฟล์ PDF ทีละหน้า พร้อมพิกัด (x, y) เพื่อใช้กรอง header/footer อัตโนมัติ
import nltk #ใช้ ตัดข้อความภาษาอังกฤษเป็นประโยค (sentence tokenization) และโหลดโมดูลที่จำเป็น เช่น punkt เพื่อให้ระบบรู้จุดสิ้นสุดของประโยค
import spacy #ใช้สำหรับ วิเคราะห์โครงสร้างประโยค (dependency parsing) เพื่อแยกว่าข้อความนั้นเป็น “ประโยคสมบูรณ์” หรือไม่ (มีประธาน+กริยา)
import pandas as pd #ใช้สำหรับจัดการข้อมูลในรูปแบบตาราง (DataFrame) เพื่อวิเคราะห์ภายหลัง เช่น การ merge ข้อมูล หรืออ่านผลลัพธ์ CSV อีกครั้ง
import unicodedata  # <<< NEW

# --------- Setup ---------
PDF_PATH = "/content/2023_Tisco.pdf"
OUT_DIR = "/mnt/data/data_pdf"
os.makedirs(OUT_DIR, exist_ok=True)

# NLTK tokenizers (safe if already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # ใช้เฉพาะถ้าจำเป็น

# Try load spaCy English; if not available, fallback to a lightweight predicate/subject heuristic
_SPACY_OK = True
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_OK = False
    nlp = None


# --------- Step 1 (NEW): Extract with positional header/footer removal ---------
def extract_text_from_pdf_positional_auto(
    file_path: str,
    header_margin: float = 60.0,
    footer_margin: float = 60.0,
    left_margin: float = 0.0,
    right_margin: float = 0.0,
    granularity: str = "blocks",
) -> str:
    """
    Extract text by removing any content that lies within top/bottom (and optional left/right) margins.
    - margins are in PDF points (72 pt = 1 inch)
    - granularity:
        "blocks" -> uses page.get_text("blocks") (fast, good enough for most PDFs)
        "words"  -> uses page.get_text("words") then reassembles lines (slower but finer)
    """
    pages_text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            # safe region (the area we keep)
            top = page.rect.y0 + header_margin
            bottom = page.rect.y1 - footer_margin
            left = page.rect.x0 + left_margin
            right = page.rect.x1 - right_margin

            if granularity == "words":
                # words: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
                words = page.get_text("words")
                kept_words = [
                    w for w in words
                    if (w[1] >= top and w[3] <= bottom and w[0] >= left and w[2] <= right)
                ]
                # sort into lines by y then x
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
                # blocks: (x0, y0, x1, y1, "text", block_no, block_type, ...)
                blocks = page.get_text("blocks")
                kept_blocks = []
                for b in blocks:
                    x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                    # keep only blocks fully inside the safe region
                    if (y1 <= bottom and y0 >= top and x0 >= left and x1 <= right):
                        kept_blocks.append(text)
                page_text = "\n".join(kept_blocks)

            pages_text.append(page_text)

    return "\n".join(pages_text)


# --------- Utilities / Cleaning helpers ---------
def help_ie(s: str) -> str:
    # ensure "i.e." followed by a comma to help sentence tokenizer
    return re.sub(r'i\.e\.(?!,)', 'i.e.,', s)

def tokenize_sentences(text: str):
    # Simple English sentence tokenizer (works well for รายงานภาษาอังกฤษ)
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
    # remove common bullets/symbols that often leak from PDF
    table = str.maketrans("", "", "•*#$+|")
    return [s.translate(table) for s in sentences]

def split_bullet(sentences):
    out = []
    for s in sentences:
        parts = [p.strip() for p in re.split(r"[•]", s) if p.strip()]
        out.extend(parts)
    return out

def split_number_bullet(sentences):
    """
    Split patterns like '(1) ... (2) ...' if numbered bullets are far apart.
    Conservative to avoid over-splitting.
    """
    out = []
    for s in sentences:
        tokens = s.split()
        idxs = [i for i, w in enumerate(tokens) if re.fullmatch(r"\(\d+\)", w)]
        if len(idxs) >= 2:
            # split when gaps are large enough (>=6 tokens) to likely be distinct items
            start = 0
            for i in range(1, len(idxs)):
                if idxs[i] - idxs[i-1] >= 6:
                    out.append(" ".join(tokens[start:idxs[i]]).strip())
                    start = idxs[i]
            out.append(" ".join(tokens[start:]).strip())
        else:
            out.append(s)
    return [s for s in out if s]

def remove_table_of_contents(sentences):
    """
    Drop TOC-like lines: start with 'Table of Contents'/'Contents', or dotted leaders, or --- separators
    """
    removed, kept = [], []
    for s in sentences:
        ss = s.strip()
        if (ss.lower().startswith("table of contents")
            or ss.lower().startswith("table of content")
            or ss.lower().startswith("contents")
            or ss.lower().startswith("content")
            or re.search(r"\.{6,}", ss)            # dotted leaders
            or re.search(r"-\s*-\s*-", ss)):      # --- separators
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed

# ========= STRICT ALLCAPS REMOVER =========
def _normalize_line(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _is_allcaps_token(tok: str) -> bool:
    t = tok.strip(",.;:()[]{}&/-")
    if not t:
        return False
    if t == "I":  # กัน false positive ของ "I"
        return False
    return t.isdigit() or (t.isupper() and any(c.isalpha() for c in t))

def _is_allcaps_line(s: str) -> bool:
    s = _normalize_line(s)
    tokens = [w for w in s.split() if w.strip(",.;:()[]{}&/-")]
    if len(tokens) <= 2:  # กันกรณีสั้น ๆ
        return False
    return all(_is_allcaps_token(w) for w in tokens)

def remove_head(sentences):
    """
    Strict ALLCAPS remover:
    - ลบทั้งบรรทัดถ้าทั้งประโยคเป็น ALLCAPS (เช่น 'AIS HAS BEEN ...')
    - ลบหัวเรื่อง ALLCAPS ต้นบรรทัดจนถึง comma/colon
    - ลบบล็อก ALLCAPS หลายคำที่อยู่กลาง/ท้ายบรรทัด
    - หลังตัดแล้ว ถ้าเหลือเป็น ALLCAPS ทั้งบรรทัดอีก ให้ลบทิ้ง
    """
    out, removed = [], []

    for s in sentences:
        s = _normalize_line(s)
        original = s

        # (A) ถ้าทั้งบรรทัดเป็น ALLCAPS → ลบทันที
        if _is_allcaps_line(s):
            removed.append(original)
            continue

        # (B) ตัดหัวเรื่อง ALLCAPS จนถึง comma/colon หากพบ
        m = re.match(r'^([A-Z0-9 ,/&\-\(\)]+[:\,])\s*(.*)$', s)
        if m:
            head, rest = m.group(1).rstrip(",: "), m.group(2).strip()
            if _is_allcaps_line(head):
                removed.append(head)
                s = rest
            else:
                s = original  # ไม่ใช่หัว ALLCAPS ก็คืนค่าเดิม

        # (C) ถ้าส่วนที่เหลือยังเป็น ALLCAPS ทั้งบรรทัด → ลบทันที
        if s and _is_allcaps_line(s):
            removed.append(original)
            continue

        # (D) ลบบล็อก ALLCAPS หลายคำที่อยู่กลาง/ท้ายบรรทัด
        s = re.sub(r'\b(?:[A-Z0-9&/\-]{3,}(?: [A-Z0-9&/\-]{2,})+)\b', '', s)

        # (E) เก็บกวาดช่องว่าง
        s = re.sub(r'\s{2,}', ' ', s).strip()

        # (F) ถ้าหลังลบยังเป็น ALLCAPS ทั้งบรรทัด → ลบทิ้ง
        if s and _is_allcaps_line(s):
            removed.append(original)
            continue

        if s:
            out.append(s)
        else:
            removed.append(original)

    return out, removed
# ========= END STRICT ALLCAPS REMOVER =========

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
    """
    Use spaCy (en_core_web_sm) to check presence of subject & predicate.
    If spaCy unavailable, fallback to simple heuristic (contains a verb and at least one noun/pronoun).
    """
    if _SPACY_OK and nlp is not None:
        doc = nlp(s)
        has_subj = any(tok.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass") for tok in doc)
        has_pred = any(tok.pos_ in ("VERB", "AUX") or tok.dep_ == "ROOT" for tok in doc)
        return has_subj and has_pred
    # Fallback heuristic
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


# --------- Step 2: Pipeline (tokenize + clean) ---------
def pdf_to_clean_sentences(pdf_path: str, out_prefix: str = "2023_Tisco"):
    removed_all = []

    # 1) extract (ใช้แบบ positional เพื่อตัดหัว/ท้ายกระดาษโดยอิงตำแหน่ง)
    raw = extract_text_from_pdf_positional_auto(
        pdf_path,
        header_margin=60,   # ปรับได้: 60pt ~ 0.83"
        footer_margin=60,   # ปรับได้
        left_margin=0,
        right_margin=0,
        granularity="blocks"  # ลอง "words" ถ้าหน้าตัดยาก/บล็อกคร่อม margin
    )
    raw = help_ie(raw)

    # 2) tokenize
    sents = tokenize_sentences(raw)

    # 3) cleaning steps
    kept, rem = remove_table_of_contents(sents)
    removed_all += rem

    # STRICT ALLCAPS remover
    kept, rem = remove_head(kept)
    removed_all += rem

    kept = split_bullet(kept)
    kept = split_number_bullet(kept)

    kept, rem = remove_long_short_sentence(kept, min_words=6, max_words=None)
    removed_all += rem

    kept, rem = remove_phrase(kept)
    removed_all += rem

    kept, rem = remove_too_much_digit(kept, threshold=30.0)
    removed_all += rem

    # late normalizations
    kept = remove_URLs(kept)
    kept = remove_special_chars(kept)
    kept = remove_extra_whitespace(kept)

    # Drop empties (safety)
    kept = [s for s in kept if s.strip()]
    removed_all = [s for s in removed_all if s and s.strip()]

    # Save outputs
    out_keep = os.path.join(OUT_DIR, f"{out_prefix}_sentences_output.csv")
    out_drop = os.path.join(OUT_DIR, f"{out_prefix}_sentences_removed.csv")

    with open(out_keep, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sentence Index", "Sentence"])
        for i, s in enumerate(kept, 1):
            w.writerow([i, s])

    with open(out_drop, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sentence Index", "Sentence"])
        for i, s in enumerate(removed_all, 1):
            w.writerow([i, s])

    # Console summary
    print(f"Total raw sentences: {len(sents)}")
    print(f"Kept sentences     : {len(kept)}")
    print(f"Removed sentences  : {len(removed_all)}")
    print(f"Saved kept    -> {out_keep}")
    print(f"Saved removed -> {out_drop}")

    return out_keep, out_drop


if __name__ == "__main__":
    pdf_to_clean_sentences(PDF_PATH, out_prefix="2023_Tisco")
#ลองแบบตัดตัวหนา
# -*- coding: utf-8 -*-
"""
Step 1-2 only: PDF -> sentences (clean) for /mnt/data/ADVANCE_2018.PDF
Outputs:
  - /mnt/data/data_pdf/ADVANCE_2018_sentences_output.csv
  - /mnt/data/data_pdf/ADVANCE_2018_sentences_removed.csv
"""

import os #จัดการไฟล์และโฟลเดอร์ เช่น ตรวจสอบและสร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์
import re #ใช้ Regular Expression (regex) เพื่อค้นหา แยก หรือแทนที่ข้อความตามรูปแบบ เช่น ลบหมายเลขหน้า, รวมคำที่ถูกตัดบรรทัด, ตัดคำพิเศษ
import csv #ใช้บันทึกข้อมูลผลลัพธ์ออกเป็นไฟล์ .csv ทั้ง “ประโยคที่เก็บไว้” และ “ประโยคที่ถูกลบออก” เพื่อวิเคราะห์คุณภาพ
import math #ใช้ฟังก์ชันคณิตศาสตร์ (เช่น ปัดเศษ, ตรวจ NaN ฯลฯ) — ในโค้ดนี้เตรียมไว้สำหรับขั้นตอนคำนวณสถิติ เช่น นับจำนวนประโยคหรือคำนวณเปอร์เซ็นต์
import fitz #PyMuPDF #ไลบรารีหลักสำหรับ อ่านและดึงข้อความจากไฟล์ PDF ทีละหน้า พร้อมพิกัด (x, y) เพื่อใช้กรอง header/footer อัตโนมัติ
import nltk #ใช้ ตัดข้อความภาษาอังกฤษเป็นประโยค (sentence tokenization) และโหลดโมดูลที่จำเป็น เช่น punkt เพื่อให้ระบบรู้จุดสิ้นสุดของประโยค
import spacy #ใช้สำหรับ วิเคราะห์โครงสร้างประโยค (dependency parsing) เพื่อแยกว่าข้อความนั้นเป็น “ประโยคสมบูรณ์” หรือไม่ (มีประธาน+กริยา)
import pandas as pd #ใช้สำหรับจัดการข้อมูลในรูปแบบตาราง (DataFrame) เพื่อวิเคราะห์ภายหลัง เช่น การ merge ข้อมูล หรืออ่านผลลัพธ์ CSV อีกครั้ง
import unicodedata  # <<< NEW

# --------- Setup ---------
PDF_PATH = "/content/2024_Tisco.pdf"
OUT_DIR = "/mnt/data/data_pdf"
os.makedirs(OUT_DIR, exist_ok=True)

# NLTK tokenizers (safe if already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # ใช้เฉพาะถ้าจำเป็น

# Try load spaCy English; if not available, fallback to a lightweight predicate/subject heuristic
_SPACY_OK = True
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_OK = False
    nlp = None


# --------- Step 1 (UPDATED): Extract with positional header/footer + remove ALL bold spans ---------
def extract_text_from_pdf_positional_auto(
    file_path: str,
    header_margin: float = 60.0,
    footer_margin: float = 60.0,
    left_margin: float = 0.0,
    right_margin: float = 0.0,
    granularity: str = "spans",               # <<< ต้องเป็น "spans" เพื่ออ่านฟอนต์/flags ได้
    # --- โหมดการตัดตัวหนา ---
    remove_bold_all: bool = True,             # <<< ลบ "ทุกตัวหนา" ถ้า True
    remove_bold_lines: bool = False,          # ลบทั้งบรรทัดถ้าดูเป็น heading (ปิดไว้เพื่อกันลบเกิน)
    bold_regex: str = r"(?i)\b(bold|black|heavy|semibold|demi)\b",
    heading_bold_ratio: float = 0.7,
    heading_size_multiplier: float = 1.15,
) -> str:

    def _is_inside(bbox, left, top, right, bottom):
        x0, y0, x1, y1 = bbox
        return (y1 <= bottom and y0 >= top and x0 >= left and x1 <= right)

    def _span_is_bold(span) -> bool:
        # ตรวจจากชื่อฟอนต์ + flags (บิต 2 มักหมายถึง bold)
        font = span.get("font", "") or ""
        flags = int(span.get("flags", 0) or 0)
        by_name  = bool(re.search(bold_regex, font))
        by_flags = bool(flags & 2)
        return by_name or by_flags

    def _collect_span_sizes(pdict, left, top, right, bottom):
        sizes = []
        for block in pdict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span.get("bbox", None)
                    if not bbox:
                        continue
                    x0, y0, x1, y1 = bbox
                    if (y1 <= bottom and y0 >= top and x0 >= left and x1 <= right):
                        sizes.append(float(span.get("size", 0.0) or 0.0))
        return sizes

    pages_text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            # safe region (the area we keep)
            top = page.rect.y0 + header_margin
            bottom = page.rect.y1 - footer_margin
            left = page.rect.x0 + left_margin
            right = page.rect.x1 - right_margin

            if granularity == "spans":
                pdict = page.get_text("dict")

                # median font size ของหน้า (ภายใน safe region) — เผื่อเปิดใช้ remove_bold_lines
                sizes = _collect_span_sizes(pdict, left, top, right, bottom)
                median_size = (sorted(sizes)[len(sizes)//2] if sizes else 0.0)

                line_buf = []
                for block in pdict.get("blocks", []):
                    if block.get("type", 0) != 0:
                        continue
                    for line in block.get("lines", []):
                        spans_all = 0
                        bold_count = 0
                        max_size_in_line = 0.0
                        spans_in_line = []

                        for span in line.get("spans", []):
                            text  = span.get("text", "")
                            bbox  = span.get("bbox", None)
                            size  = float(span.get("size", 0.0) or 0.0)
                            if not bbox or not text:
                                continue
                            if not _is_inside(bbox, left, top, right, bottom):
                                continue

                            spans_all += 1
                            max_size_in_line = max(max_size_in_line, size)
                            is_bold = _span_is_bold(span)

                            # --- โหมด "ตัดทุกตัวหนา": ข้ามทันที ---
                            if remove_bold_all and is_bold:
                                continue

                            if is_bold:
                                bold_count += 1

                            spans_in_line.append(text)

                        if not spans_in_line:
                            # ทั้งบรรทัดเป็นตัวหนา → ไม่มี span รอดเลย ก็ข้ามบรรทัดนี้
                            continue

                        looks_like_heading = False
                        if remove_bold_lines:
                            bold_ratio = bold_count / max(1, spans_all)
                            if median_size:
                                looks_like_heading = (bold_ratio >= heading_bold_ratio) and (max_size_in_line >= median_size * heading_size_multiplier)
                            else:
                                looks_like_heading = (bold_ratio >= heading_bold_ratio)

                        if looks_like_heading:
                            continue  # ตัดทั้งบรรทัด
                        else:
                            # รวมข้อความที่รอด (ไม่มีตัวหนาแล้วถ้า remove_bold_all=True)
                            merged = "".join(spans_in_line)
                            # กันคำติด: บาง PDF ไม่มีช่องว่างใน span
                            merged = re.sub(r'[ \t]+', ' ', merged)
                            if merged.strip():
                                line_buf.append(merged)

                page_text = "\n".join(line_buf)

            elif granularity == "words":
                words = page.get_text("words")
                kept_words = [
                    w for w in words
                    if (w[1] >= top and w[3] <= bottom and w[0] >= left and w[2] <= right)
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
                # blocks: (x0, y0, x1, y1, "text", block_no, block_type, ...)
                blocks = page.get_text("blocks")
                kept_blocks = []
                for b in blocks:
                    x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                    if (y1 <= bottom and y0 >= top and x0 >= left and x1 <= right):
                        kept_blocks.append(text)
                page_text = "\n".join(kept_blocks)

            pages_text.append(page_text)

    return "\n".join(pages_text)


# --------- Utilities / Cleaning helpers ---------
def help_ie(s: str) -> str:
    s = re.sub(r'\bi\.e\.(?=\s*\w)(?!\s*,)', 'i.e.,', s, flags=re.IGNORECASE)
    s = re.sub(r'\be\.g\.(?=\s*\w)(?!\s*,)', 'e.g.,', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNo\.(?=\s*\w)(?!\s*,)',  'No.,',  s, flags=re.IGNORECASE)
    return s
#แบบเก่า
#def help_ie(s: str) -> str:
    # ensure "i.e." followed by a comma to help sentence tokenizer
    # return re.sub(r'i\.e\.(?!,)', 'i.e.,', s)

def tokenize_sentences(text: str):
    # Simple English sentence tokenizer (works well for รายงานภาษาอังกฤษ)
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
    # remove common bullets/symbols that often leak from PDF
    table = str.maketrans("", "", "•*#+|")
    return [s.translate(table) for s in sentences]

def split_bullet(sentences):
    out = []
    for s in sentences:
        parts = [p.strip() for p in re.split(r"[•]", s) if p.strip()]
        out.extend(parts)
    return out

def split_number_bullet(sentences):
    """
    Split patterns like '(1) ... (2) ...' if numbered bullets are far apart.
    Conservative to avoid over-splitting.
    """
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

def remove_table_of_contents(sentences):
    """
    Drop TOC-like lines: start with 'Table of Contents'/'Contents', or dotted leaders, or --- separators
    """
    removed, kept = [], []
    for s in sentences:
        ss = s.strip()
        if (ss.lower().startswith("table of contents")
            or ss.lower().startswith("table of content")
            or ss.lower().startswith("contents")
            or ss.lower().startswith("content")
            or re.search(r"\.{6,}", ss)
            or re.search(r"-\s*-\s*-", ss)):
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed

# ========= STRICT ALLCAPS REMOVER =========
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
# ========= END STRICT ALLCAPS REMOVER =========

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
        has_subj = any(tok.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass") for tok in doc)
        has_pred = any(tok.pos_ in ("VERB", "AUX") or tok.dep_ == "ROOT" for tok in doc)
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


# --------- Step 2: Pipeline (tokenize + clean) ---------
def pdf_to_clean_sentences(pdf_path: str, out_prefix: str = "2024_Tisco"):
    removed_all = []

    # 1) extract (ตัดหัว/ท้าย + ลบทุก span ตัวหนา)
    raw = extract_text_from_pdf_positional_auto(
        pdf_path,
        header_margin=60,
        footer_margin=60,
        left_margin=0,
        right_margin=0,
        granularity="spans",          # <<< สำคัญ
        remove_bold_all=True,         # <<< ลบ "ทุกตัวหนา"
        remove_bold_lines=False       # กันลบเกิน; เปิดได้ถ้าต้องการลบหัวข้อยกบรรทัด
    )
    raw = help_ie(raw)

    # 2) tokenize
    sents = tokenize_sentences(raw)

    # 3) cleaning steps
    kept, rem = remove_table_of_contents(sents)
    removed_all += rem

    kept, rem = remove_head(kept)  # STRICT ALLCAPS remover
    removed_all += rem

    kept = split_bullet(kept)
    kept = split_number_bullet(kept)

    kept, rem = remove_long_short_sentence(kept, min_words=6, max_words=None)
    removed_all += rem

    kept, rem = remove_phrase(kept)
    removed_all += rem

    kept, rem = remove_too_much_digit(kept, threshold=30.0)
    removed_all += rem

    # late normalizations
    kept = remove_URLs(kept)
    kept = remove_special_chars(kept)
    kept = remove_extra_whitespace(kept)

    # Drop empties (safety)
    kept = [s for s in kept if s.strip()]
    removed_all = [s for s in removed_all if s and s.strip()]

    # Save outputs
    out_keep = os.path.join(OUT_DIR, f"{out_prefix}_sentences_output.csv")
    out_drop = os.path.join(OUT_DIR, f"{out_prefix}_sentences_removed.csv")

    with open(out_keep, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sentence Index", "Sentence"])
        for i, s in enumerate(kept, 1):
            w.writerow([i, s])

    with open(out_drop, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sentence Index", "Sentence"])
        for i, s in enumerate(removed_all, 1):
            w.writerow([i, s])

    # Console summary
    print(f"Total raw sentences: {len(sents)}")
    print(f"Kept sentences     : {len(kept)}")
    print(f"Removed sentences  : {len(removed_all)}")
    print(f"Saved kept    -> {out_keep}")
    print(f"Saved removed -> {out_drop}")

    return out_keep, out_drop


if __name__ == "__main__":
    pdf_to_clean_sentences(PDF_PATH, out_prefix="2024_Tisco")
#แปลงไฟล์ csv เป็น xlsx เพื่อป้องกันการ Export File แล้วเกิดปัญหาคำเพี้ยน

# Define the paths
output_csv_path = "/mnt/data/data_pdf//2024_Tisco_sentences_output.csv"
output_xlsx_path = "/mnt/data/data_pdf/x_2024_Tisco_sentences_output.xlsx"

# Load the CSV into a DataFrame
try:
    df_output = pd.read_csv(output_csv_path, encoding='utf-8')

    # Save the DataFrame to an Excel file (.xlsx)
    df_output.to_excel(output_xlsx_path, index=False)

    print(f"Successfully saved data to {output_xlsx_path}")

except FileNotFoundError:
    print(f"Error: The file {output_csv_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
