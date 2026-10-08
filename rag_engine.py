import os
import glob
import hashlib
import logging
import re
import unicodedata
from functools import lru_cache

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config

logger = logging.getLogger("rag_engine")


# ======================================================================
# TIỆN ÍCH CHUẨN HÓA TEXT (dùng cho keyword matching)
# ======================================================================

def _fold(text: str) -> str:
    """Hạ chữ thường + bỏ dấu tiếng Việt (đ -> d) để so khớp không phụ thuộc dấu/kiểu gõ (kỳ/kì)."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


# Viết tắt thường gặp -> dạng đầy đủ (áp dụng cho CẢ câu hỏi và tài liệu khi so khớp)
_ABBREVIATIONS = {
    "clb": "câu lạc bộ",
    "gv": "giáo viên",
    "hs": "học sinh",
    "gvcn": "giáo viên chủ nhiệm",
}
_ABBR_FOLDED = {_fold(k): _fold(v).split() for k, v in _ABBREVIATIONS.items()}

_STOPWORDS = {
    _fold(w) for w in [
        "có", "những", "các", "là", "của", "trường", "nào", "ai", "gì", "cho",
        "về", "hiện", "nay", "được", "bao", "nhiêu", "theo", "thông", "tin",
        "hãy", "biết", "mình", "em", "tôi", "bạn", "ở", "và", "thì", "không",
        "muốn", "hỏi", "xin", "với", "trong", "một", "này", "đó", "như", "thế",
        "nhé", "giúp", "cần", "đang", "còn", "vậy", "mấy",
    ]
}


def _tokens(text: str) -> list:
    out = []
    for tok in re.findall(r"\w+", _fold(text)):
        out.extend(_ABBR_FOLDED.get(tok, [tok]))
    return out


@lru_cache(maxsize=8192)
def _doc_features(text: str):
    """(tập token, chuỗi ' t1 t2 t3 ') của một chunk — cache để không tính lại mỗi câu hỏi."""
    toks = _tokens(text)
    return frozenset(toks), " " + " ".join(toks) + " "


def _query_phrases(q_tokens: list) -> set:
    """Cụm 2-4 từ liên tiếp trong câu hỏi, bỏ cụm chỉ gồm stopword."""
    phrases = set()
    for n in (2, 3, 4):
        for i in range(len(q_tokens) - n + 1):
            window = q_tokens[i:i + n]
            if all(t in _STOPWORDS for t in window):
                continue
            phrases.add(" ".join(window))
    return phrases


# ======================================================================
# NẠP FILE
# ======================================================================

def _extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _reformat_giaovien_file(content: str) -> str:
    """
    giao_vien.txt: bảng "Họ và tên | Ngày sinh | Giới tính | Môn dạy | Chức vụ"
    (không có cột STT). Nhận diện dòng dữ liệu bằng cột Ngày sinh khớp mẫu
    dd/mm/yyyy thay vì dựa vào cột đầu là số.

    Gom theo TỪNG MÔN/BỘ PHẬN thành 1 section (## ...) thay vì 1 chunk khổng lồ,
    để embedding sát nội dung hơn; thêm 1 section tổng hợp số lượng toàn trường.
    """
    DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")

    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 5:
            continue

        # Cột Ngày sinh phải khớp dd/mm/yyyy -> loại dòng tiêu đề / dòng gạch ngang
        if not DATE_RE.match(cols[1]):
            continue

        name, dob, gender, subject, role = cols[:5]
        rows.append({"name": name, "dob": dob, "gender": gender, "subject": subject, "role": role})

    if not rows:
        logger.warning("Không phân tích được dòng dữ liệu nào trong giao_vien.txt")
        return content  # fallback: giữ nguyên nội dung gốc, tránh mất dữ liệu

    # Gom theo môn/bộ phận, GIỮ NGUYÊN thứ tự xuất hiện trong file gốc.
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)

    section_parts = []
    for subject, people in by_subject.items():
        lines = [f"## Bộ môn/công việc: {subject}", f"Số lượng: {len(people)} người."]
        for p in people:
            lines.append(
                f"- {p['name']} | Ngày sinh: {p['dob']} | Giới tính: {p['gender']} "
                f"| Chức vụ: {p['role']} | Môn/công việc: {subject}."
            )
        section_parts.append("\n".join(lines))

    # ---- Section tổng hợp toàn trường ----
    n_giao_vien = sum(1 for r in rows if r["role"] == "Giáo viên")
    n_khac = len(rows) - n_giao_vien
    subject_count_lines = [f"- {subject}: {len(people)} người" for subject, people in by_subject.items()]

    summary_chunk = (
        "## Tổng hợp số lượng giáo viên và nhân viên toàn trường theo môn học/bộ phận\n"
        f"Tổng số giáo viên và nhân viên trường THPT Hoài Đức A: {len(rows)} người, "
        f"gồm {n_giao_vien} giáo viên và {n_khac} nhân viên/chức vụ khác, "
        f"chia theo {len(by_subject)} môn/bộ phận.\n"
        + "\n".join(subject_count_lines)
    )

    return "\n\n".join(section_parts) + "\n\n" + summary_chunk


_TEACHER_FILE = "giao_vien.txt"
_TEACHER_HEADING_PREFIX = "## Bộ môn/công việc:"

_SPECIAL_FILE_PREPROCESSORS = {
    _TEACHER_FILE: _reformat_giaovien_file,
}

# Cách gọi khác của môn học (đã bỏ dấu) -> một phần tên môn trong file. Bổ sung theo dữ liệu thật.
# Cố ý KHÔNG có các từ dễ nhầm như "tin", "sinh", "anh", "sử".
_SUBJECT_ALIASES = {
    "the duc": "the chat",
    "gdcd": "cong dan",
    "gdqp": "quoc phong",
}

_LIST_VERBS = ("danh sach", "liet ke", "ke ten", "tat ca", "toan bo")
_HISTORY_PHRASES = (
    "hieu truong thu", "hieu truong dau", "hieu truong qua cac thoi ky",
    "hieu truong qua cac thoi ki", "danh sach hieu truong", "lich su hieu truong",
)
_HISTORY_DOC_PHRASES = ("hieu truong qua cac thoi ky", "hieu truong qua cac thoi ki")


class RAGEngine:
    def __init__(self):
        from image_index import ImageIndex
        self.image_index = ImageIndex(config.IMAGE_MANIFEST, config.IMAGE_URL_PREFIX)
        # EphemeralClient (in-memory): file .txt/.pdf trong DATA_DIR là nguồn sự thật,
        # mỗi lần server khởi động ingest_directory() build lại toàn bộ vector store.
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        # section_key -> {source, heading, text, chunks, joined}
        # Dùng để trả về NGUYÊN section (parent) khi một chunk nhỏ (child) trúng.
        self._sections = {}

    # ------------------------------------------------------------------
    # NẠP DỮ LIỆU
    # ------------------------------------------------------------------
    @staticmethod
    def _split_heading(part: str):
        if part.startswith("## "):
            first, _, rest = part.partition("\n")
            return first.strip(), rest.strip()
        return None, part

    def add_documents(self, docs_text: str, source: str = "unknown") -> int:
        """
        Tách theo section "## ", rồi chia nhỏ từng section (RecursiveCharacterTextSplitter)
        để embedding/keyword matching chính xác. Mỗi chunk con được gắn lại tiêu đề section
        (không mất ngữ cảnh), và toàn bộ section được lưu để trả về nguyên vẹn khi truy xuất.
        """
        parts = re.split(r"(?=^## )", docs_text, flags=re.MULTILINE)

        documents, ids, metadatas = [], [], []
        total = 0

        for sec_idx, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            heading, body = self._split_heading(part)
            pieces = [p.strip() for p in self.text_splitter.split_text(body) if p.strip()]
            if not pieces:
                pieces = [""]  # section chỉ có tiêu đề

            section_key = f"{source}#{sec_idx}"
            self._sections[section_key] = {
                "source": source,
                "heading": heading,
                "text": part,
                "chunks": pieces,
                "joined": _doc_features(part)[1],
            }

            for part_idx, piece in enumerate(pieces):
                text = f"{heading}\n{piece}".strip() if heading else piece
                raw_id = f"{source}_{sec_idx}_{part_idx}_{text}"
                chunk_hash = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

                documents.append(text)
                ids.append(f"doc_{chunk_hash}")
                metadatas.append({
                    "source": source,
                    "chunk_id": total,
                    "section": section_key,
                    "part": part_idx,
                })
                total += 1

        if documents:
            logger.info(f"Bắt đầu upsert {len(documents)} chunks...")
            self.collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            logger.info("Upsert hoàn tất.")

        return total

    def ingest_directory(self, data_dir: str = None, rebuild: bool = False) -> dict:
        """Nạp toàn bộ .txt/.pdf trong thư mục. Gọi lúc server startup và từ load_data.py."""
        data_dir = data_dir or config.DATA_DIR
        stats = {"files": 0, "chunks": 0, "skipped": []}

        if rebuild:
            try:
                self.chroma_client.delete_collection(name=config.COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)
            self._sections = {}

        if not os.path.isdir(data_dir):
            logger.warning(f"Thư mục dữ liệu không tồn tại: {data_dir}")
            return stats

        paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.pdf")))

        for path in paths:
            source_name = os.path.basename(path)
            try:
                if path.lower().endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                else:
                    content = _extract_text_from_pdf(path)

                if not content.strip():
                    stats["skipped"].append(source_name)
                    continue

                preprocessor = _SPECIAL_FILE_PREPROCESSORS.get(source_name)
                if preprocessor:
                    content = preprocessor(content)

                num_chunks = self.add_documents(content, source=source_name)
                stats["files"] += 1
                stats["chunks"] += num_chunks
                logger.info(f"Đã nạp {source_name}: {num_chunks} đoạn")
            except Exception as e:
                logger.error(f"Lỗi khi nạp {source_name}: {e}")
                stats["skipped"].append(source_name)

        return stats

    # ------------------------------------------------------------------
    # TRUY XUẤT
    # ------------------------------------------------------------------
    def _intent_boosts(self, q_joined: str) -> dict:
        """
        Các luật ưu tiên theo ý định câu hỏi. Trả về {section_key: mức_cộng_điểm}.
        Chỉ là CỘNG ĐIỂM (không loại trừ chunk nào, không return sớm) nên không làm mất
        nguồn/hình ảnh và không chặn các chunk đúng khác.
        """
        boosts = {}
        history_boost = getattr(config, "HISTORY_BOOST", 1.0)
        list_boost = getattr(config, "TEACHER_LIST_BOOST", 1.5)

        # 1) Lịch sử hiệu trưởng
        if any(f" {p} " in q_joined for p in _HISTORY_PHRASES):
            for key, sec in self._sections.items():
                if any(f" {p} " in sec["joined"] for p in _HISTORY_DOC_PHRASES):
                    boosts[key] = history_boost

        # 2) "Danh sách / liệt kê giáo viên môn X": cộng điểm cho đúng section môn X
        if " giao vien " in q_joined and any(f" {v} " in q_joined for v in _LIST_VERBS):
            for key, sec in self._sections.items():
                heading = sec["heading"] or ""
                if sec["source"] != _TEACHER_FILE or not heading.startswith(_TEACHER_HEADING_PREFIX):
                    continue
                subject = _fold(heading.split(":", 1)[1].strip())
                subject = " ".join(_tokens(subject))
                matched = f" {subject} " in q_joined or any(
                    f" {alias} " in q_joined and target in subject
                    for alias, target in _SUBJECT_ALIASES.items()
                )
                if matched:
                    boosts[key] = list_boost

        return boosts

    def _build_context(self, item: dict) -> str:
        """Trả về nguyên section chứa chunk trúng; nếu section quá dài thì lấy chunk trúng + chunk liền kề."""
        md = item["metadata"]
        sec = self._sections.get(md.get("section"))
        if not sec:
            return item["doc"]

        max_chars = getattr(config, "MAX_CONTEXT_CHARS", 3500)
        if len(sec["text"]) <= max_chars:
            return sec["text"]

        part = md.get("part", 0)
        chunks = sec["chunks"]
        body = "\n".join(chunks[max(0, part - 1):part + 2])
        return f"{sec['heading']}\n{body}" if sec["heading"] else body

    def retrieve_with_threshold(self, query_text: str, debug: bool = False):
        """
        Hybrid retrieval:
        1. Vector search trên toàn bộ chunk.
        2. Chấm điểm: distance - (độ phủ từ khóa) - (khớp cụm từ, có trần) - (cộng điểm theo ý định).
        3. Lọc theo SIMILARITY_THRESHOLD trên ĐIỂM CUỐI (để keyword có thể "cứu" chunk đúng).
        4. Gom theo section, lấy TOP_K section tốt nhất, trả về nguyên section (parent).
        """
        total = self.collection.count()
        if total == 0:
            return None

        results = self.collection.query(
            query_texts=[query_text],
            n_results=total,
            include=["documents", "distances", "metadatas"],
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        if not documents:
            return None

        q_tokens = _tokens(query_text)
        q_joined = " " + " ".join(q_tokens) + " "
        query_terms = {t for t in q_tokens if t not in _STOPWORDS}
        phrases = _query_phrases(q_tokens)
        boosts = self._intent_boosts(q_joined)

        kw_weight = getattr(config, "KEYWORD_WEIGHT", 0.5)
        phrase_weight = getattr(config, "PHRASE_WEIGHT", 0.1)
        phrase_cap = getattr(config, "PHRASE_CAP", 0.4)

        candidates = []
        for doc, distance, metadata in zip(documents, distances, metadatas):
            doc_tokens, doc_joined = _doc_features(doc)

            matched = sum(1 for t in query_terms if t in doc_tokens)
            coverage = matched / len(query_terms) if query_terms else 0.0

            phrase_hits = sum(1 for p in phrases if f" {p} " in doc_joined)
            phrase_bonus = min(phrase_hits * phrase_weight, phrase_cap)

            intent_bonus = boosts.get(metadata.get("section"), 0.0)

            score = distance - kw_weight * coverage - phrase_bonus - intent_bonus

            if score > config.SIMILARITY_THRESHOLD:
                continue

            candidates.append({
                "doc": doc,
                "distance": distance,
                "metadata": metadata,
                "score": score,
                "coverage": coverage,
            })

        candidates.sort(key=lambda x: x["score"])

        if debug or logger.isEnabledFor(logging.DEBUG):
            log = logger.info if debug else logger.debug
            log(f"[RAG] query={query_text!r} terms={sorted(query_terms)} boosts={list(boosts)}")
            for c in candidates[:8]:
                log(
                    f"[RAG]  score={c['score']:.3f} dist={c['distance']:.3f} cov={c['coverage']:.2f} "
                    f"{c['metadata'].get('section')} | {c['doc'][:70]!r}"
                )

        if not candidates:
            return None

        # Mỗi section chỉ giữ chunk tốt nhất, rồi lấy TOP_K section
        best_by_section = {}
        for c in candidates:
            key = c["metadata"].get("section") or c["doc"]
            if key not in best_by_section:
                best_by_section[key] = c  # candidates đã sort -> chunk đầu tiên là tốt nhất

        selected = list(best_by_section.values())[:config.TOP_K]

        contexts, sources = [], []
        for item in selected:
            ctx = self._build_context(item)
            if ctx not in contexts:
                contexts.append(ctx)

            metadata = item["metadata"]
            sources.append({
                "source": metadata.get("source", "unknown"),
                "chunk_id": metadata.get("chunk_id", -1),
                "distance": round(item["distance"], 4),
                "snippet": item["doc"][:200],
            })

        try:
            images = self.image_index.find(query_text)
        except Exception:
            logger.exception("Lỗi khi tìm hình ảnh")
            images = []

        return {
            "context": "\n\n".join(contexts),
            "sources": sources,
            "images": images,
        }