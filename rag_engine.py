import os
import glob
import hashlib
import logging
import re

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import config
logger = logging.getLogger("rag_engine")


def _extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _reformat_giaovien_file(content: str) -> str:
    """
    giao_vien.txt: bảng "Họ và tên | Ngày sinh | Giới tính | Môn dạy | Chức vụ"
    (không có cột STT). Nhận diện dòng dữ liệu bằng cột Ngày sinh khớp mẫu
    dd/mm/yyyy thay vì dựa vào cột đầu là số — bền hơn khi số cột/định dạng
    bảng thay đổi giữa các lần xuất dữ liệu.

    Gom theo TỪNG MÔN/BỘ PHẬN thành 1 chunk/môn thay vì để cả bảng thành 1
    chunk khổng lồ. Lý do bắt buộc phải làm việc này: nếu để nguyên 1 chunk
    chứa toàn bộ ~100 người, embedding của chunk đó đại diện một "mức trung
    bình ngữ nghĩa" bị chi phối bởi các khuôn mẫu tên phổ biến nhất trong
    bảng; câu hỏi về những tên lệch khuôn mẫu đó (tên ngắn, họ hiếm) có thể
    có vector distance vượt SIMILARITY_THRESHOLD và bị loại ở bước lọc distance
    — TRƯỚC khi keyword/phrase bonus trong retrieve_with_threshold() kịp "cứu"
    lại, dù tên đó khớp chính xác trong text. Chia nhỏ theo môn giúp mỗi
    chunk có embedding sát với các thành viên bên trong nó hơn nhiều.
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

        # Cột Ngày sinh phải khớp dd/mm/yyyy -> phân biệt với dòng tiêu đề
        # ("Họ và tên | Ngày sinh | ...") và dòng gạch ngang phân cách.
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

    # ---- Chunk tổng hợp toàn trường ----
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


_SPECIAL_FILE_PREPROCESSORS = {
    "giao_vien.txt": _reformat_giaovien_file,
}


class RAGEngine:
    def __init__(self):
        from image_index import ImageIndex
        self.image_index = ImageIndex(config.IMAGE_MANIFEST, config.IMAGE_URL_PREFIX)
        # QUYẾT ĐỊNH KIẾN TRÚC: EphemeralClient (in-memory), KHÔNG ghi ra đĩa.
        # File .txt/.pdf trong DATA_DIR là nguồn sự thật (nằm trong git repo).
        # Mỗi lần server khởi động, ingest_directory() build lại toàn bộ vector store
        # từ các file đó — loại bỏ rủi ro "ChromaDB không sống sót qua redeploy".
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

    def add_documents(self, docs_text: str, source: str = "unknown") -> int:
        """
        Chia tài liệu thành chunks và upsert vào ChromaDB.
        ID ổn định (hash của source + index + nội dung chunk) nên ingest lại nhiều lần
        không tạo bản ghi trùng — giữ nguyên cải tiến bạn đã làm.
        """
        # Chia dữ liệu theo từng section ## thay vì cắt theo số ký tự
        parts = re.split(r"(?=^## )", docs_text, flags=re.MULTILINE)

        chunks = []

        for part in parts:
            part = part.strip()

            if part:
                chunks.append(part)
        for i, chunk in enumerate(chunks):
            if "HIỆU TRƯỞNG" in chunk.upper() or "NGUYỄN TRUNG KIÊN" in chunk.upper():
                print(f"\n===== CHUNK {i} =====")
                print(chunk)
                print("====================")
        documents, ids, metadatas = [], [], []
        for idx, chunk in enumerate(chunks):
            raw_id = f"{source}_{idx}_{chunk}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

            documents.append(chunk)
            ids.append(f"doc_{chunk_id}")
            metadatas.append({"source": source, "chunk_id": idx})

        if documents:
            logger.info(f"Bắt đầu upsert {len(documents)} chunks...")
            self.collection.upsert(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            logger.info("Upsert hoàn tất.")

        return len(chunks)

    def ingest_directory(self, data_dir: str = None) -> dict:
        """Nạp toàn bộ .txt/.pdf trong thư mục. Gọi lúc server startup và từ load_data.py."""
        data_dir = data_dir or config.DATA_DIR
        stats = {"files": 0, "chunks": 0, "skipped": []}

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

    def retrieve_with_threshold(self, query_text: str):
        """
        Hybrid retrieval:
        - Vector search trên toàn bộ corpus
        - Keyword/phrase matching để ưu tiên chunk có từ khóa của câu hỏi
        - Lọc theo threshold
        """

        if self.collection.count() == 0:
            return None

        candidate_k = self.collection.count()

        results = self.collection.query(
            query_texts=[query_text],
            n_results=candidate_k,
            include=["documents", "distances", "metadatas"],
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return None

        # =========================
        # KEYWORD MATCHING
        # =========================

        import re

        query_lower = query_text.lower()

        # Lấy các cụm từ có nghĩa trong câu hỏi.
        # Giữ lại cả cụm từ 2-3 từ vì chúng quan trọng hơn từ đơn.
        words = re.findall(r"\w+", query_lower)

        query_terms = set(words)

        # Một số từ quá chung, không có nhiều giá trị khi matching
        stopwords = {
            "có", "những", "các", "là", "của", "trường",
            "nào", "ai", "gì", "cho", "về", "hiện",
            "nay", "được", "bao", "nhiêu", "theo",
            "thông", "tin", "hãy", "cho", "biết"
        }

        query_terms -= stopwords

        candidates = []

        for doc, distance, metadata in zip(
            documents, distances, metadatas
        ):
            doc_lower = doc.lower()

            # Đếm số từ khóa thực sự xuất hiện trong document
            keyword_matches = sum(
                1
                for term in query_terms
                if term in doc_lower
            )

            # Bonus mạnh hơn nếu toàn bộ một cụm quan trọng xuất hiện
            phrase_bonus = 0

            # Sinh MỌI cụm liền kề 2-4 từ (sliding window) từ danh sách từ gốc,
            # thay vì chỉ bắt 1 cụm dài nhất không chồng lấn như bản trước.
            # Cách cũ dùng re.findall không chồng lấn có thể "nuốt" luôn stopword
            # phía sau (vd query "Nguyễn Thu Hiền là ai" -> chỉ trích được cụm
            # "nguyễn thu hiền là", KHÔNG BAO GIỜ khớp chính xác với "nguyễn thu
            # hiền" trong tài liệu) -> tên người bị hỏi không được cộng bonus dù
            # khớp 100%. Sliding window đảm bảo cụm tên đúng (không dính thêm
            # "là"/"ai") luôn được thử.
            important_phrases = set()
            for n in (2, 3, 4):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    if phrase not in stopwords:
                        important_phrases.add(phrase)

            for phrase in important_phrases:
                if phrase in doc_lower:
                    phrase_bonus += 0.20

            # Keyword càng khớp -> score càng thấp -> ưu tiên
            score = (
                distance
                - keyword_matches * 0.10
                - phrase_bonus
            )

            # QUAN TRỌNG: lọc theo SCORE (đã cộng bonus khớp từ khóa/cụm từ),
            # KHÔNG lọc theo distance thô như bản trước.
            # Lý do (đã xác nhận bằng log thực tế của người dùng): bản trước
            # loại chunk NGAY khi distance thô > threshold, TRƯỚC KHI biết
            # chunk đó có khớp tên/cụm từ chính xác hay không. Hệ quả: câu hỏi
            # "Nguyễn Thu Hiền là ai" bị loại đúng chunk Tiếng Anh chứa tên cô
            # (distance thô hơi cao) và trả về nhầm chunk Toán/Ngữ văn (distance
            # thấp hơn nhưng không liên quan) — dù tên khớp 100% trong text.
            # Cho phép keyword/phrase bonus "cứu" ứng viên trước khi áp
            # threshold giải quyết đúng lỗi này, mà không nới lỏng threshold
            # cho các chunk hoàn toàn không khớp từ khóa nào (score của chúng
            # không đổi vì bonus = 0).
            if score > config.SIMILARITY_THRESHOLD:
                continue

            candidates.append({
                "doc": doc,
                "distance": distance,
                "metadata": metadata,
                "score": score,
                "keyword_matches": keyword_matches,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda x: x["score"])

        selected = candidates[:config.TOP_K]

        valid_docs = []
        sources = []

        for item in selected:
            doc = item["doc"]
            distance = item["distance"]
            metadata = item["metadata"]

            valid_docs.append(doc)

            sources.append({
                "source": metadata.get("source", "unknown"),
                "chunk_id": metadata.get("chunk_id", -1),
                "distance": round(distance, 4),
                "snippet": doc[:200],
            })

        return {
            "context": "\n\n".join(valid_docs),
            "sources": sources,
            "images": self.image_index.find(query_text),
        }