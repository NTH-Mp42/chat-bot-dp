import os
import secrets
import time
import logging
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Header, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import llm_router
from rag_engine import RAGEngine
import tkb_store
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI(title="Chatbot Hỗ Trợ Học Tập & Thông Tin Trường", version="1.0")

# CORS mở toàn bộ cho các phương thức cần thiết
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()

NO_INFO_ANSWER = "Xin lỗi, hiện tại tôi chưa có thông tin này trong hệ thống dữ liệu của nhà trường."
# Các câu hỏi có thể trả lời trực tiếp bằng LLM,
# không cần truy xuất dữ liệu trường.
DIRECT_KEYWORDS = [
    "chào",
    "hello",
    "hi",
    "cảm ơn",
    "thanks",
    "thank you",
    "bạn là ai",
    "bạn có thể làm gì",
    "bằng bao nhiêu",
    "pythagore",
    "định luật",
    "cách làm bài",
    "bằng mấy",
    "làm thế nào",
    "làm bài",
    "bằng mấy",
    "từ mới",
    "nghĩa",
    "khai triển",
]
RAG_KEYWORDS = [
    "trường",
    "thpt hoài đức a",
    "hoài đức a",
    "hiệu trưởng",
    "thành lập",
    "kỷ niệm",
    "kỉ niệm",
    "clb",
    "câu lạc bộ",
    "giáo viên",
    "học sinh",
    "lịch học",
    "nội quy",
    "phòng học",
    "cơ sở vật chất",
]
TKB_KEYWORDS = [
    "thời khóa biểu",
    "thời khoá biểu",
    "tkb",
    "lịch học",
    "lịch dạy",
    "tiết học",
    "tiết mấy",
    "thứ 2",
    "thứ 3",
    "thứ 4",
    "thứ 5",
    "thứ 6",
    "thứ 7",
]

def _keyword_in_text(keyword: str, text: str) -> bool:
    r"""
    So khớp từ khóa theo RANH GIỚI TỪ (word boundary), không dùng substring
    thô như bản trước. Lý do: `if "hi" in q` khớp nhầm vào giữa các từ như
    "Hiền", "khi", "thi" -> chatbot tưởng học sinh đang chào rồi trả lời
    "direct" thay vì tra RAG, dù câu hỏi thực chất hỏi về 1 giáo viên cụ thể.
    (?<!\w)...(?!\w) đảm bảo ký tự ngay trước/sau từ khóa không phải là 1
    ký tự chữ/số khác -> "hi" chỉ khớp khi đứng riêng, không khớp bên trong
    "Hiền".
    """
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return re.search(pattern, text) is not None


def detect_mode(query: str) -> str:
    q = query.lower().strip()

    # =========================
    # TKB
    # =========================
    # Có tên lớp 10A..., 11A..., 12...
    has_class = bool(
        re.search(r"\b(?:10|11|12)[a-zđ]\d{1,2}\b", q, re.IGNORECASE)
    )

    tkb_words = [
        "thời khóa biểu",
        "tkb",
        "lịch học",
        "lịch dạy",
        "tiết học",
        "ai dạy",
    ]

    has_tkb_word = any(
        _keyword_in_text(word, q)
        for word in tkb_words
    )

    # Ví dụ:
    # "11A1 thứ 2 học gì?"
    # "11A1 tiết 3 ai dạy?"
    # "lịch dạy của thầy Quyết"
    if has_class or has_tkb_word:
        return "tkb"

    # =========================
    # RAG
    # =========================
    for keyword in RAG_KEYWORDS:
        if _keyword_in_text(keyword, q):
            return "rag"

    # =========================
    # DIRECT
    # =========================
    for keyword in DIRECT_KEYWORDS:
        if _keyword_in_text(keyword, q):
            return "direct"

    return "rag"
def handle_tkb_query(query: str) -> str:
    q = query.lower().strip()

    # ========================================================
    # 1. TÌM LỚP
    # ========================================================

    class_match = re.search(
        r"\b((?:10|11|12)[a-zđ]\d{1,2})\b",
        q,
        re.IGNORECASE
    )

    lop = class_match.group(1).upper() if class_match else None

    # ========================================================
    # 2. XÁC ĐỊNH THỨ
    # ========================================================

    thu = None

    thu_match = re.search(
        r"\bthứ\s*([2-7])\b",
        q,
        re.IGNORECASE
    )

    if thu_match:
        thu = int(thu_match.group(1))

    # ========================================================
    # 3. XÁC ĐỊNH TIẾT
    # ========================================================

    tiet = None

    tiet_match = re.search(
        r"\b(?:tiết|tiet)\s*([1-6])\b",
        q,
        re.IGNORECASE
    )

    if tiet_match:
        tiet = int(tiet_match.group(1))

    # ========================================================
    # 4. HỎI TKB CỦA LỚP
    # ========================================================

    if lop:

        # 11A1 Thứ 2 tiết 3
        if thu is not None and tiet is not None:
            result = tkb_store.ai_day(
                lop,
                thu,
                tiet
            )

            if not result:
                return (
                    f"Lớp {lop} không có tiết học "
                    f"vào Thứ {thu}, tiết {tiet}."
                )

            return (
                f"Thứ {thu}, tiết {tiet} của lớp {lop}:\n"
                f"Môn: {result['mon']}\n"
                f"Giáo viên: {result['ho_ten']}\n"
                f"Mã GV: {result['gv_code']}"
            )

        # 11A1 Thứ 2
        if thu is not None:
            result = tkb_store.lich_lop_theo_ngay(lop, thu)

            if not result:
                return f"Lớp {lop} không có tiết học vào Thứ {thu}."

            lines = [
                f"Thời khóa biểu lớp {lop} - Thứ {thu}:"
            ]

            for item in result:
                lines.append(
                    f"- Tiết {item['tiet']}: "
                    f"{item['mon']} - "
                    f"{item['ho_ten']} "
                    f"({item['buoi']})"
                )

            return "\n".join(lines)

        result = tkb_store.tiet_trong_lop(lop)

        if not result:
            return f"Không tìm thấy thời khóa biểu của lớp {lop}."

        lines = [
            f"Thời khóa biểu lớp {lop}:"
        ]

        for item in result:
            lines.append(
                f"- Thứ {item['thu']}, tiết {item['tiet']}: "
                f"{item['mon']} - "
                f"{item['ho_ten']} "
                f"({item['buoi']})"
            )

        return "\n".join(lines)

    # ========================================================
    # 5. HỎI LỊCH GIÁO VIÊN
    # ========================================================

    # "lịch dạy của Nguyễn Hữu Quyết"
    teacher_query = re.search(
        r"(?:lịch\s+dạy\s+của|lịch\s+của|lịch\s+dạy|lịch|thời\s+khóa\s+biểu\s+của)"
        r"\s+(?:(?:thầy|cô)\s+)?(.+)",
        query,
        re.IGNORECASE
    )

    if teacher_query:
        name = teacher_query.group(1).strip(" ?.")

        matches = tkb_store.tim_giao_vien_theo_ten(name)

        if not matches:
            return (
                f"Không tìm thấy giáo viên có tên "
                f"“{name}” trong dữ liệu TKB."
            )

        if len(matches) > 1:
            lines = [
                f"- {x['ho_ten']} | {x['mon']} | Mã GV: {x['gv_code']}"
                for x in matches
            ]

            return (
                "Có nhiều giáo viên trùng tên:\n"
                + "\n".join(lines)
                + "\n\nVui lòng cho biết mã giáo viên nếu muốn "
                  "xem chính xác lịch dạy."
            )

        gv_code = matches[0]["gv_code"]
        teacher_name = matches[0]["ho_ten"]

        rows = tkb_store.lich_giao_vien(gv_code)

        if not rows:
            return f"Không tìm thấy lịch dạy của {teacher_name}."

        # Nếu người dùng hỏi theo ngày cụ thể
        if thu is not None:
            rows = [
                item for item in rows
                if item["thu"] == thu
            ]

            if not rows:
                return (
                    f"{teacher_name} không có tiết dạy "
                    f"vào Thứ {thu}."
                )

            lines = [
                f"Lịch dạy của {teacher_name} - Thứ {thu}:"
            ]

            for item in sorted(rows, key=lambda x: x["tiet"]):
                lines.append(
                    f"- Tiết {item['tiet']}: "
                    f"{item['mon']} - lớp {item['lop']} "
                    f"({item['buoi']})"
                )

            return "\n".join(lines)

        # Không hỏi ngày → trả toàn bộ lịch
        lines = [
            f"Lịch dạy của {teacher_name}:"
        ]

        for item in sorted(
            rows,
            key=lambda x: (x["thu"], x["tiet"])
        ):
            lines.append(
                f"- Thứ {item['thu']}, tiết {item['tiet']}: "
                f"{item['mon']} - lớp {item['lop']} "
                f"({item['buoi']})"
            )

        return "\n".join(lines)

    # ========================================================
    # 6. KHÔNG ĐỦ THÔNG TIN
    # ========================================================

    return (
        "Bạn có thể hỏi theo dạng:\n"
        "- Lớp 11A1 Thứ 2 học gì?\n"
        "- Lớp 11A1 Thứ 2 tiết 3 ai dạy?\n"
        "- Lịch dạy của Nguyễn Hữu Quyết?\n"
        "- Thầy Quyết dạy những lớp nào?"
    )
@app.on_event("startup")
def startup_event():
    # Fail sớm, rõ ràng nếu thiếu key — tránh lỗi mơ hồ lúc có người chat thử.
    config.validate_config()
    stats = rag.ingest_directory()
    logger.info(f"Khởi tạo RAG: {stats['files']} file, {stats['chunks']} đoạn, bỏ qua: {stats['skipped']}")
    if stats["files"] == 0:
        logger.warning(f"KHÔNG có file dữ liệu nào được nạp từ '{config.DATA_DIR}'. Chatbot sẽ luôn trả lời 'chưa có thông tin'.")


class ChatRequest(BaseModel):
    query: str


'''@app.get("/")
def read_root():
    return {"status": "ok", "message": "Chatbot API đang hoạt động bình thường!", "chunks_loaded": rag.collection.count()}
'''
@app.get("/api")
def read_root():
    return {
        "status": "ok",
        "message": "Chatbot API đang hoạt động bình thường!",
        "chunks_loaded": rag.collection.count()
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "chunks_loaded": rag.collection.count()}



@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    start_time = time.perf_counter()
    query = req.query.strip()

    print(">>> QUERY:", repr(query))
    print(">>> MODE:", detect_mode(query))

    if not query:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài.")

    # Chỉ gọi retrieval MỘT lần (bản trước gọi 2 lần cho cùng 1 câu hỏi, tốn gấp đôi thời gian).
    mode = detect_mode(query)

    logger.info(f"Query mode: {mode} | Query: {query}")
        # =========================
    # TKB MODE
    # =========================
    if mode == "tkb":
        try:
            answer = handle_tkb_query(query)
        except Exception:
            logger.exception("Lỗi khi truy vấn TKB")
            raise HTTPException(
                status_code=500,
                detail="Lỗi khi truy vấn thời khóa biểu."
            )

        latency = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        return {
            "answer": answer,
            "provider": "sqlite",
            "model": "tkb_store",
            "latency_ms": latency,
            "sources": ["data/tkb.db"],
            "images": [],
        }
    # =========================
    # DIRECT MODE
    # =========================
    if mode == "direct":
        try:
            result = llm_router.generate(query, mode="direct")
        except RuntimeError as e:
            logger.error(str(e))
            raise HTTPException(
                status_code=503,
                detail="Hiện tại chatbot không thể xử lý yêu cầu. Vui lòng thử lại sau."
            )

        latency = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "answer": result.text,
            "provider": result.provider,
            "model": result.model,
            "latency_ms": latency,
            "sources": [],
        }


    # =========================
    # RAG MODE
    # =========================
    try:
        rag_result = rag.retrieve_with_threshold(query)
    except Exception:
        logger.exception("Lỗi khi truy xuất RAG")
        raise HTTPException(
            status_code=500,
            detail="Lỗi khi truy xuất dữ liệu."
        )

    if not rag_result:
        latency = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "answer": NO_INFO_ANSWER,
            "images": [],
            "provider": "none",
            "model": "none",
            "latency_ms": latency,
            "sources": [],
        }

    context = rag_result["context"]
    sources = rag_result["sources"]

    print("\n===== RAG CONTEXT =====")
    print(context)
    print("=======================\n")

    current_year = datetime.now().year

    images = rag_result.get("images", [])
    image_note = ""

    if images:
        titles = "; ".join(i["title"] for i in images)
        image_note = (
            f"\nẢNH ĐÍNH KÈM (hệ thống tự hiển thị ngay bên dưới câu trả lời của bạn): {titles}\n"
            "Nếu người dùng muốn xem ảnh, hãy giới thiệu ngắn gọn và nói ảnh ở bên dưới. "
            "KHÔNG được nói là không có ảnh.\n"
        )

    context = rag_result["context"]

    logger.info(
        f"RAG context: {len(context):,} chars | "
        f"TOP_K={config.TOP_K}"
    )

    full_prompt = f"""
    NĂM HIỆN TẠI: {current_year}

    {image_note}

    NGỮ CẢNH:
    {context}

    CÂU HỎI:
    {query}
    """

    logger.info(f"Full prompt: {len(full_prompt):,} chars")
    try:
        result = llm_router.generate(full_prompt, mode="rag")
    except RuntimeError as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=503,
            detail="Hiện tại chatbot không thể xử lý yêu cầu. Vui lòng thử lại sau."
        )

    latency = round((time.perf_counter() - start_time) * 1000, 2)

    # Dùng đúng biến `sources` đã lấy được ở trên (bản trước hardcode [] ở đây — bug).
    return {
        "answer": result.text,
        "images": rag_result.get("images", []),
        "provider": result.provider,
        "model": result.model,
        "latency_ms": latency,
        "sources": sources,
    }



# =========================
# ADMIN AUTH & DATA MANAGEMENT
# =========================
security = HTTPBearer(auto_error=False)
active_admin_tokens = set()

class LoginRequest(BaseModel):
    username: str
    password: str

class FileSaveRequest(BaseModel):
    content: str

def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_admin_token: str = Header(None, alias="X-Admin-Token")
):
    token = credentials.credentials if credentials else x_admin_token
    if not token or token not in active_admin_tokens:
        raise HTTPException(
            status_code=401,
            detail="Phiên đăng nhập quản trị không hợp lệ hoặc đã hết hạn."
        )
    return True

@app.post("/api/admin/login")
def admin_login(req: LoginRequest):
    if req.username == config.ADMIN_USERNAME and req.password == config.ADMIN_PASSWORD:
        token = secrets.token_hex(24)
        active_admin_tokens.add(token)
        return {
            "success": True,
            "token": token,
            "username": req.username,
            "message": "Đăng nhập quản trị thành công."
        }
    raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không chính xác.")

@app.post("/api/admin/logout")
def admin_logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_admin_token: str = Header(None, alias="X-Admin-Token")
):
    token = credentials.credentials if credentials else x_admin_token
    if token and token in active_admin_tokens:
        active_admin_tokens.remove(token)
    return {"success": True, "message": "Đã đăng xuất thành công."}

@app.get("/api/admin/status")
def admin_status(_: bool = Depends(get_current_admin)):
    files_count = len([f for f in os.listdir(config.DATA_DIR) if os.path.isfile(os.path.join(config.DATA_DIR, f))]) if os.path.exists(config.DATA_DIR) else 0
    return {
        "status": "ok",
        "authenticated": True,
        "chunks_loaded": rag.collection.count(),
        "files_count": files_count
    }

@app.get("/api/admin/files")
def admin_list_files(_: bool = Depends(get_current_admin)):
    if not os.path.exists(config.DATA_DIR):
        return {"files": []}
    files = []
    for f in sorted(os.listdir(config.DATA_DIR)):
        f_path = os.path.join(config.DATA_DIR, f)
        if os.path.isfile(f_path) and (f.lower().endswith(".txt") or f.lower().endswith(".pdf")):
            st = os.stat(f_path)
            files.append({
                "name": f,
                "size": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).strftime("%d/%m/%Y %H:%M:%S"),
                "is_editable": f.lower().endswith(".txt"),
                "extension": os.path.splitext(f)[1].lower()
            })
    return {"files": files}

@app.get("/api/admin/files/{filename}")
def admin_get_file(filename: str, _: bool = Depends(get_current_admin)):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(config.DATA_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp yêu cầu.")
    if not safe_name.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ chỉnh sửa trực tiếp tệp văn bản định dạng .txt.")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"filename": safe_name, "content": content}
    except Exception as e:
        logger.exception(f"Lỗi khi đọc file {safe_name}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file: {str(e)}")

@app.post("/api/admin/files/{filename}")
def admin_save_file(filename: str, req: FileSaveRequest, _: bool = Depends(get_current_admin)):
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ lưu tệp văn bản định dạng .txt.")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    file_path = os.path.join(config.DATA_DIR, safe_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.content)
        stats = rag.ingest_directory(rebuild=True)
        return {
            "success": True,
            "message": f"Đã lưu tệp '{safe_name}' và cập nhật dữ liệu thành công.",
            "stats": stats,
            "chunks_count": rag.collection.count()
        }
    except Exception as e:
        logger.exception(f"Lỗi khi lưu tệp {safe_name}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu tệp: {str(e)}")

@app.post("/api/admin/upload")
async def admin_upload_file(file: UploadFile = File(...), _: bool = Depends(get_current_admin)):
    safe_name = os.path.basename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in [".txt", ".pdf"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tải lên tệp .txt hoặc .pdf.")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    file_path = os.path.join(config.DATA_DIR, safe_name)
    try:
        content_bytes = await file.read()
        with open(file_path, "wb") as f:
            f.write(content_bytes)
        stats = rag.ingest_directory(rebuild=True)
        return {
            "success": True,
            "message": f"Đã tải lên tệp '{safe_name}' thành công và nạp vào hệ thống.",
            "filename": safe_name,
            "stats": stats,
            "chunks_count": rag.collection.count()
        }
    except Exception as e:
        logger.exception(f"Lỗi khi tải lên tệp {safe_name}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải tệp: {str(e)}")

@app.delete("/api/admin/files/{filename}")
def admin_delete_file(filename: str, _: bool = Depends(get_current_admin)):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(config.DATA_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp để xóa.")
    try:
        os.remove(file_path)
        stats = rag.ingest_directory(rebuild=True)
        return {
            "success": True,
            "message": f"Đã xóa tệp '{safe_name}' và cập nhật lại hệ thống.",
            "stats": stats,
            "chunks_count": rag.collection.count()
        }
    except Exception as e:
        logger.exception(f"Lỗi khi xóa tệp {safe_name}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa tệp: {str(e)}")

@app.post("/api/admin/reindex")
def admin_reindex(_: bool = Depends(get_current_admin)):
    try:
        stats = rag.ingest_directory(rebuild=True)
        return {
            "success": True,
            "message": "Đã đồng bộ lại toàn bộ dữ liệu thành công.",
            "stats": stats,
            "chunks_count": rag.collection.count()
        }
    except Exception as e:
        logger.exception("Lỗi khi reindex")
        raise HTTPException(status_code=500, detail=f"Lỗi khi đồng bộ dữ liệu: {str(e)}")

# Phục vụ frontend từ chính FastAPI — một URL Railway duy nhất cho cả frontend và
# backend, không lỗi CORS giữa 2 domain, không cần deploy HTML riêng.
# Đặt SAU các route API để "/chat", "/health" không bị StaticFiles nuốt mất.
@app.get("/api/images")
def list_images():
    return rag.image_index.all_grouped()
app.mount("/", StaticFiles(directory="static", html=True), name="static")