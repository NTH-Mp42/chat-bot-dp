import time
import logging
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import llm_router
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI(title="Chatbot Hỗ Trợ Học Tập & Thông Tin Trường", version="1.0")

# CORS mở toàn bộ: FastAPI giờ phục vụ luôn frontend (cùng domain), nên đây chỉ là
# lớp bảo vệ cho ai gọi thẳng API (Postman, app khác) — không còn nhận API key từ
# client nữa (khác bản cũ), nên không có rủi ro lạm dụng quota qua đây.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

rag = RAGEngine()

NO_INFO_ANSWER = "Xin lỗi, hiện tại tôi chưa có thông tin này trong hệ thống dữ liệu của nhà trường."
# Các câu hỏi có thể trả lời trực tiếp bằng LLM,
# không cần truy xuất dữ liệu trường.
DIRECT_KEYWORDS = [
    "xin chào",
    "chào bạn",
    "hello",
    "hi",
    "cảm ơn",
    "thanks",
    "thank you",
    "bạn là ai",
    "bạn có thể làm gì",
    "2 + 2",
    "2+2",
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

    # Câu hỏi liên quan đến thông tin riêng của trường -> RAG
    for keyword in RAG_KEYWORDS:
        if _keyword_in_text(keyword, q):
            return "rag"

    # Câu giao tiếp / câu hỏi đơn giản -> Direct
    for keyword in DIRECT_KEYWORDS:
        if _keyword_in_text(keyword, q):
            return "direct"

    # Mặc định vẫn dùng RAG
    return "rag"
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


# Phục vụ frontend từ chính FastAPI — một URL Railway duy nhất cho cả frontend và
# backend, không lỗi CORS giữa 2 domain, không cần deploy HTML riêng.
# Đặt SAU các route API để "/chat", "/health" không bị StaticFiles nuốt mất.
@app.get("/api/images")
def list_images():
    return rag.image_index.all_grouped()
app.mount("/", StaticFiles(directory="static", html=True), name="static")