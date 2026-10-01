import os
from dotenv import load_dotenv
load_dotenv()

# =========================
# LLM MODELS
# =========================
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# =========================
# API KEYS
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Thứ tự thử provider — Gemini trước, Groq fallback (đã xác nhận với người dùng)
PROVIDER_ORDER = [p.strip() for p in os.getenv("PROVIDER_ORDER", "gemini,groq").split(",") if p.strip()]

# =========================
# RAG
# =========================
# Không dùng DB_PATH nữa: ChromaDB chạy in-memory (EphemeralClient) và được build lại
# từ file trong DATA_DIR mỗi lần server khởi động, để không phụ thuộc ổ đĩa persistent
# của hosting free-tier (rủi ro đã audit ở rag_engine.py bản trước).
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "school_data_local")
DATA_DIR = os.getenv("DATA_DIR", "data")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.0"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))
# QUAN TRỌNG: KHÔNG lồng vào DATA_DIR. server.py phục vụ file tĩnh thật sự từ
# thư mục "static/" ở gốc project (app.mount("/", StaticFiles(directory="static"))),
# nên manifest.json VÀ toàn bộ ảnh .jpg phải nằm trong "static/images/" — đúng nơi
# ImageIndex sẽ kiểm tra file có tồn tại hay không (dựa vào thư mục chứa manifest).
# Nếu để IMAGE_MANIFEST trỏ vào DATA_DIR ("data/static/images/..."), ImageIndex sẽ
# không tìm thấy manifest (trừ khi bạn cố tình để 2 bản giống hệt ở cả 2 nơi) và
# thư viện ảnh sẽ luôn rỗng một cách lặng lẽ — không có lỗi nào hiện ra để biết.
IMAGE_MANIFEST = os.path.join("static", "images", "manifest.json")
IMAGE_URL_PREFIX = "/images"

def validate_config():
    """Fail sớm và rõ ràng lúc startup nếu thiếu key cho provider nằm trong PROVIDER_ORDER."""
    missing = []
    if "gemini" in PROVIDER_ORDER and not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if "groq" in PROVIDER_ORDER and not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise RuntimeError(
            f"Thiếu biến môi trường: {', '.join(missing)}. "
            f"Kiểm tra file .env (local) hoặc biến môi trường trên Railway."
        )