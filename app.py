import streamlit as st

import config
import llm_router
from rag_engine import RAGEngine

# Streamlit KHÔNG còn là sản phẩm chính cho giám khảo — đây là công cụ nội bộ để
# đồng đội upload dữ liệu, xem chunks, và test retrieval/LLM trước khi đưa vào data/
# để deploy thật (server.py mới là sản phẩm chính, đọc từ data/ lúc startup).

st.set_page_config(page_title="RAG Dev Console - Trường THPT Hoài Đức A", page_icon="🛠️", layout="centered")
st.title("🛠️ RAG Dev Console")
st.caption("Công cụ nội bộ: nạp dữ liệu, xem chunks, test retrieval & câu trả lời — không phải sản phẩm cho giám khảo.")


@st.cache_resource
def get_rag_engine():
    return RAGEngine()


rag = get_rag_engine()

if "loaded_once" not in st.session_state:
    with st.spinner("Đang nạp dữ liệu có sẵn từ thư mục data/..."):
        stats = rag.ingest_directory()
    st.session_state.loaded_once = True
    st.session_state.last_stats = stats

# --- Sidebar: trạng thái key + nạp dữ liệu mới ---
st.sidebar.header("⚙️ Trạng thái")
st.sidebar.write(f"Gemini key: {'✅ đã cấu hình' if config.GEMINI_API_KEY else '❌ thiếu'}")
st.sidebar.write(f"Groq key: {'✅ đã cấu hình' if config.GROQ_API_KEY else '❌ thiếu'}")
st.sidebar.write(f"Số đoạn (chunks) hiện có: **{rag.collection.count()}**")

st.sidebar.divider()
st.sidebar.header("📂 Nạp dữ liệu mới")
uploaded_file = st.sidebar.file_uploader("Tải file (.txt hoặc .pdf)", type=["txt", "pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Nạp vào hệ thống (test tạm thời)", use_container_width=True):
        with st.spinner("Đang xử lý..."):
            if uploaded_file.name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(uploaded_file)
                content = "\n".join((p.extract_text() or "") for p in reader.pages)
            else:
                content = uploaded_file.read().decode("utf-8")

            num_chunks = rag.add_documents(content, source=uploaded_file.name)
            st.sidebar.success(f"Đã nạp {num_chunks} đoạn từ '{uploaded_file.name}'.")
            st.sidebar.info(
                "⚠️ Đây chỉ là bộ nhớ tạm của phiên Streamlit này. Để đưa vào sản phẩm "
                f"thật, hãy lưu file này vào thư mục `{config.DATA_DIR}/` của repo, "
                "commit, push, rồi để Railway tự redeploy."
            )

st.sidebar.divider()
if st.sidebar.button("🔄 Nạp lại toàn bộ từ data/", use_container_width=True):
    with st.spinner("Đang nạp lại..."):
        stats = rag.ingest_directory()
    st.sidebar.success(f"Đã nạp {stats['files']} file, {stats['chunks']} đoạn.")
    if stats["skipped"]:
        st.sidebar.warning(f"Bỏ qua (lỗi/rỗng): {stats['skipped']}")

# --- Khu vực chính: test retrieval + LLM ---
st.divider()
query = st.text_input("Nhập câu hỏi để test:")

col1, col2 = st.columns(2)
test_retrieval = col1.button("🔎 Chỉ xem retrieval (không gọi LLM)", use_container_width=True)
test_full = col2.button("💬 Test trả lời đầy đủ (RAG + LLM)", use_container_width=True)

if (test_retrieval or test_full) and query.strip():
    rag_result = rag.retrieve_with_threshold(query)

    if not rag_result:
        st.warning("Không tìm thấy đoạn nào đạt ngưỡng similarity — sẽ trả lời 'chưa có thông tin'.")
    else:
        st.subheader("📚 Các đoạn được truy xuất (đạt ngưỡng)")
        for i, src in enumerate(rag_result["sources"], start=1):
            with st.expander(f"{i}. {src['source']} (chunk #{src['chunk_id']}) — distance: {src['distance']}"):
                st.write(src["snippet"])

        if test_full:
            with st.spinner("Đang gọi LLM..."):
                full_prompt = f"NGỮ CẢNH:\n{rag_result['context']}\n\nCÂU HỎI:\n{query}"
                try:
                    result = llm_router.generate(full_prompt)
                    st.subheader("🤖 Câu trả lời")
                    st.markdown(result.text)
                    st.caption(f"Provider: {result.provider} | Model: {result.model}")
                except RuntimeError as e:
                    st.error(f"Tất cả provider đều lỗi: {e}")
elif (test_retrieval or test_full) and not query.strip():
    st.info("Nhập câu hỏi trước đã.")
