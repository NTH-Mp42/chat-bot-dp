"""
Script chạy tay để kiểm tra việc nạp dữ liệu trước khi commit/deploy.
Server thật (server.py) tự gọi rag.ingest_directory() lúc startup — script này
chỉ để bạn kiểm tra nhanh ở local xem file trong data/ có nạp đúng không.
"""
import config
from rag_engine import RAGEngine


def main():
    print(f"Đang khởi tạo RAGEngine, nạp dữ liệu từ '{config.DATA_DIR}'...")
    rag = RAGEngine()
    stats = rag.ingest_directory()

    if stats["files"] == 0:
        print(f"❌ Không nạp được file nào từ '{config.DATA_DIR}'. Kiểm tra lại thư mục có tồn tại và có file .txt/.pdf không.")
        return

    print(f"✅ Đã nạp {stats['files']} file, tổng {stats['chunks']} đoạn.")
    if stats["skipped"]:
        print(f"⚠️ Bỏ qua (lỗi hoặc rỗng): {stats['skipped']}")
    print(f"📊 Tổng số bản ghi trong collection: {rag.collection.count()}")


if __name__ == "__main__":
    main()
