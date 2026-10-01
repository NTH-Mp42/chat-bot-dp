"""
image_index.py - Tra cứu ảnh minh họa cho chatbot dựa trên manifest.json.

Nguyên tắc thiết kế
-------------------
1. ĐỘ CHÍNH XÁC > ĐỘ PHỦ. Thà không hiện ảnh còn hơn hiện sai ảnh (nhất là ảnh
   người). Vì vậy chỉ khớp theo CỤM TỪ NGUYÊN VẸN do con người khai báo (alias),
   không dùng độ tương đồng "mờ".
2. KHÔNG tốn thêm request LLM nào: toàn bộ logic chạy tất định trong Python.
3. KHÔNG làm sập chatbot: manifest thiếu/lỗi -> ghi log rõ ràng, chạy tiếp
   (không có ảnh). Khi reload() mà manifest mới lỗi -> giữ lại bản đang chạy tốt.
4. Lỗi dữ liệu phải lộ ra ngay lúc khởi động: file khai báo mà không tồn tại,
   alias trùng giữa 2 ảnh, ảnh nằm trong thư mục nhưng chưa khai báo...

Cấu trúc manifest.json
----------------------
{
  "categories": {
    "clb": {"label": "Câu lạc bộ", "aliases": ["câu lạc bộ", "clb"]}
  },
  "images": [
    {
      "file": "clb_bong_da.jpg",          # bắt buộc, chỉ tên file (không có đường dẫn)
      "title": "Câu lạc bộ Bóng đá",       # bắt buộc, hiện làm chú thích ảnh
      "category": "clb",                   # tùy chọn
      "aliases": ["clb bóng đá"],          # tùy chọn: cụm từ trong câu hỏi sẽ kích hoạt ảnh này
      "exclude": ["phó hiệu trưởng"]       # tùy chọn: nếu câu hỏi chứa cụm này thì KHÔNG hiện ảnh
    }
  ]
}

Chạy thử không cần bật server:
    python image_index.py static/images/manifest.json "Hiệu trưởng là ai?" "CLB bóng đá"
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger("image_index")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
# Cụm khớp quá ngắn (vd "a", "ba") rất dễ khớp nhầm -> bỏ qua và cảnh báo.
MIN_PHRASE_CHARS = 3

# (cụm đã chuẩn hóa, regex đã biên dịch)
_Phrase = tuple[str, "re.Pattern[str]"]


# --------------------------------------------------------------------------- #
# Chuẩn hóa văn bản
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """
    Chữ thường, bỏ dấu tiếng Việt, ký tự lạ -> khoảng trắng.
    'Hiệu trưởng là ai?' -> 'hieu truong la ai'. Nhờ vậy người dùng gõ không
    dấu vẫn khớp, và alias khai báo có dấu hay không dấu đều như nhau.
    (Chữ 'đ' không tách dấu được bằng NFD nên phải thay riêng.)
    """
    text = unicodedata.normalize("NFC", text or "").lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _compile_phrases(phrases, owner: str) -> tuple[_Phrase, ...]:
    """Chuẩn hóa + biên dịch danh sách cụm từ. Khớp theo RANH GIỚI TỪ,
    nên 'kien' không khớp trong 'kiender' và 'clb' không khớp trong 'clbx'."""
    if phrases is None:
        return ()
    if not isinstance(phrases, list):
        logger.warning("[%s] Trường aliases/exclude phải là danh sách, bỏ qua.", owner)
        return ()

    compiled: list[_Phrase] = []
    seen: set[str] = set()
    for raw in phrases:
        norm = normalize(str(raw))
        if len(norm) < MIN_PHRASE_CHARS:
            logger.warning("[%s] Bỏ qua cụm quá ngắn hoặc rỗng: %r", owner, raw)
            continue
        if norm in seen:
            continue
        seen.add(norm)
        compiled.append((norm, re.compile(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])")))
    return tuple(compiled)


# --------------------------------------------------------------------------- #
# Cấu trúc dữ liệu
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Category:
    key: str
    label: str
    aliases: tuple[_Phrase, ...]


@dataclass(frozen=True)
class ImageEntry:
    file: str
    url: str
    title: str
    category: str | None
    aliases: tuple[_Phrase, ...]
    excludes: tuple[_Phrase, ...]
    order: int  # thứ tự khai báo trong manifest, dùng để xếp hạng ổn định

    def to_public(self) -> dict:
        """Chỉ trả những gì frontend cần; không lộ alias/đường dẫn nội bộ."""
        return {"url": self.url, "title": self.title, "category": self.category}


# --------------------------------------------------------------------------- #
# ImageIndex
# --------------------------------------------------------------------------- #
class ImageIndex:
    def __init__(self, manifest_path: str | Path, url_prefix: str = "/images", max_images: int = 6):
        self.manifest_path = Path(manifest_path)
        self.url_prefix = url_prefix.rstrip("/")
        self.max_images = max_images
        # Gộp 2 thứ vào 1 tuple để reload() hoán đổi nguyên tử; luồng đang đọc
        # (FastAPI chạy endpoint sync trong threadpool) luôn thấy trạng thái nhất quán.
        self._state: tuple[list[ImageEntry], dict[str, Category]] = ([], {})
        self.reload()

    # ----- nạp & kiểm tra dữ liệu ------------------------------------------ #
    def reload(self) -> None:
        """Đọc lại manifest. Nếu lỗi, giữ nguyên trạng thái hiện tại."""
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("manifest phải là một object có 'categories' và 'images'")
        except FileNotFoundError:
            logger.warning("Không thấy manifest ảnh: %s (chatbot chạy không có ảnh)", self.manifest_path)
            return
        except (OSError, ValueError) as exc:  # JSONDecodeError là con của ValueError
            logger.error("Manifest ảnh lỗi, giữ nguyên dữ liệu cũ: %s", exc)
            return

        categories = self._parse_categories(raw.get("categories", {}))
        entries = self._parse_images(raw.get("images", []), categories)
        self._audit_orphans(entries)
        self._state = (entries, categories)
        logger.info("Đã nạp %d ảnh, %d nhóm từ %s", len(entries), len(categories), self.manifest_path)

    @staticmethod
    def _parse_categories(raw) -> dict[str, Category]:
        categories: dict[str, Category] = {}
        if not isinstance(raw, dict):
            logger.warning("'categories' phải là object, bỏ qua.")
            return categories
        for key, spec in raw.items():
            if not isinstance(spec, dict):
                logger.warning("[category:%s] Khai báo sai định dạng, bỏ qua.", key)
                continue
            categories[key] = Category(
                key=key,
                label=str(spec.get("label") or key),
                aliases=_compile_phrases(spec.get("aliases"), f"category:{key}"),
            )
        return categories

    def _parse_images(self, raw, categories: dict[str, Category]) -> list[ImageEntry]:
        entries: list[ImageEntry] = []
        if not isinstance(raw, list):
            logger.warning("'images' phải là danh sách, bỏ qua.")
            return entries

        base_dir = self.manifest_path.parent
        seen_files: set[str] = set()
        alias_owner: dict[str, str] = {}

        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                logger.warning("images[%d] sai định dạng, bỏ qua.", idx)
                continue

            file = str(item.get("file") or "").strip()
            title = str(item.get("title") or "").strip()

            # Chặn path traversal: chỉ chấp nhận tên file thuần.
            if not file or "/" in file or "\\" in file or Path(file).name != file:
                logger.warning("images[%d] 'file' không hợp lệ (chỉ nhận tên file): %r", idx, file)
                continue
            if Path(file).suffix.lower() not in ALLOWED_EXTENSIONS:
                logger.warning("[%s] Đuôi file không được hỗ trợ, bỏ qua.", file)
                continue
            if not title:
                logger.warning("[%s] Thiếu 'title', bỏ qua.", file)
                continue
            if file in seen_files:
                logger.warning("[%s] Khai báo trùng file, giữ bản đầu tiên.", file)
                continue
            if not (base_dir / file).is_file():
                logger.warning("[%s] Có trong manifest nhưng KHÔNG tồn tại trên đĩa, bỏ qua.", file)
                continue

            category = item.get("category") or None
            if category and category not in categories:
                logger.warning("[%s] category '%s' chưa được khai báo trong 'categories'.", file, category)

            aliases = _compile_phrases(item.get("aliases"), file)
            excludes = _compile_phrases(item.get("exclude"), file)
            if not aliases and not category:
                logger.info("[%s] Không có alias lẫn category: chỉ hiện trong thư viện ảnh.", file)

            # Alias trùng giữa 2 ảnh -> câu hỏi đó sẽ ra cả 2 ảnh, thường là nhầm lẫn khi khai báo.
            for norm, _ in aliases:
                other = alias_owner.setdefault(norm, file)
                if other != file:
                    logger.warning("Alias '%s' trùng giữa %s và %s.", norm, other, file)

            seen_files.add(file)
            entries.append(
                ImageEntry(
                    file=file,
                    url=f"{self.url_prefix}/{quote(file)}",
                    title=title,
                    category=category,
                    aliases=aliases,
                    excludes=excludes,
                    order=len(entries),
                )
            )
        return entries

    def _audit_orphans(self, entries: list[ImageEntry]) -> None:
        """Cảnh báo ảnh nằm trong thư mục nhưng chưa khai báo (hay gặp khi đổi tên file)."""
        base_dir = self.manifest_path.parent
        try:
            on_disk = {p.name for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS}
        except OSError:
            return
        orphans = sorted(on_disk - {e.file for e in entries})
        if orphans:
            logger.warning("Ảnh có trong thư mục nhưng chưa dùng được (chưa khai báo/khai báo lỗi): %s", orphans)

    # ----- truy vấn --------------------------------------------------------- #
    def find(self, query: str, limit: int | None = None) -> list[dict]:
        """
        Trả về tối đa `limit` ảnh phù hợp với câu hỏi.

        Thứ tự ưu tiên:
          1. Khớp CỤ THỂ theo alias của từng ảnh (vd 'hiệu trưởng', 'clb bóng đá').
             - Ảnh có cụm 'exclude' xuất hiện trong câu hỏi bị loại hẳn.
             - Nếu một alias nằm gọn trong alias DÀI HƠN của ảnh khác thì bị loại
               ('clb' không được kéo theo ảnh khi người dùng hỏi 'clb bóng đá').
          2. Nếu không có khớp cụ thể: khớp theo NHÓM (vd 'các câu lạc bộ' ->
             tất cả ảnh nhóm clb).
          3. Không khớp gì -> danh sách rỗng.
        """
        limit = self.max_images if limit is None else limit
        entries, categories = self._state
        q = normalize(query)
        if limit <= 0 or not q or not entries:
            return []

        # (start, end, entry): span của alias DÀI NHẤT khớp trong câu hỏi
        hits: list[tuple[int, int, ImageEntry]] = []
        for entry in entries:
            if any(p.search(q) for _, p in entry.excludes):
                continue
            best: tuple[int, int] | None = None
            for _, pattern in entry.aliases:
                for m in pattern.finditer(q):
                    if best is None or (m.end() - m.start()) > (best[1] - best[0]):
                        best = (m.start(), m.end())
            if best:
                hits.append((best[0], best[1], entry))

        if hits:
            kept = [
                h
                for h in hits
                if not any(
                    o[2] is not h[2] and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0])
                    for o in hits
                )
            ]
            kept.sort(key=lambda h: (-(h[1] - h[0]), h[2].order))  # cụm dài trước, rồi theo thứ tự manifest
            return [h[2].to_public() for h in kept[:limit]]

        picked: list[ImageEntry] = []
        for cat in categories.values():
            if any(p.search(q) for _, p in cat.aliases):
                picked.extend(e for e in entries if e.category == cat.key)
        return [e.to_public() for e in picked[:limit]]

    def all_grouped(self) -> list[dict]:
        """Cho mục 'Thư viện ảnh': danh sách nhóm theo đúng thứ tự khai báo."""
        entries, categories = self._state
        groups = []
        for cat in categories.values():
            imgs = [e.to_public() for e in entries if e.category == cat.key]
            if imgs:
                groups.append({"category": cat.key, "label": cat.label, "images": imgs})
        rest = [e.to_public() for e in entries if e.category not in categories]
        if rest:
            groups.append({"category": "khac", "label": "Khác", "images": rest})
        return groups

    @property
    def stats(self) -> dict:
        entries, categories = self._state
        return {"images": len(entries), "categories": len(categories)}


# --------------------------------------------------------------------------- #
# CLI: thử nhanh matching mà không cần chạy server
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        sys.exit('Cách dùng: python image_index.py <manifest.json> "câu hỏi 1" "câu hỏi 2" ...')
    index = ImageIndex(sys.argv[1])
    print(index.stats)
    for question in sys.argv[2:]:
        print(f"\n? {question}")
        for img in index.find(question) or [{"title": "(không có ảnh)"}]:
            print("   ->", img["title"])
