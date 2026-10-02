"""
image_index.py - Thư viện ảnh + tra cứu ảnh minh họa cho chatbot.

Nguyên tắc thiết kế
-------------------
1. QUÉT THƯ MỤC LÀ NGUỒN SỰ THẬT VỀ "CÓ ẢNH NÀO". Khi khởi động, mọi ảnh trong
   thư mục chứa manifest (kể cả thư mục con) đều vào Thư viện ảnh. Mỗi thư mục
   con = một nhóm; nhãn nhóm lấy từ "categories" trong manifest.
2. MANIFEST CHỈ BỔ SUNG THÔNG TIN (title, aliases, exclude) cho từng ảnh. Ảnh
   chưa khai báo vẫn có trong thư viện (tiêu đề suy từ tên file) nhưng KHÔNG
   bao giờ tự hiện trong câu trả lời của chatbot.
3. ĐỘ CHÍNH XÁC > ĐỘ PHỦ. Chatbot chỉ gửi ảnh khi câu hỏi chứa NGUYÊN VẸN một
   alias do con người khai báo. Hỏi chung ("các câu lạc bộ") -> không gửi ảnh;
   hỏi riêng ("CLB bóng đá") -> gửi ảnh đó. Không có khớp "mờ", không đoán theo nhóm.
4. KHÔNG tốn thêm request LLM: logic chạy tất định trong Python.
5. KHÔNG làm sập chatbot: manifest thiếu/lỗi -> ghi log, vẫn quét thư mục.
   Khi reload() mà dữ liệu mới lỗi -> giữ lại bản đang chạy tốt.

Cấu trúc thư mục
----------------
static/images/
    manifest.json
    so_do/so_do.jpg
    clb/clb_bong_da.jpg          <- thư mục con "clb" = nhóm "clb"

Cấu trúc manifest.json
----------------------
{
  "categories": {                          # nhãn + thứ tự của các nhóm (khóa = tên thư mục)
    "clb": {"label": "Câu lạc bộ", "order": 3}
  },
  "images": {                              # khóa = đường dẫn tương đối so với manifest
    "clb/clb_bong_da.jpg": {
      "title": "Câu lạc bộ Bóng đá",       # tùy chọn: chú thích ảnh (mặc định suy từ tên file)
      "aliases": ["bóng đá"],              # tùy chọn: cụm trong câu hỏi sẽ kích hoạt ảnh này
      "exclude": ["phó hiệu trưởng"]       # tùy chọn: câu hỏi chứa cụm này thì KHÔNG hiện ảnh
    }
  }
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
# Alias chỉ gồm 1 từ ngắn (vd "lac" trong "câu lạc bộ") dễ khớp nhầm vào từ thông dụng.
SHORT_SINGLE_WORD = 3

# (cụm đã chuẩn hóa, regex đã biên dịch)
_Phrase = tuple[str, "re.Pattern[str]"]
_FAR = 10**9  # thứ tự cho ảnh chưa khai báo: xếp sau ảnh đã khai báo


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
        if " " not in norm and len(norm) <= SHORT_SINGLE_WORD:
            logger.warning(
                "[%s] Cụm %r chỉ 1 từ ngắn, dễ khớp nhầm vào từ thông dụng "
                "(vd 'lac' nằm trong 'câu lạc bộ'). Hãy dùng cụm dài hơn nếu có thể.",
                owner, raw,
            )
        seen.add(norm)
        compiled.append((norm, re.compile(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])")))
    return tuple(compiled)


def _title_from_filename(rel_path: str) -> str:
    """'clb/clb_bong_da.jpg' -> 'Clb bong da' (chỉ là phương án dự phòng; nên khai báo title)."""
    stem = Path(rel_path).stem
    words = re.sub(r"[_\-]+", " ", stem).strip()
    return words[:1].upper() + words[1:] if words else rel_path


# --------------------------------------------------------------------------- #
# Cấu trúc dữ liệu
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Category:
    key: str
    label: str
    order: int


@dataclass(frozen=True)
class ImageEntry:
    path: str  # đường dẫn tương đối, dùng '/' (vd 'clb/clb_bong_da.jpg')
    url: str
    title: str
    category: str | None  # tên thư mục con chứa ảnh; None nếu ảnh nằm ngay thư mục gốc
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
        self.root = self.manifest_path.parent
        self.url_prefix = url_prefix.rstrip("/")
        self.max_images = max_images
        # Gộp vào 1 tuple để reload() hoán đổi nguyên tử; luồng đang đọc
        # (FastAPI chạy endpoint sync trong threadpool) luôn thấy trạng thái nhất quán.
        self._state: tuple[list[ImageEntry], dict[str, Category]] = ([], {})
        self._loaded = False
        self.reload()

    # ----- nạp & kiểm tra dữ liệu ------------------------------------------ #
    def reload(self) -> None:
        """Quét lại thư mục + đọc lại manifest. Lỗi thì giữ nguyên trạng thái đang chạy tốt."""
        raw = self._read_manifest()
        if raw is None:  # manifest hỏng và đã có dữ liệu tốt trước đó
            return

        try:
            on_disk = self._scan_disk()
        except OSError as exc:
            logger.error("Không quét được thư mục ảnh %s: %s", self.root, exc)
            return

        categories = self._parse_categories(raw.get("categories", {}))
        entries = self._build_entries(on_disk, raw.get("images", {}), categories)
        self._check_categories(entries, categories)

        self._state = (entries, categories)
        self._loaded = True
        logger.info("Đã nạp %d ảnh, %d nhóm từ %s", len(entries), len(categories), self.root)

    def _read_manifest(self) -> dict | None:
        """Trả dict manifest. Thiếu/lỗi: trả {} (vẫn quét thư mục) nếu chưa có dữ liệu tốt,
        trả None (giữ bản cũ) nếu đã có."""
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("manifest phải là một object có 'categories' và 'images'")
            return raw
        except FileNotFoundError:
            logger.warning("Không thấy manifest: %s. Ảnh vẫn được quét nhưng không có alias/nhãn nhóm.",
                           self.manifest_path)
            return {}
        except (OSError, ValueError) as exc:  # JSONDecodeError là con của ValueError
            if self._loaded:
                logger.error("Manifest ảnh lỗi, giữ nguyên dữ liệu cũ: %s", exc)
                return None
            logger.error("Manifest ảnh lỗi (%s). Ảnh vẫn được quét nhưng không có alias/nhãn nhóm.", exc)
            return {}

    def _scan_disk(self) -> list[str]:
        """Danh sách đường dẫn tương đối (dạng 'thu_muc/anh.jpg') của mọi ảnh dưới thư mục gốc."""
        found = []
        for p in self.root.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue  # file/thư mục ẩn
            if p.suffix.lower() in ALLOWED_EXTENSIONS:
                found.append(rel.as_posix())
        return sorted(found)

    @staticmethod
    def _parse_categories(raw) -> dict[str, Category]:
        categories: dict[str, Category] = {}
        if not isinstance(raw, dict):
            logger.warning("'categories' phải là object, bỏ qua.")
            return categories
        for pos, (key, spec) in enumerate(raw.items()):
            if not isinstance(spec, dict):
                logger.warning("[category:%s] Khai báo sai định dạng, bỏ qua.", key)
                continue
            order = spec.get("order")
            categories[key] = Category(
                key=key,
                label=str(spec.get("label") or key),
                order=order if isinstance(order, int) else pos,
            )
        return categories

    def _build_entries(self, on_disk: list[str], raw_images, categories: dict[str, Category]) -> list[ImageEntry]:
        if not isinstance(raw_images, dict):
            logger.warning("'images' phải là object khóa theo đường dẫn tương đối "
                           "(vd \"clb/clb_bong_da.jpg\": {...}), bỏ qua toàn bộ thông tin bổ sung.")
            raw_images = {}

        disk_set = set(on_disk)
        declared_order = {path: i for i, path in enumerate(raw_images)}

        for path in raw_images:
            if path not in disk_set:
                logger.warning("[%s] Có trong manifest nhưng KHÔNG có trên đĩa "
                               "(sai tên file/thư mục? phân biệt hoa-thường).", path)

        entries: list[ImageEntry] = []
        alias_owner: dict[str, str] = {}

        for path in on_disk:
            spec = raw_images.get(path)
            if spec is None:
                spec = {}
            elif not isinstance(spec, dict):
                logger.warning("[%s] Khai báo sai định dạng, dùng giá trị mặc định.", path)
                spec = {}

            parts = path.split("/")
            category = parts[0] if len(parts) > 1 else None

            title = str(spec.get("title") or "").strip()
            if not title:
                title = _title_from_filename(path)
                logger.info("[%s] Chưa có 'title', tạm dùng %r.", path, title)

            aliases = _compile_phrases(spec.get("aliases"), path)
            excludes = _compile_phrases(spec.get("exclude"), path)

            if path not in declared_order:
                logger.info("[%s] Chưa khai báo trong manifest: chỉ hiện trong thư viện ảnh.", path)

            # Alias trùng giữa 2 ảnh -> câu hỏi đó sẽ ra cả 2 ảnh, thường là nhầm lẫn khi khai báo.
            for norm, _ in aliases:
                other = alias_owner.setdefault(norm, path)
                if other != path:
                    logger.warning("Alias '%s' trùng giữa %s và %s.", norm, other, path)

            entries.append(
                ImageEntry(
                    path=path,
                    url=f"{self.url_prefix}/{quote(path, safe='/')}",
                    title=title,
                    category=category,
                    aliases=aliases,
                    excludes=excludes,
                    order=declared_order.get(path, _FAR),
                )
            )

        # Ảnh đã khai báo (đúng thứ tự manifest) trước, ảnh chưa khai báo theo tên file.
        entries.sort(key=lambda e: (e.order, e.path))
        return entries

    @staticmethod
    def _check_categories(entries: list[ImageEntry], categories: dict[str, Category]) -> None:
        used = {e.category for e in entries if e.category}
        for key in sorted(used - set(categories)):
            logger.warning("Thư mục '%s' có ảnh nhưng chưa khai báo trong 'categories': "
                           "thư viện sẽ hiện tên thư mục thay vì nhãn.", key)
        for key in sorted(set(categories) - used):
            logger.warning("Nhóm '%s' được khai báo nhưng thư mục không có ảnh nào.", key)

    # ----- truy vấn --------------------------------------------------------- #
    def find(self, query: str, limit: int | None = None) -> list[dict]:
        """
        Trả về tối đa `limit` ảnh mà câu hỏi NHẮC ĐÍCH DANH (khớp nguyên vẹn alias).

        - Ảnh có cụm 'exclude' xuất hiện trong câu hỏi bị loại hẳn.
        - Nếu một alias nằm gọn trong alias DÀI HƠN của ảnh khác thì bị loại
          ('clb' không được kéo theo ảnh khi người dùng hỏi 'clb bóng đá').
        - Không khớp alias nào -> danh sách rỗng. Không có chuyện đoán theo nhóm:
          hỏi chung về một nhóm thì chỉ trả lời bằng chữ.
        """
        limit = self.max_images if limit is None else limit
        entries, _ = self._state
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

        kept = [
            h
            for h in hits
            if not any(
                o[2] is not h[2] and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0])
                for o in hits
            )
        ]
        kept.sort(key=lambda h: (-(h[1] - h[0]), h[2].order, h[2].path))  # cụm dài trước, rồi theo manifest
        return [h[2].to_public() for h in kept[:limit]]

    def all_grouped(self) -> list[dict]:
        """Cho mục 'Thư viện ảnh': nhóm theo thư mục, nhãn lấy từ 'categories', sắp theo 'order'.
        Thứ tự: nhóm đã khai báo (theo order) -> thư mục chưa khai báo (theo tên) -> ảnh ở thư mục gốc."""
        entries, categories = self._state
        by_cat: dict[str | None, list[ImageEntry]] = {}
        for e in entries:
            by_cat.setdefault(e.category, []).append(e)

        declared = sorted((c for c in categories.values() if c.key in by_cat), key=lambda c: (c.order, c.key))
        undeclared = sorted(k for k in by_cat if k is not None and k not in categories)
        ordered_keys: list[str | None] = [c.key for c in declared] + undeclared
        if None in by_cat:
            ordered_keys.append(None)

        groups = []
        for key in ordered_keys:
            label = categories[key].label if key in categories else (key or "Khác")
            groups.append({
                "category": key or "khac",
                "label": label,
                "images": [e.to_public() for e in by_cat[key]],
            })
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
    for group in index.all_grouped():
        print(f"[{group['label']}] " + ", ".join(i["title"] for i in group["images"]))
    for question in sys.argv[2:]:
        print(f"\n? {question}")
        for img in index.find(question) or [{"title": "(không có ảnh)"}]:
            print("   ->", img["title"])