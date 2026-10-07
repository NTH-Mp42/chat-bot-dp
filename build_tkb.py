from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EXCEL_FILE = DATA_DIR / "thoi_khoa_bieu.xlsx"
TEACHER_FILE = DATA_DIR / "giao_vien.txt"
DB_FILE = DATA_DIR / "tkb.db"

SHEETS = {
    "TKB_Lop_Sang": "sang",
    "TKB_Lop_Chieu": "chieu",
}

# Mẫu mã lớp của trường.
# Có thể mở rộng nếu sau này xuất hiện lớp mới.
CLASS_RE = re.compile(
    r"^(10|11|12)[A-Z][0-9]{1,2}$",
    re.IGNORECASE
)

# Ô TKB có dạng:
# Toán - QuyếtT
# VẬT LÝ - NaL
# NGỮ VĂN - HươngV
SUBJECT_CODE_RE = re.compile(
    r"^\s*(.+?)\s+-\s+(\S+)\s*$"
)

# Các kiểu ghi thứ thường gặp.
DAY_RE = re.compile(
    r"^(?:thứ\s*)?([2-7]|hai|ba|bốn|năm|sáu|bảy)$",
    re.IGNORECASE
)


# ============================================================
# NORMALIZE
# ============================================================

def clean_text(value) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_day(value):
    """
    Chuyển:
        2 / Thứ 2 / THỨ 2 -> 2
        7 / Thứ 7 -> 7
        CN / Chủ nhật -> 8
    """

    text = clean_text(value).lower()

    text = text.replace("thứ", "").strip()

    if text in {"2", "3", "4", "5", "6", "7"}:
        return int(text)

    return None


def parse_subject_code(value):
    """
    Parse:
        TOAN - QuyếtT
        Vật lý - NaL

    -> ("TOAN", "QuyếtT")
    """

    text = clean_text(value)

    if not text:
        return None

    match = SUBJECT_CODE_RE.match(text)

    if not match:
        return None

    subject = match.group(1).strip()
    gv_code = match.group(2).strip()

    if not subject or not gv_code:
        return None

    return subject, gv_code


# ============================================================
# TEACHER FILE
# ============================================================

def load_teachers():
    """
    Đọc data/giao_vien.txt.

    Hỗ trợ cả:
        Họ tên | Ngày sinh | Giới tính | Môn | Chức vụ | Mã GV

    và dòng nhân viên không có mã GV.
    """

    if not TEACHER_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file giáo viên: {TEACHER_FILE}"
        )

    teachers = {}
    duplicate_codes = []

    with TEACHER_FILE.open(
        "r",
        encoding="utf-8-sig"
    ) as f:

        for line_no, raw_line in enumerate(f, 1):

            line = raw_line.strip()

            if not line:
                continue

            if "|" not in line:
                continue

            if line.startswith("---") or line.startswith("==="):
                continue

            cols = [
                c.strip()
                for c in line.split("|")
            ]

            # Bỏ header
            if cols[0].lower() in {
                "họ và tên",
                "stt",
            }:
                continue

            # Format chính:
            # name | dob | gender | subject | role | code
            if len(cols) < 5:
                continue

            name = cols[0]
            dob = cols[1]
            gender = cols[2]
            subject = cols[3]
            role = cols[4]

            gv_code = ""

            if len(cols) >= 6:
                gv_code = cols[5].strip()

            # Nhân viên không tham gia TKB có thể không có code.
            if not gv_code:
                continue

            if gv_code in teachers:
                duplicate_codes.append(
                    (
                        line_no,
                        gv_code,
                        teachers[gv_code]["name"],
                        name,
                    )
                )
                continue

            teachers[gv_code] = {
                "gv_code": gv_code,
                "name": name,
                "dob": dob,
                "gender": gender,
                "subject": subject,
                "role": role,
            }

    if duplicate_codes:
        print("\n❌ PHÁT HIỆN MÃ GIÁO VIÊN TRÙNG:")

        for item in duplicate_codes:
            line_no, code, old_name, new_name = item

            print(
                f"  Dòng {line_no}: {code} "
                f"-> {old_name} / {new_name}"
            )

        raise ValueError(
            "Mã giáo viên bị trùng. Không tạo DB."
        )

    print(
        f"✓ Đọc được {len(teachers)} mã giáo viên"
    )

    return teachers


# ============================================================
# CLASS DETECTION
# ============================================================

def is_class_name(value):
    text = clean_text(value)

    if not text:
        return False

    text = text.upper()

    return bool(CLASS_RE.match(text))


def find_class_columns(ws):
    """
    Tìm các cột chứa tên lớp trong phần header.

    Không phụ thuộc tuyệt đối vào số dòng header.
    """

    found = {}

    # Chỉ tìm trong 10 dòng đầu.
    # Header TKB thông thường nằm ở đây.
    max_scan_row = min(ws.max_row, 15)

    for row in ws.iter_rows(
        min_row=1,
        max_row=max_scan_row
    ):
        for cell in row:

            value = clean_text(cell.value)

            if not is_class_name(value):
                continue

            cls = value.upper()
            col = cell.column

            # Nếu cùng một lớp xuất hiện nhiều lần,
            # giữ vị trí đầu tiên.
            if cls not in found:
                found[cls] = col

    return found


# ============================================================
# FIND DAY / PERIOD
# ============================================================

def find_period(value):
    """
    Nhận diện tiết:
        1
        2
        ...
        6
        Tiết 1
        Tiết 2
        ...
        Tiết 6

    Không coi "Thứ 2", "Thứ 3"... là tiết.
    """

    text = clean_text(value)

    if not text:
        return None

    # "Tiết 1", "Tiết 2", ...
    match = re.fullmatch(
        r"tiết\s*([1-6])",
        text,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    # Chỉ chấp nhận ô chứa đúng một số 1-6
    if text in {"1", "2", "3", "4", "5", "6"}:
        return int(text)

    return None

def detect_day_from_row(ws, row_idx, day_state):
    """
    Cột A là cột Ngày/Thứ.

    Excel có merged cell ở cột A:
        Thứ 2
        [trống]
        [trống]
        [trống]
        Thứ 3

    Vì vậy chỉ đọc CỘT A, không quét các cột khác.
    """

    value = ws.cell(
        row=row_idx,
        column=1
    ).value

    day = normalize_day(value)

    if day is not None:
        day_state["day"] = day

    return day_state.get("day")

# ============================================================
# MATRIX PARSER
# ============================================================

def parse_matrix_sheet(ws, buoi, teachers):
    """
    Đọc một sheet TKB_Lop_Sang / TKB_Lop_Chieu.

    Mỗi ô có dạng:
        MÔN - MãGV

    Ví dụ:
        Toán - QuyếtT

    Tên lớp được lấy từ header cột.
    Thứ được forward-fill vì Excel có merged cells.
    """

    print(
        f"\n--- Đọc sheet: {ws.title} ({buoi}) ---"
    )

    class_columns = find_class_columns(ws)

    if not class_columns:
        raise ValueError(
            f"Không tìm thấy tên lớp trong sheet {ws.title}."
        )

    print(
        f"✓ Tìm thấy {len(class_columns)} lớp"
    )

    slots = []

    invalid_teacher_codes = []
    malformed_cells = []

    day_state = {
        "day": None
    }

    # --------------------------------------------------------
    # Duyệt từng dòng
    # --------------------------------------------------------

    for row_idx in range(1, ws.max_row + 1):

        day = detect_day_from_row(
            ws,
            row_idx,
            day_state
        )

        if day is None:
            continue

        # Tìm tiết ở các cột đầu.
        # Cột B là cột Tiết.
        period = find_period(
            ws.cell(
                row=row_idx,
                column=2
            ).value
        )

        if period is None:
            continue

        # ----------------------------------------------------
        # Đọc từng lớp
        # ----------------------------------------------------

        for cls, col in class_columns.items():

            value = ws.cell(
                row=row_idx,
                column=col
            ).value

            value = clean_text(value)

            if not value:
                continue

            parsed = parse_subject_code(value)

            if parsed is None:
                malformed_cells.append(
                    (
                        ws.title,
                        row_idx,
                        col,
                        cls,
                        value,
                    )
                )
                continue

            subject, gv_code = parsed

            # ------------------------------------------------
            # Kiểm tra mã GV
            # ------------------------------------------------

            if gv_code not in teachers:

                invalid_teacher_codes.append(
                    {
                        "sheet": ws.title,
                        "row": row_idx,
                        "col": col,
                        "lop": cls,
                        "thu": day,
                        "tiet": period,
                        "subject": subject,
                        "gv_code": gv_code,
                    }
                )

            slots.append(
                {
                    "lop": cls,
                    "thu": day,
                    "tiet": period,
                    "mon": subject,
                    "gv_code": gv_code,
                    "buoi": buoi,
                }
            )

    print(
        f"✓ Đọc được {len(slots)} tiết có dữ liệu"
    )

    return (
        slots,
        invalid_teacher_codes,
        malformed_cells,
    )


# ============================================================
# SQLITE
# ============================================================

def create_database(slots, teachers):
    """
    Tạo database mới hoàn toàn.

    Không append vào DB cũ.
    """

    if DB_FILE.exists():
        DB_FILE.unlink()

    conn = sqlite3.connect(DB_FILE)

    cur = conn.cursor()

    # --------------------------------------------------------
    # Giáo viên
    # --------------------------------------------------------

    cur.execute("""
        CREATE TABLE giaovien (
            gv_code TEXT PRIMARY KEY,
            ho_ten TEXT NOT NULL,
            ngay_sinh TEXT,
            gioi_tinh TEXT,
            mon TEXT,
            chuc_vu TEXT
        )
    """)

    # --------------------------------------------------------
    # Lớp
    # --------------------------------------------------------

    cur.execute("""
        CREATE TABLE lop (
            lop TEXT PRIMARY KEY,
            khoi INTEGER,
            buoi TEXT
        )
    """)

    # --------------------------------------------------------
    # Thời khóa biểu
    # --------------------------------------------------------

    cur.execute("""
        CREATE TABLE slot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lop TEXT NOT NULL,
            thu INTEGER NOT NULL,
            tiet INTEGER NOT NULL,
            mon TEXT NOT NULL,
            gv_code TEXT NOT NULL,
            buoi TEXT NOT NULL,
            
            FOREIGN KEY (gv_code)
                REFERENCES giaovien(gv_code),

            FOREIGN KEY (lop)
                REFERENCES lop(lop)
        )
    """)

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    cur.execute("""
        CREATE INDEX idx_slot_lop_thu_tiet
        ON slot(lop, thu, tiet)
    """)

    cur.execute("""
        CREATE INDEX idx_slot_gv_thu_tiet
        ON slot(gv_code, thu, tiet)
    """)

    cur.execute("""
        CREATE INDEX idx_slot_lop
        ON slot(lop)
    """)

    cur.execute("""
        CREATE INDEX idx_slot_gv
        ON slot(gv_code)
    """)

    # --------------------------------------------------------
    # Insert teachers
    # --------------------------------------------------------

    for teacher in teachers.values():

        cur.execute(
            """
            INSERT INTO giaovien (
                gv_code,
                ho_ten,
                ngay_sinh,
                gioi_tinh,
                mon,
                chuc_vu
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                teacher["gv_code"],
                teacher["name"],
                teacher["dob"],
                teacher["gender"],
                teacher["subject"],
                teacher["role"],
            )
        )

    # --------------------------------------------------------
    # Insert classes
    # --------------------------------------------------------

    classes = {}

    for slot in slots:

        lop = slot["lop"]

        if lop not in classes:

            khoi = int(lop[:2])

            classes[lop] = {
                "lop": lop,
                "khoi": khoi,
                "buoi": slot["buoi"],
            }

    for cls in classes.values():

        cur.execute(
            """
            INSERT INTO lop (
                lop,
                khoi,
                buoi
            )
            VALUES (?, ?, ?)
            """,
            (
                cls["lop"],
                cls["khoi"],
                cls["buoi"],
            )
        )

    # --------------------------------------------------------
    # Insert slots
    # --------------------------------------------------------

    for slot in slots:

        cur.execute(
            """
            INSERT INTO slot (
                lop,
                thu,
                tiet,
                mon,
                gv_code,
                buoi
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                slot["lop"],
                slot["thu"],
                slot["tiet"],
                slot["mon"],
                slot["gv_code"],
                slot["buoi"],
            )
        )

    conn.commit()

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    teacher_count = cur.execute(
        "SELECT COUNT(*) FROM giaovien"
    ).fetchone()[0]

    class_count = cur.execute(
        "SELECT COUNT(*) FROM lop"
    ).fetchone()[0]

    slot_count = cur.execute(
        "SELECT COUNT(*) FROM slot"
    ).fetchone()[0]

    conn.close()

    return (
        teacher_count,
        class_count,
        slot_count,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BUILD THỜI KHÓA BIỂU → SQLITE")
    print("=" * 60)

    # --------------------------------------------------------
    # Kiểm tra file
    # --------------------------------------------------------

    if not EXCEL_FILE.exists():

        raise FileNotFoundError(
            f"\nKhông tìm thấy:\n{EXCEL_FILE}\n\n"
            "Hãy đặt file Excel TKB vào thư mục data/"
        )

    # --------------------------------------------------------
    # Giáo viên
    # --------------------------------------------------------

    teachers = load_teachers()

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    print(
        f"\n✓ Đang đọc: {EXCEL_FILE.name}"
    )

    wb = load_workbook(
        EXCEL_FILE,
        data_only=True,
        read_only=False,
    )

    print(
        "\nCác sheet:",
        ", ".join(wb.sheetnames)
    )

    all_slots = []

    all_invalid_codes = []

    all_malformed = []

    # --------------------------------------------------------
    # Parse 2 sheet TKB
    # --------------------------------------------------------

    for sheet_name, buoi in SHEETS.items():

        if sheet_name not in wb.sheetnames:

            raise ValueError(
                f"Không tìm thấy sheet: {sheet_name}"
            )

        ws = wb[sheet_name]

        slots, invalid, malformed = parse_matrix_sheet(
            ws,
            buoi,
            teachers
        )

        all_slots.extend(slots)
        all_invalid_codes.extend(invalid)
        all_malformed.extend(malformed)

    wb.close()

    # --------------------------------------------------------
    # Thống kê
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("KIỂM TRA DỮ LIỆU")
    print("=" * 60)

    print(
        f"Mã GV trong giao_vien.txt : {len(teachers)}"
    )

    print(
        f"Tổng số slot              : {len(all_slots)}"
    )

    class_set = {
        slot["lop"]
        for slot in all_slots
    }

    print(
        f"Số lớp                    : {len(class_set)}"
    )

    # --------------------------------------------------------
    # Invalid teacher code
    # --------------------------------------------------------

    if all_invalid_codes:

        print(
            "\n❌ MÃ GIÁO VIÊN KHÔNG TỒN TẠI:"
        )

        unique_codes = sorted(
            {
                x["gv_code"]
                for x in all_invalid_codes
            }
        )

        for code in unique_codes:

            examples = [
                x
                for x in all_invalid_codes
                if x["gv_code"] == code
            ]

            first = examples[0]

            print(
                f"  {code} "
                f"→ {first['lop']} "
                f"Thứ {first['thu']} "
                f"tiết {first['tiet']}"
            )

        print(
            "\n❌ DỪNG. Không tạo database."
        )

        return 1

    # --------------------------------------------------------
    # Malformed cells
    # --------------------------------------------------------

    if all_malformed:

        print(
            "\n⚠️ Các ô có dữ liệu nhưng không đọc được:"
        )

        for item in all_malformed[:30]:

            sheet, row, col, lop, value = item

            print(
                f"  {sheet}!{row}:{col} "
                f"{lop}: {value}"
            )

        if len(all_malformed) > 30:

            print(
                f"  ... và "
                f"{len(all_malformed) - 30} ô khác"
            )

        print(
            "\n⚠️ Những ô này đã được bỏ qua."
        )

    # --------------------------------------------------------
    # Build DB
    # --------------------------------------------------------

    print(
        "\n✓ Không phát hiện lỗi logic."
    )

    print(
        "\nĐang tạo database..."
    )

    teacher_count, class_count, slot_count = create_database(
        all_slots,
        teachers
    )

    print("\n" + "=" * 60)
    print("BUILD THÀNH CÔNG")
    print("=" * 60)

    print(
        f"Giáo viên : {teacher_count}"
    )

    print(
        f"Lớp       : {class_count}"
    )

    print(
        f"Tiết học  : {slot_count}"
    )

    print(
        f"Database  : {DB_FILE}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())