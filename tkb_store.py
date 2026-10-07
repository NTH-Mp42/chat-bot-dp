import sqlite3
from pathlib import Path
from typing import Optional


# ============================================================
# CẤU HÌNH
# ============================================================

DB_PATH = Path(__file__).resolve().parent / "data" / "tkb.db"


# ============================================================
# KẾT NỐI DATABASE
# ============================================================

def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy database TKB: {DB_PATH}"
        )

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def normalize_thu(thu):
    """
    Chuyển nhiều cách viết về số thứ:
        2, "2", "Thứ 2", "thứ hai" -> 2
        ...
        7, "7", "Thứ 7", "thứ bảy" -> 7
    """

    if thu is None:
        return None

    if isinstance(thu, int):
        if 2 <= thu <= 7:
            return thu
        return None

    s = str(thu).strip().lower()

    mapping = {
        "2": 2,
        "thứ 2": 2,
        "thu 2": 2,
        "thứ hai": 2,
        "thu hai": 2,

        "3": 3,
        "thứ 3": 3,
        "thu 3": 3,
        "thứ ba": 3,
        "thu ba": 3,

        "4": 4,
        "thứ 4": 4,
        "thu 4": 4,
        "thứ tư": 4,
        "thu tu": 4,

        "5": 5,
        "thứ 5": 5,
        "thu 5": 5,
        "thứ năm": 5,
        "thu nam": 5,

        "6": 6,
        "thứ 6": 6,
        "thu 6": 6,
        "thứ sáu": 6,
        "thu sau": 6,

        "7": 7,
        "thứ 7": 7,
        "thu 7": 7,
        "thứ bảy": 7,
        "thu bay": 7,
    }

    return mapping.get(s)


def normalize_lop(lop):
    """Chuẩn hóa tên lớp."""
    if lop is None:
        return None

    return str(lop).strip().upper()


def get_buoi_label(buoi):
    """Chuyển tên buổi thành cách hiển thị đẹp."""
    if buoi == "sang":
        return "Sáng"

    if buoi == "chieu":
        return "Chiều"

    return str(buoi)


# ============================================================
# THÔNG TIN CƠ BẢN
# ============================================================

def get_all_classes():
    """
    Trả về toàn bộ lớp.

    Kết quả:
    [
        {
            "lop": "10A1",
            "khoi": "10",
            "buoi": "chieu"
        },
        ...
    ]
    """

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT lop, khoi, buoi
            FROM lop
            ORDER BY lop
        """).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


def get_all_teachers():
    """
    Trả về toàn bộ giáo viên có mã GV.
    """

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT gv_code, ho_ten, mon, chuc_vu
            FROM giaovien
            ORDER BY ho_ten
        """).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


def get_teacher(gv_code):
    """
    Tìm giáo viên theo mã GV.
    """

    if not gv_code:
        return None

    conn = get_connection()

    try:
        row = conn.execute("""
            SELECT gv_code, ho_ten, mon, chuc_vu
            FROM giaovien
            WHERE gv_code = ?
        """, (gv_code.strip(),)).fetchone()

        return dict(row) if row else None

    finally:
        conn.close()


# ============================================================
# TKB CỦA LỚP
# ============================================================

def lich_lop_theo_ngay(lop, thu):
    """
    Lấy thời khóa biểu của một lớp trong một ngày.

    Ví dụ:
        lich_lop_theo_ngay("11A1", 2)

    Trả về:
        [
            {
                "tiet": 1,
                "mon": "...",
                "gv_code": "...",
                "ho_ten": "...",
                "buoi": "sang"
            },
            ...
        ]
    """

    lop = normalize_lop(lop)
    thu = normalize_thu(thu)

    if not lop or thu is None:
        return []

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT
                s.tiet,
                s.mon,
                s.gv_code,
                s.buoi,
                g.ho_ten,
                g.chuc_vu
            FROM slot s
            LEFT JOIN giaovien g
                ON s.gv_code = g.gv_code
            WHERE s.lop = ?
              AND s.thu = ?
            ORDER BY s.tiet
        """, (lop, thu)).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


def lich_lop_ca_tuan(lop):
    """
    Lấy toàn bộ TKB trong tuần của một lớp.

    Kết quả được sắp xếp:
        Thứ -> Tiết
    """

    lop = normalize_lop(lop)

    if not lop:
        return []

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT
                s.thu,
                s.tiet,
                s.mon,
                s.gv_code,
                s.buoi,
                g.ho_ten,
                g.chuc_vu
            FROM slot s
            LEFT JOIN giaovien g
                ON s.gv_code = g.gv_code
            WHERE s.lop = ?
            ORDER BY s.thu, s.tiet
        """, (lop,)).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# ============================================================
# TÌM AI DẠY LỚP
# ============================================================

def ai_day(lop, thu, tiet):
    """
    Tìm giáo viên + môn dạy một lớp ở một tiết cụ thể.

    Ví dụ:
        ai_day("11A1", 2, 2)

    Trả về:
        {
            "lop": "11A1",
            "thu": 2,
            "tiet": 2,
            "mon": "SINH",
            "gv_code": "NguyệtS",
            "ho_ten": "Doãn Thị Minh Nguyệt",
            ...
        }

    Nếu không có tiết -> None
    """

    lop = normalize_lop(lop)
    thu = normalize_thu(thu)

    if not lop or thu is None or tiet is None:
        return None

    try:
        tiet = int(tiet)
    except (TypeError, ValueError):
        return None

    conn = get_connection()

    try:
        row = conn.execute("""
            SELECT
                s.lop,
                s.thu,
                s.tiet,
                s.mon,
                s.gv_code,
                s.buoi,
                g.ho_ten,
                g.chuc_vu
            FROM slot s
            LEFT JOIN giaovien g
                ON s.gv_code = g.gv_code
            WHERE s.lop = ?
              AND s.thu = ?
              AND s.tiet = ?
            LIMIT 1
        """, (lop, thu, tiet)).fetchone()

        return dict(row) if row else None

    finally:
        conn.close()


# ============================================================
# TKB CỦA GIÁO VIÊN
# ============================================================

def lich_giao_vien(gv_code):
    """
    Lấy toàn bộ lịch dạy của giáo viên theo mã GV.
    """

    if not gv_code:
        return []

    gv_code = gv_code.strip()

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT
                s.lop,
                s.thu,
                s.tiet,
                s.mon,
                s.gv_code,
                s.buoi,
                g.ho_ten,
                g.mon AS mon_gv,
                g.chuc_vu
            FROM slot s
            LEFT JOIN giaovien g
                ON s.gv_code = g.gv_code
            WHERE s.gv_code = ?
            ORDER BY s.thu, s.tiet, s.lop
        """, (gv_code,)).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# ============================================================
# TÌM GIÁO VIÊN THEO TÊN
# ============================================================

def tim_giao_vien_theo_ten(ho_ten):
    """
    Tìm giáo viên theo tên.

    Có thể trả về nhiều người nếu trùng tên.
    Đây là chủ ý để chatbot không đoán nhầm.
    """

    if not ho_ten:
        return []

    ho_ten = ho_ten.strip()

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT
                gv_code,
                ho_ten,
                mon,
                chuc_vu
            FROM giaovien
            WHERE ho_ten LIKE ?
            ORDER BY ho_ten, gv_code
        """, (f"%{ho_ten}%",)).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# ============================================================
# TÌM GIÁO VIÊN DẠY MỘT LỚP
# ============================================================

def giao_vien_cua_lop(lop):
    """
    Lấy danh sách giáo viên xuất hiện trong TKB của một lớp.
    """

    lop = normalize_lop(lop)

    if not lop:
        return []

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT DISTINCT
                s.gv_code,
                g.ho_ten,
                g.mon,
                g.chuc_vu
            FROM slot s
            LEFT JOIN giaovien g
                ON s.gv_code = g.gv_code
            WHERE s.lop = ?
            ORDER BY g.ho_ten
        """, (lop,)).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# ============================================================
# ĐẾM SỐ TIẾT
# ============================================================

def dem_tiet(lop=None, mon=None, gv_code=None):
    """
    Đếm số tiết.

    Có thể dùng:
        dem_tiet(lop="11A1")
        dem_tiet(lop="11A1", mon="TOAN")
        dem_tiet(gv_code="QuyếtT")
    """

    conditions = []
    params = []

    if lop:
        conditions.append("s.lop = ?")
        params.append(normalize_lop(lop))

    if mon:
        conditions.append("s.mon = ?")
        params.append(mon.strip())

    if gv_code:
        conditions.append("s.gv_code = ?")
        params.append(gv_code.strip())

    sql = """
        SELECT COUNT(*) AS so_tiet
        FROM slot s
    """

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    conn = get_connection()

    try:
        row = conn.execute(sql, params).fetchone()
        return row["so_tiet"]

    finally:
        conn.close()


# ============================================================
# TÌM CÁC TIẾT TRỐNG CỦA LỚP
# ============================================================

def tiet_trong_lop(lop, thu=None):
    """
    Tìm các tiết chưa có dữ liệu của một lớp.

    Chỉ xét tiết 1 -> 6.
    """

    lop = normalize_lop(lop)

    if not lop:
        return []

    if thu is not None:
        thu = normalize_thu(thu)

        if thu is None:
            return []

        days = [thu]
    else:
        days = range(2, 8)

    conn = get_connection()

    try:
        result = []

        for day in days:
            rows = conn.execute("""
                SELECT tiet
                FROM slot
                WHERE lop = ?
                  AND thu = ?
            """, (lop, day)).fetchall()

            used = {row["tiet"] for row in rows}

            for tiet in range(1, 7):
                if tiet not in used:
                    result.append({
                        "thu": day,
                        "tiet": tiet
                    })

        return result

    finally:
        conn.close()


# ============================================================
# THỐNG KÊ DATABASE
# ============================================================

def thong_ke():
    """
    Kiểm tra nhanh database.
    """

    conn = get_connection()

    try:
        teacher_count = conn.execute("""
            SELECT COUNT(*) AS n
            FROM giaovien
        """).fetchone()["n"]

        class_count = conn.execute("""
            SELECT COUNT(*) AS n
            FROM lop
        """).fetchone()["n"]

        slot_count = conn.execute("""
            SELECT COUNT(*) AS n
            FROM slot
        """).fetchone()["n"]

        return {
            "teachers": teacher_count,
            "classes": class_count,
            "slots": slot_count
        }

    finally:
        conn.close()


# ============================================================
# FORMAT KẾT QUẢ
# ============================================================

def format_lich_lop(rows, lop=None, thu=None):
    """
    Chuyển kết quả truy vấn thành text dễ đưa cho LLM.
    """

    if not rows:
        if lop and thu:
            return f"Không có tiết học nào của lớp {lop} vào Thứ {thu}."

        if lop:
            return f"Không tìm thấy thời khóa biểu của lớp {lop}."

        return "Không có dữ liệu thời khóa biểu."

    lines = []

    if lop and thu:
        lines.append(f"Thời khóa biểu lớp {lop} - Thứ {thu}:")

    for row in rows:
        tiet = row.get("tiet", "?")
        mon = row.get("mon") or "Không rõ môn"
        gv = row.get("ho_ten") or row.get("gv_code") or "Chưa xác định"
        buoi = get_buoi_label(row.get("buoi"))

        lines.append(
            f"- Tiết {tiet}: {mon} - {gv} ({buoi})"
        )

    return "\n".join(lines)


def format_lich_giao_vien(rows, gv_code=None):
    """
    Format lịch giáo viên thành text.
    """

    if not rows:
        if gv_code:
            return f"Không tìm thấy lịch dạy của giáo viên {gv_code}."

        return "Không có dữ liệu."

    teacher_name = rows[0].get("ho_ten") or gv_code

    lines = [
        f"Lịch dạy của {teacher_name}:"
    ]

    for row in rows:
        lines.append(
            f"- Thứ {row['thu']}, tiết {row['tiet']}: "
            f"{row['mon']} - lớp {row['lop']} "
            f"({get_buoi_label(row.get('buoi'))})"
        )

    return "\n".join(lines)


# ============================================================
# TEST TRỰC TIẾP
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST TKB STORE")
    print("=" * 60)

    # 1. Thống kê database
    print("\n[1] Thống kê database:")
    print(thong_ke())

    # 2. TKB 11A1 Thứ 2
    print("\n[2] TKB 11A1 - Thứ 2:")

    rows = lich_lop_theo_ngay("11A1", 2)

    print(format_lich_lop(
        rows,
        lop="11A1",
        thu=2
    ))

    # 3. Ai dạy 11A1 Thứ 2 tiết 2?
    print("\n[3] Ai dạy 11A1 - Thứ 2 - tiết 2?")

    result = ai_day("11A1", 2, 2)

    if result:
        print(
            f"Môn: {result['mon']}\n"
            f"GV: {result['ho_ten']}\n"
            f"Mã GV: {result['gv_code']}"
        )
    else:
        print("Không có dữ liệu.")

    # 4. Lịch giáo viên QuyếtT
    print("\n[4] Lịch giáo viên QuyếtT:")

    rows = lich_giao_vien("QuyếtT")

    print(format_lich_giao_vien(
        rows,
        gv_code="QuyếtT"
    ))

    # 5. Tìm giáo viên theo tên
    print("\n[5] Tìm giáo viên Nguyễn Thị Vân:")

    teachers = tim_giao_vien_theo_ten("Nguyễn Thị Vân")

    for teacher in teachers:
        print(
            f"- {teacher['ho_ten']} | "
            f"{teacher['mon']} | "
            f"{teacher['gv_code']}"
        )

    print("\n" + "=" * 60)
    print("TEST HOÀN TẤT")
    print("=" * 60)

def tim_giao_vien_theo_ten_va_mon(ten, mon):
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM giaovien
        WHERE lower(ho_ten) LIKE ?
          AND lower(mon) LIKE ?
        ORDER BY ho_ten
        """,
        (
            f"%{ten.lower()}%",
            f"%{mon.lower()}%",
        )
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]