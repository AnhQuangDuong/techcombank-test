# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pymupdf>=1.24.0",
#     "google-genai>=1.0.0",
#     "python-dotenv>=1.0.0",
#     "pillow>=10.0.0",
#     "pydantic>=2.0.0",
#     "beautifulsoup4>=4.12.0",
# ]
# ///
"""
Script đọc toàn bộ từng trang PDF thực tế (không cắt đôi),
sử dụng Gemini OCR trích xuất danh sách các khối nội dung:
[{"location": "left" | "right" | "all", "content": "..."}]
Sau đó ánh xạ (map) nội dung vào đúng các trang logic:
- location == 'left'  -> gán vào trang logic bên trái
- location == 'right' -> gán vào trang logic bên phải
- location == 'all'   -> gán vào CẢ trang logic bên trái và bên phải (hoặc trang bìa).
"""

import argparse
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

try:
    import pymupdf
except ImportError:
    print(
        f"Lỗi: Thư viện 'pymupdf' chưa được cài đặt.\nChạy trực tiếp qua uv:\n    uv run {sys.argv[0]}",
        file=sys.stderr,
    )
    sys.exit(1)

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Tải biến môi trường từ .env
load_dotenv()

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    print(
        f"Lỗi: Thư viện 'google-genai' chưa được cài đặt.\nChạy trực tiếp qua uv:\n    uv run {sys.argv[0]}",
        file=sys.stderr,
    )
    sys.exit(1)

# Import prompts từ file prompts.py
try:
    from prompts import (
        SYSTEM_PROMPT,
        TABLE_INSTRUCTION_PROMPT,
        TOC_SYSTEM_PROMPT,
        TOC_USER_PROMPT,
        USER_PROMPT,
        TOCExtractionOutput,
    )
except ImportError:
    print(
        "Lỗi: Không tìm thấy file 'prompts.py' trong cùng thư mục.\n"
        "Vui lòng đảm bảo file prompts.py chứa đầy đủ các prompts và schema.",
        file=sys.stderr,
    )
    sys.exit(1)


from layout_tracker import LayoutTracker


class ContentBlock(BaseModel):
    location: Literal["left", "right", "all"] = Field(
        description="Vị trí của nội dung: 'left' (nửa trái), 'right' (nửa phải), hoặc 'all' (toàn bộ trang / trải dài cả hai bên)"
    )
    content: str = Field(
        description="Nội dung Markdown (# cho chủ đề chính, ##, ### cho phân cấp), bảng HTML <table>...</table>, biểu đồ trích xuất đầy đủ điểm dữ liệu, hoặc chuỗi rỗng nếu không có nội dung."
    )


class PageParseOutput(BaseModel):
    blocks: List[ContentBlock] = Field(
        default_factory=list, description="Danh sách các khối nội dung được trích xuất từ trang"
    )


def get_gemini_config() -> tuple[str, str]:
    """
    Lấy bắt buộc cấu hình từ file .env:
    - GEMINI_API_KEY
    - GEMINI_MODEL
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        print(
            "Lỗi: Chưa thiết lập GEMINI_API_KEY trong file .env.\n"
            "Vui lòng cấu hình trong file .env, ví dụ:\n"
            "    GEMINI_API_KEY=AIzaSy...",
            file=sys.stderr,
        )
        sys.exit(1)

    model = os.getenv("GEMINI_MODEL")
    if not model or not model.strip():
        print(
            "Lỗi: Chưa thiết lập GEMINI_MODEL trong file .env.\n"
            "Vui lòng cấu hình tên model trong file .env, ví dụ:\n"
            "    GEMINI_MODEL=gemini-2.5-flash",
            file=sys.stderr,
        )
        sys.exit(1)

    return api_key.strip(), model.strip()


def init_gemini_client(api_key: str) -> genai.Client:
    """Khởi tạo Gemini client với API key được truyền từ .env."""
    return genai.Client(api_key=api_key)


def check_page_has_meaningful_content(page: pymupdf.Page, header_height: float = 60.0) -> bool:
    """
    Kiểm tra trang có nội dung thực sự hay không, LOẠI TRỪ phần header ở mép trên
    (nơi chứa số trang và tên báo cáo/chương cố định ở góc trái và góc phải).
    """
    w, h = page.rect.width, page.rect.height
    body_rect = pymupdf.Rect(0, header_height, w, h)

    # 1. Kiểm tra văn bản trong phần thân trang
    body_text = page.get_text(clip=body_rect).strip()
    if len(body_text) > 0:
        return True

    # 2. Kiểm tra hình ảnh giao cắt với phần thân trang
    for img in page.get_image_info():
        img_rect = pymupdf.Rect(img["bbox"])
        if img_rect.intersects(body_rect) and img_rect.height > 20 and img_rect.width > 20:
            return True

    # 3. Kiểm tra nét vẽ vector (bảng, biểu đồ) trong phần thân trang
    for d in page.get_drawings():
        dr = d["rect"]
        if dr.intersects(body_rect) and (dr.height > 10 or dr.width > 50):
            return True

    return False


def build_parsing_plan(
    doc: pymupdf.Document,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    header_height: float = 60.0,
) -> list[dict]:
    """
    Luồng tiền xử lý (Preprocessing) trước khi gọi LLM:
    - Trích xuất số trang logic bên trái và bên phải từ header mép trên.
    - Trích xuất và duy trì tên chương (current_chapter).
    - Kiểm tra nội dung trang (loại trừ header cố định).
    """
    if end_idx is None:
        end_idx = len(doc)

    plans = []
    current_chapter = "Báo cáo thường niên"

    for idx in range(min(len(doc), end_idx)):
        page = doc[idx]
        pdf_page_num = idx + 1
        w, _ = page.rect.width, page.rect.height
        mid_x = w / 2

        if idx == 0:
            current_chapter = "Bìa tài liệu"
            plan = {
                "pdf_page": 1,
                "is_cover": True,
                "left_logical_page": 1,
                "right_logical_page": None,
                "chapter": "Bìa tài liệu",
                "has_content": True,
            }
            if idx >= start_idx:
                plans.append(plan)
            continue

        # Header góc trên bên trái (left logical page number + title doc)
        top_left = pymupdf.Rect(0, 0, mid_x, header_height)
        tl_text = page.get_text(clip=top_left).strip()

        # Header góc trên bên phải (right logical page number + chapter name)
        top_right = pymupdf.Rect(mid_x, 0, w, header_height)
        tr_text = page.get_text(clip=top_right).strip()

        fallback_left = idx * 2
        fallback_right = idx * 2 + 1

        # Trích xuất số trang logic bên trái
        left_num = fallback_left
        for line in tl_text.split("\n"):
            line = line.strip()
            if line.isdigit():
                left_num = int(line)
                break

        # Trích xuất số trang logic bên phải và tên chương
        right_num = fallback_right
        detected_chapter = None
        for line in tr_text.split("\n"):
            line = line.strip()
            if line.isdigit():
                right_num = int(line)
            elif len(line) > 2:
                detected_chapter = line

        if detected_chapter:
            current_chapter = detected_chapter

        # Kiểm tra trang có nội dung thực sự hay không (loại trừ header)
        has_content = check_page_has_meaningful_content(page, header_height=header_height)

        plan = {
            "pdf_page": pdf_page_num,
            "is_cover": False,
            "left_logical_page": left_num,
            "right_logical_page": right_num,
            "chapter": current_chapter,
            "has_content": has_content,
        }

        if idx >= start_idx:
            plans.append(plan)

    return plans


def locate_table_bbox_from_gemini(
    page: pymupdf.Page,
    gem_table_soup,
    location: str = "all",
) -> pymupdf.Rect | None:
    """
    Xác định Bounding Box chính xác của toàn bộ bảng trên trang PDF dựa trên:
    - Text dòng đầu (header)
    - Text các dòng dữ liệu ở giữa (middle data rows) để phân biệt các bảng có header/footer trùng nhau
    - Text dòng cuối (footer/data row cuối)
    Gộp tọa độ bằng cách: phần đầu, các dòng giữa và phần cuối phải thẳng hàng (cùng trục X),
    với y0 < y_mid < y1, dôi ra vừa chạm viền ngoài.
    """
    w = page.rect.width
    h = page.rect.height
    scope = pymupdf.Rect(0, 60.0, w, h)
    if location == "left":
        scope = pymupdf.Rect(0, 60.0, w / 2 + 5, h)
    elif location == "right":
        scope = pymupdf.Rect(w / 2 - 5, 60.0, w, h)

    rows = gem_table_soup.find_all("tr")
    if not rows:
        return None

    def extract_cells_text(tr):
        return [c.get_text().strip() for c in tr.find_all(["th", "td"]) if len(c.get_text().strip()) >= 3]

    first_texts = extract_cells_text(rows[0])

    last_texts = []
    for r in reversed(rows):
        cells = extract_cells_text(r)
        if cells:
            last_texts = cells
            break

    if not first_texts or not last_texts:
        return None

    # Lấy thêm vài nội dung ở giữa bảng để phân biệt nếu đầu hoặc cuối bảng trùng nhau
    mid_rows_texts = []
    n_rows = len(rows)
    if n_rows > 2:
        step_indices = sorted(list(set([n_rows // 4, n_rows // 2, (3 * n_rows) // 4])))
        for idx in step_indices:
            if 0 < idx < n_rows - 1:
                cells = extract_cells_text(rows[idx])
                if cells:
                    mid_rows_texts.append(cells)

    def find_matches(texts):
        matches = []
        for txt in texts:
            query = txt if len(txt) <= 40 else txt[:35]
            res = page.search_for(query, clip=scope)
            if res:
                matches.extend(res)
        return matches

    first_matches = find_matches(first_texts)
    last_matches = find_matches(last_texts)
    mid_matches = [find_matches(m_cells) for m_cells in mid_rows_texts]
    mid_matches = [m for m in mid_matches if m]

    # Tìm nhóm (first_rect, mid_rects, last_rect) thỏa mãn:
    # 1. Thẳng hàng trục X (cùng cột / chiều rộng bảng)
    # 2. y0 < y_mid_1 <= y_mid_2 <= ... < y1
    valid_groups = []
    for l_rect in last_matches:
        for f_rect in first_matches:
            if f_rect.y1 >= l_rect.y0:
                continue

            max_h = (n_rows + 5) * 60.0
            if (l_rect.y1 - f_rect.y0) > max_h:
                continue

            cur_y = f_rect.y0
            matched_mids = []
            valid_chain = True
            for m_group in mid_matches:
                cands = [
                    m
                    for m in m_group
                    if m.y0 >= cur_y - 2.0 and m.y1 <= l_rect.y1 + 2.0 and abs(m.x0 - f_rect.x0) < (w / 2)
                ]
                if cands:
                    best_cand = min(cands, key=lambda m: m.y0)
                    matched_mids.append(best_cand)
                    cur_y = best_cand.y0
                else:
                    valid_chain = False
                    break

            if mid_matches and not valid_chain:
                continue

            all_pts = [f_rect, l_rect] + matched_mids
            xs = [r.x0 for r in all_pts] + [r.x1 for r in all_pts]
            if max(xs) - min(xs) > (w / 2 + 50):
                continue

            score = len(matched_mids) * 1000 - abs(l_rect.y1 - f_rect.y0 - n_rows * 20)
            valid_groups.append((score, f_rect, matched_mids, l_rect))

    if valid_groups:
        valid_groups.sort(key=lambda x: x[0], reverse=True)
        _, best_f, best_mids, best_l = valid_groups[0]

        all_rects = [best_f, best_l] + best_mids
        y0_text = min(r.y0 for r in all_rects)
        y1_text = max(r.y1 for r in all_rects)
        x0_text = min(r.x0 for r in all_rects)
        x1_text = max(r.x1 for r in all_rects)

        # Quét các đường nét vẽ vector để khóa chặt viền bảng sát sao (+- 1.5 pt)
        box_lines = []
        for d in page.get_drawings():
            r = d["rect"]
            if r.intersects(scope) and (y0_text - 15 <= r.y1 and r.y0 <= y1_text + 15):
                if abs(r.x0 - x0_text) < (w / 2) or abs(r.x1 - x1_text) < (w / 2):
                    box_lines.append(r)

        if box_lines:
            x0 = min(r.x0 for r in box_lines)
            x1 = max(r.x1 for r in box_lines)
            y0_cands = [r.y0 for r in box_lines if abs(r.y0 - y0_text) <= 20]
            y1_cands = [r.y1 for r in box_lines if abs(r.y1 - y1_text) <= 20]
            y0 = min(y0_cands) - 1.5 if y0_cands else y0_text - 3.0
            y1 = max(y1_cands) + 1.5 if y1_cands else y1_text + 3.0
        else:
            x0 = max(scope.x0, x0_text - 10)
            x1 = min(scope.x1, x1_text + 10)
            y0 = y0_text - 3.0
            y1 = y1_text + 3.0

        return pymupdf.Rect(max(scope.x0, x0), max(scope.y0, y0), min(scope.x1, x1), min(scope.y1, y1))

    # Fallback dự phòng nếu text OCR có khác biệt nhỏ không tìm thấy bằng search_for
    raw_tabs = [t for t in page.find_tables().tables if scope.intersects(pymupdf.Rect(t.bbox))]
    if not raw_tabs:
        return None

    bot_tab = None
    for t in reversed(raw_tabs):
        df_text = " ".join(" ".join(str(c).lower() for c in r if c) for r in t.extract())
        if any(bt.lower() in df_text for bt in last_texts):
            bot_tab = t
            break

    if bot_tab is None:
        return None

    bot_df_text = " ".join(" ".join(str(c).lower() for c in r if c) for r in bot_tab.extract())
    if any(tt.lower() in bot_df_text for tt in first_texts):
        top_tab = bot_tab
    else:
        top_tab = None
        cand_tops = [
            t for t in raw_tabs if t.bbox[3] <= bot_tab.bbox[1] + 35.0 and abs(t.bbox[0] - bot_tab.bbox[0]) <= 5.0
        ]
        for t in reversed(cand_tops):
            df_text = " ".join(" ".join(str(c).lower() for c in r if c) for r in t.extract())
            if any(tt.lower() in df_text for tt in first_texts):
                top_tab = t
                break

    if top_tab is None:
        top_tab = bot_tab

    x0 = min(top_tab.bbox[0], bot_tab.bbox[0]) - 2.0
    y0 = top_tab.bbox[1] - 2.0
    x1 = max(top_tab.bbox[2], bot_tab.bbox[2]) + 2.0
    y1 = bot_tab.bbox[3] + 2.0
    return pymupdf.Rect(max(scope.x0, x0), y0, min(scope.x1, x1), y1)


def extract_table_from_bbox(page: pymupdf.Page, table_rect: pymupdf.Rect):
    """Trích xuất bảng PyMuPDF trọn vẹn bên trong table_rect với join_tolerance mở rộng."""
    try:
        tabs = page.find_tables(clip=table_rect, snap_tolerance=5, join_tolerance=25).tables
        if not tabs:
            return None
        return max(tabs, key=lambda t: (t.row_count * t.col_count))
    except Exception:
        return None


def compute_table_topology(tab, header_rows_hint: int | None = None) -> dict:
    """
    Trích xuất ma trận (rowspan, colspan) và xác định các ô bị đè (covered).
    Đồng thời xác định số dòng Header k đảm bảo Topology Span Closure.
    """
    R = tab.row_count
    C = tab.col_count

    spans = {}
    covered = {}

    for r in range(R):
        for c in range(C):
            covered[(r, c)] = False

    for r in range(R):
        row = tab.rows[r]
        for c in range(C):
            cell = row.cells[c]
            if cell is None or covered[(r, c)]:
                covered[(r, c)] = True
                spans[(r, c)] = (0, 0)
                continue

            x0, y0, x1, y1 = cell

            # 1. Tính colspan
            colspan = 1
            for next_c in range(c + 1, C):
                if tab.rows[r].cells[next_c] is None:
                    col_in_span = False
                    for other_r in range(R):
                        other_cell = tab.rows[other_r].cells[next_c]
                        if other_cell is not None:
                            if other_cell[0] >= x0 - 1.5 and other_cell[2] <= x1 + 1.5:
                                col_in_span = True
                                break
                    if col_in_span:
                        colspan += 1
                    else:
                        break
                else:
                    break

            # 2. Tính rowspan
            rowspan = 1
            for next_r in range(r + 1, R):
                if tab.rows[next_r].cells[c] is not None:
                    break
                row_bottom = tab.rows[next_r].bbox[3]
                if row_bottom <= y1 + 1.5:
                    rowspan += 1
                else:
                    break

            spans[(r, c)] = (rowspan, colspan)

            for dr in range(rowspan):
                for dc in range(colspan):
                    if dr == 0 and dc == 0:
                        continue
                    covered[(r + dr, c + dc)] = True

    # Xác định số dòng header k
    k = 1
    if header_rows_hint and header_rows_hint > 0:
        k = header_rows_hint
    elif hasattr(tab, "header") and tab.header and tab.header.bbox:
        header_y1 = tab.header.bbox[3]
        count = 0
        for i, row in enumerate(tab.rows):
            if row.bbox and row.bbox[3] <= header_y1 + 2.0:
                count = i + 1
            else:
                break
        k = max(1, count)

    # Đảm bảo không cắt ngang ô có rowspan ở header
    topo_k = k
    for r in range(min(k, R)):
        for c in range(C):
            rs, _ = spans.get((r, c), (1, 1))
            if rs > 1 and r + rs > topo_k:
                topo_k = r + rs
    k = min(topo_k, R)

    return {
        "row_count": R,
        "col_count": C,
        "header_rows": k,
        "spans": spans,
        "covered": covered,
    }


def get_table_tokens(obj) -> set[str]:
    """Trích xuất tập hợp các từ và số đặc trưng để tính độ trùng lặp nội dung."""
    if hasattr(obj, "get_text"):
        text = obj.get_text(separator=" ")
    elif isinstance(obj, (list, tuple)):
        text = " ".join(" ".join(str(c) for c in r if c) for r in obj)
    else:
        text = str(obj)
    return set(w.lower() for w in re.findall(r"[\w\.,]+", text) if len(w) >= 2)


def flatten_table_headers(table, separator: str = " - "):
    """
    Làm phẳng các dòng header đa cấp (multi-level thead) thành duy nhất 1 dòng tr.
    Nếu header có k >= 2 dòng với rowspan/colspan, hàm này ánh xạ phân cấp cha -> con
    và ghép thành 1 chuỗi tên cột đầy đủ (ví dụ: 'Năm 2024 - Hợp nhất').
    """
    thead = table.find("thead")
    if not thead:
        return table

    header_rows = thead.find_all("tr")
    if len(header_rows) <= 1:
        return table

    R_head = len(header_rows)
    grid = {}  # (r, c) -> text

    for r, row in enumerate(header_rows):
        c = 0
        for cell in row.find_all(["th", "td"]):
            while (r, c) in grid:
                c += 1
            try:
                rowspan = int(cell.get("rowspan", 1) or 1)
            except (ValueError, TypeError):
                rowspan = 1
            try:
                colspan = int(cell.get("colspan", 1) or 1)
            except (ValueError, TypeError):
                colspan = 1

            text = cell.get_text(separator=" ", strip=True).replace("\n", " ")
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = text
            c += colspan

    if not grid:
        return table

    C = max(col for (_, col) in grid.keys()) + 1

    soup = BeautifulSoup("", "html.parser")
    new_tr = soup.new_tag("tr")

    for col_idx in range(C):
        level_texts = []
        for r in range(R_head):
            t = grid.get((r, col_idx), "").strip()
            if t and (not level_texts or level_texts[-1] != t):
                level_texts.append(t)
        col_name = separator.join(level_texts) if level_texts else ""
        th = soup.new_tag("th")
        th.string = col_name
        new_tr.append(th)

    thead.clear()
    thead.append(new_tr)
    return table


def render_pymupdf_table_to_html(
    tab: pymupdf.table.Table,
    flatten_headers: bool = True,
    header_separator: str = " - ",
    header_rows_hint: int = 1,
) -> str:
    """
    Render 1 đối tượng Table của PyMuPDF thành chuỗi HTML <table>...</table>
    bảo toàn cấu trúc ô (rowspan, colspan) và làm phẳng header nếu flatten_headers=True.
    """
    topo = compute_table_topology(tab, header_rows_hint=header_rows_hint)
    extract = tab.extract()
    k = topo["header_rows"]

    soup = BeautifulSoup("", "html.parser")
    table = soup.new_tag("table")
    thead = soup.new_tag("thead")
    tbody = soup.new_tag("tbody")

    for r in range(topo["row_count"]):
        tr_tag = soup.new_tag("tr")
        is_header = r < k
        for c in range(topo["col_count"]):
            if topo["covered"][(r, c)]:
                continue

            rowspan, colspan = topo["spans"].get((r, c), (1, 1))
            tag_name = "th" if is_header else "td"
            cell_tag = soup.new_tag(tag_name)

            if rowspan > 1:
                cell_tag["rowspan"] = str(rowspan)
            if colspan > 1:
                cell_tag["colspan"] = str(colspan)

            cell_text = ""
            if r < len(extract) and c < len(extract[r]):
                cell_text = (extract[r][c] or "").strip().replace("\n", " ")

            cell_tag.string = cell_text
            tr_tag.append(cell_tag)

        if is_header:
            thead.append(tr_tag)
        else:
            tbody.append(tr_tag)

    if len(thead.find_all("tr")) > 0:
        table.append(thead)
    if len(tbody.find_all("tr")) > 0:
        table.append(tbody)

    if flatten_headers:
        flatten_table_headers(table, separator=header_separator)

    return str(table)


def prompt_unassigned_action(
    pdf_page_num: Optional[int],
    location: str,
    total_raw_tabs: int,
    unassigned_count: int,
    action: Optional[str] = None,
) -> str:
    """
    Hiển thị cảnh báo nguy cơ parse sai bảng và lấy lựa chọn xử lý từ người dùng hoặc CLI:
    [1] Để nguyên như hiện tại (kết quả enhance thông thường)
    [2] Không sử dụng PyMuPDF để enhance kết quả bảng ở page này nữa, sử dụng lại Gemini để parse lại cho bảng này
    [3] Sử dụng kết quả các bảng từ PyMuPDF để thay vào table cho page này
    """
    banner = (
        f"\n{'=' * 75}\n"
        f"[!] CẢNH BÁO NGUY CƠ PARSE SAI BẢNG DO CÓ BẢNG CHƯA ĐƯỢC GÁN (UNASSIGNED):\n"
        f"    - PDF Trang: {pdf_page_num if pdf_page_num is not None else 'N/A'} (Vị trí: '{location}')\n"
        f"    - Tổng số bảng PyMuPDF phát hiện: {total_raw_tabs}\n"
        f"    - Số bảng chưa được gán (unassigned): {unassigned_count}\n"
        f"    Yêu cầu: Đã lưu kết quả parse hiện tại vào file 'temp_page.md'. Vui lòng mở file xem và chọn:\n"
        f"    Lựa chọn xử lý:\n"
        f"      [1] Để nguyên như hiện tại\n"
        f"      [2] Không sử dụng PyMuPDF để enhance kết quả bảng ở page này nữa, sử dụng lại Gemini để parse lại cho bảng này\n"
        f"      [3] Sử dụng kết quả các bảng từ PyMuPDF để thay vào table cho page này\n"
    )

    if action in ("1", "2", "3"):
        print(banner + f"    => Tự động chọn [{action}] theo cấu hình CLI / tham số.\n{'=' * 75}\n")
        return action

    # Nếu đang chạy trong terminal tương tác (interactive)
    if sys.stdin.isatty():
        print(banner + f"{'=' * 75}")
        choice = input("Nhập lựa chọn của bạn (1/2/3) [mặc định: 1]: ").strip()
        if choice not in ("1", "2", "3"):
            print("Lựa chọn không hợp lệ, mặc định chọn [1].")
            return "1"
        return choice
    else:
        # Chế độ non-interactive (CI, test, cron)
        print(banner + f"    => Môi trường non-interactive: Mặc định chọn [1].\n{'=' * 75}\n")
        return "1"


def enhance_table_with_pymupdf_headers(
    content: str,
    page: pymupdf.Page,
    location: str = "all",
    flatten_headers: bool = True,
    header_separator: str = " - ",
    pdf_page_num: Optional[int] = None,
    unassigned_action: Optional[str] = None,
) -> str:
    """
    Tự động chuẩn hóa và thay thế bảng Gemini bằng bảng chuẩn xác 100% từ PyMuPDF:
    - Với mỗi raw_tab từ PyMuPDF: gán cho bảng Gemini có độ tương đồng token cao nhất (>= 60%).
    - Đồng thời gom các phần bảng collinear liền kề (giải quyết case 1 bảng bị PyMuPDF tách đôi do subheader/dòng không viền).
    - Tái tạo bảng HTML trực tiếp từ 100% dữ liệu PyMuPDF (giữ nguyên độ chuẩn xác của số liệu, STT, tỷ lệ %, subheader và rowspan/colspan).
    - Làm phẳng các dòng header đa cấp nếu flatten_headers=True.
    - Phát hiện các bảng unassigned và đưa ra 3 lựa chọn xử lý cho người dùng.
    """
    try:
        w, h = page.rect.width, page.rect.height
        scope = page.rect
        if location == "left":
            scope = pymupdf.Rect(0, 0, w / 2 + 5, h)
        elif location == "right":
            scope = pymupdf.Rect(w / 2 - 5, 0, w, h)

        raw_tabs = [t for t in page.find_tables().tables if scope.intersects(pymupdf.Rect(t.bbox))]
        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("table")
        if not tables and not raw_tabs:
            return content

        # 1. Gán mỗi raw_tab cho bảng Gemini phù hợp nhất (best-match >= 60%)
        gem_tokens_list = [get_table_tokens(t) for t in tables]
        gem_table_map = {i: [] for i in range(len(tables))}

        for t in raw_tabs:
            t_tokens = get_table_tokens(t.extract())
            if not t_tokens:
                continue
            best_i = None
            best_ov = 0.0
            for g_idx, g_tokens in enumerate(gem_tokens_list):
                if not g_tokens:
                    continue
                ov = len(t_tokens & g_tokens) / len(t_tokens)
                if ov > best_ov:
                    best_ov = ov
                    best_i = g_idx
            if best_i is not None and best_ov >= 0.60:
                gem_table_map[best_i].append((best_ov, t))

        # Gom thêm các phần bảng collinear chưa được gán nhưng nằm liền kề (trong vòng 40pt)
        for g_idx in range(len(tables)):
            if gem_table_map[g_idx]:
                anchor = max(gem_table_map[g_idx], key=lambda x: x[0])[1]
                for t in raw_tabs:
                    # Bỏ qua nếu bảng t đã được gán vào bất kỳ bảng Gemini nào
                    if any(t in [x[1] for x in gem_table_map[k]] for k in range(len(tables))):
                        continue
                    t_tokens = get_table_tokens(t.extract())
                    if not t_tokens:
                        continue
                    ov = len(t_tokens & gem_tokens_list[g_idx]) / len(t_tokens)
                    if ov >= 0.25:
                        is_collinear = abs(t.bbox[0] - anchor.bbox[0]) <= 5.0 and abs(t.bbox[2] - anchor.bbox[2]) <= 5.0
                        is_nearby = abs(t.bbox[1] - anchor.bbox[3]) <= 40.0 or abs(anchor.bbox[1] - t.bbox[3]) <= 40.0
                        if is_collinear and is_nearby and t.col_count == anchor.col_count:
                            gem_table_map[g_idx].append((ov, t))

        # Xác định collinear_tabs cho từng bảng Gemini và tổng hợp các assigned_tabs
        table_collinear_tabs: dict[int, list] = {}
        assigned_tabs = set()

        for g_idx, table in enumerate(tables):
            gem_rows = table.find_all("tr")
            if not gem_rows:
                continue
            gem_row_count = len(gem_rows)
            matched_tabs = [x[1] for x in gem_table_map.get(g_idx, [])]
            collinear_tabs = []

            if len(matched_tabs) > 1:
                anchor = max(gem_table_map[g_idx], key=lambda x: x[0])[1]
                collinear = [
                    t
                    for t in matched_tabs
                    if abs(t.bbox[0] - anchor.bbox[0]) <= 5.0 and abs(t.bbox[2] - anchor.bbox[2]) <= 5.0
                ]
                collinear.sort(key=lambda t: t.bbox[1])
                valid_split = True
                for t_prev, t_next in zip(collinear[:-1], collinear[1:]):
                    gap = t_next.bbox[1] - t_prev.bbox[3]
                    if gap > 40.0 or t_next.col_count != t_prev.col_count:
                        valid_split = False
                        break
                if valid_split:
                    collinear_tabs = collinear
                else:
                    collinear_tabs = [anchor]
            elif len(matched_tabs) == 1:
                collinear_tabs = matched_tabs
            else:
                # Fallback: định vị bằng text anchor nếu không khớp raw_tabs
                bbox = locate_table_bbox_from_gemini(page, table, location=location)
                if bbox:
                    fb_tab = extract_table_from_bbox(page, bbox)
                    if fb_tab:
                        collinear_tabs = [fb_tab]

            if collinear_tabs:
                total_rows = sum(t.row_count for t in collinear_tabs)
                if 0.4 <= total_rows / gem_row_count <= 3.0:
                    table_collinear_tabs[g_idx] = collinear_tabs
                    assigned_tabs.update(collinear_tabs)

        unassigned_tabs = [t for t in raw_tabs if t not in assigned_tabs]

        # Kiểm tra nếu có bảng chưa được gán (unassigned)
        if unassigned_tabs:
            # Ghi toàn bộ nội dung parse hiện tại ra temp_page.md để người dùng tiện mở xem
            try:
                temp_file = Path("temp_page.md")
                preview_header = (
                    f"# BẢN XEM TRƯỚC: PDF TRANG {pdf_page_num if pdf_page_num else 'N/A'} (Vị trí: {location})\n\n"
                    f"> **Lưu ý:** Trang này có {len(raw_tabs)} bảng PyMuPDF nhưng có {len(unassigned_tabs)} bảng chưa được gán.\n\n"
                    f"---\n\n"
                )
                temp_file.write_text(preview_header + content + "\n", encoding="utf-8")
            except Exception:
                pass

            chosen = prompt_unassigned_action(
                pdf_page_num=pdf_page_num,
                location=location,
                total_raw_tabs=len(raw_tabs),
                unassigned_count=len(unassigned_tabs),
                action=unassigned_action,
            )
            if chosen == "2":
                # Lựa chọn 2: Sử dụng lại Gemini OCR gốc, không can thiệp PyMuPDF
                return content
            elif chosen == "3":
                # Lựa chọn 3: Thay thế bằng toàn bộ các bảng từ PyMuPDF
                sorted_tabs = sorted(raw_tabs, key=lambda t: (t.bbox[1], t.bbox[0]))
                rendered_htmls = [
                    render_pymupdf_table_to_html(
                        t,
                        flatten_headers=flatten_headers,
                        header_separator=header_separator,
                    )
                    for t in sorted_tabs
                ]
                tables_fragment = BeautifulSoup("\n\n".join(rendered_htmls), "html.parser")
                if tables:
                    tables[0].replace_with(tables_fragment)
                    for extra_tbl in tables[1:]:
                        extra_tbl.decompose()
                else:
                    soup.append(tables_fragment)
                return str(soup)
            # chosen == "1": tiếp tục theo logic collinear_tabs chuẩn hóa bên dưới

        # 2. Xử lý từng bảng Gemini
        for g_idx, table in enumerate(tables):
            collinear_tabs = table_collinear_tabs.get(g_idx, [])
            if not collinear_tabs:
                continue

            # 3. Tái tạo bảng HTML từ collinear_tabs (bảo toàn 100% dữ liệu PyMuPDF)
            new_table = soup.new_tag("table")
            new_thead = soup.new_tag("thead")
            new_tbody = soup.new_tag("tbody")

            thead = table.find("thead")
            hint_k = len(thead.find_all("tr")) if thead else 1

            collinear_tabs.sort(key=lambda t: t.bbox[1])
            t_top = collinear_tabs[0]
            topo_top = compute_table_topology(t_top, header_rows_hint=hint_k)
            extract_top = t_top.extract()
            k = topo_top["header_rows"]
            col_count = t_top.col_count

            # 3a. Render bảng trên cùng (header + các dòng data đầu tiên nếu có)
            for r in range(topo_top["row_count"]):
                tr_tag = soup.new_tag("tr")
                is_header = r < k
                for c in range(topo_top["col_count"]):
                    if topo_top["covered"][(r, c)]:
                        continue

                    rowspan, colspan = topo_top["spans"].get((r, c), (1, 1))
                    tag_name = "th" if is_header else "td"
                    cell_tag = soup.new_tag(tag_name)

                    if rowspan > 1:
                        cell_tag["rowspan"] = str(rowspan)
                    if colspan > 1:
                        cell_tag["colspan"] = str(colspan)

                    cell_text = ""
                    if r < len(extract_top) and c < len(extract_top[r]):
                        cell_text = (extract_top[r][c] or "").strip().replace("\n", " ")

                    cell_tag.string = cell_text
                    tr_tag.append(cell_tag)

                if is_header:
                    new_thead.append(tr_tag)
                else:
                    new_tbody.append(tr_tag)

            # 3b. Render các bảng kế tiếp và text nằm trong khoảng cách (gap) giữa các bảng
            for prev_t, next_t in zip(collinear_tabs[:-1], collinear_tabs[1:]):
                gap_rect = pymupdf.Rect(prev_t.bbox[0] - 2, prev_t.bbox[3], prev_t.bbox[2] + 2, next_t.bbox[1])
                gap_words = page.get_text("words", clip=gap_rect)
                if gap_words:
                    gap_words.sort(key=lambda w: (w[1], w[0]))
                    gap_text = " ".join(w[4] for w in gap_words).strip()
                    if gap_text:
                        tr_tag = soup.new_tag("tr")
                        td_tag = soup.new_tag("td", colspan=str(col_count))
                        td_tag.string = gap_text
                        tr_tag.append(td_tag)
                        new_tbody.append(tr_tag)

                topo_next = compute_table_topology(next_t, header_rows_hint=0)
                extract_next = next_t.extract()
                for r in range(topo_next["row_count"]):
                    tr_tag = soup.new_tag("tr")
                    for c in range(topo_next["col_count"]):
                        if topo_next["covered"][(r, c)]:
                            continue

                        rowspan, colspan = topo_next["spans"].get((r, c), (1, 1))
                        cell_tag = soup.new_tag("td")

                        if rowspan > 1:
                            cell_tag["rowspan"] = str(rowspan)
                        if colspan > 1:
                            cell_tag["colspan"] = str(colspan)

                        cell_text = ""
                        if r < len(extract_next) and c < len(extract_next[r]):
                            cell_text = (extract_next[r][c] or "").strip().replace("\n", " ")

                        cell_tag.string = cell_text
                        tr_tag.append(cell_tag)

                    new_tbody.append(tr_tag)

            if len(new_thead.find_all("tr")) > 0:
                new_table.append(new_thead)
            if len(new_tbody.find_all("tr")) > 0:
                new_table.append(new_tbody)

            table.replace_with(new_table)

        if flatten_headers:
            for tbl in soup.find_all("table"):
                flatten_table_headers(tbl, separator=header_separator)

        return str(soup)
    except Exception:
        return content


def render_page_to_png_bytes(page: pymupdf.Page, is_cover: bool = False, dpi: int = 200) -> bytes:
    """
    Render toàn bộ trang PDF thực tế thành dữ liệu ảnh PNG (bytes).
    Nếu không phải trang bìa (is_cover=False), vẽ một đường kẻ xám siêu mảnh (1px)
    ở chính giữa nếp gấp (x = width / 2) bằng Pillow trên ảnh render để KHÔNG làm thay đổi
    đối tượng vector `page` (tránh việc PyMuPDF nhận nhầm đường kẻ giữa trang thành đường viền cột bảng).
    """
    pix = page.get_pixmap(dpi=dpi)
    raw_png = pix.tobytes("png")
    if is_cover:
        return raw_png

    img = Image.open(io.BytesIO(raw_png))
    draw = ImageDraw.Draw(img)
    mid_x = round(img.width / 2.0)
    draw.line([(mid_x, 0), (mid_x, img.height)], fill=(192, 192, 192), width=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def clean_and_parse_json(text: str) -> dict:
    """Xử lý và parse chuỗi JSON từ phản hồi của Gemini."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


def call_gemini_ocr_with_retry(
    client: genai.Client,
    image_bytes: bytes,
    model: str,
    user_prompt_text: str,
    system_prompt_text: str = SYSTEM_PROMPT,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> list[dict]:
    """Gửi ảnh trang tài liệu tới Gemini để trích xuất JSON các khối {location, content}."""
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image_part, user_prompt_text],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt_text,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=PageParseOutput,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            raw_text = response.text.strip() if response.text else "{}"
            parsed = clean_and_parse_json(raw_text)
            return parsed.get("blocks", [])
        except APIError as e:
            print(f"    [Cảnh báo] Lỗi Gemini API (lần thử {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                sleep_time = retry_delay * (2 ** (attempt - 1))
                print(f"    Đợi {sleep_time:.1f}s trước khi thử lại...")
                time.sleep(sleep_time)
            else:
                raise
        except Exception as e:
            print(f"    [Cảnh báo] Lỗi khi gọi/parse Gemini (lần thử {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                sleep_time = retry_delay * (2 ** (attempt - 1))
                print(f"    Đợi {sleep_time:.1f}s trước khi thử lại...")
                time.sleep(sleep_time)
            else:
                raise

    return []


def build_toc_ranges(
    raw_toc: TOCExtractionOutput | dict,
    total_logical_pages: int = 400,
) -> dict[str, list[tuple[str, tuple[int, int]]]]:
    """
    Chuyển đổi dữ liệu trích xuất mục lục sang định dạng:
    {
      "Chương x: <Tên chương>": [
        ("<Ý quan trọng 1>", (start_page, end_page)),
        ("<Ý quan trọng 2>", (start_page, end_page)),
      ],
      ...
    }
    Khoảng (start, end) được tính toán liên tục: end = next_start - 1.
    Mục cuối cùng lấy đến total_logical_pages.
    """
    if isinstance(raw_toc, dict):
        chapters_data = raw_toc.get("chapters", [])
    else:
        chapters_data = raw_toc.chapters

    parsed_chapters = []
    for ch in chapters_data:
        if isinstance(ch, dict):
            ch_title = ch.get("chapter_title", "").strip()
            ch_start = ch.get("start_page")
            sections = ch.get("sections", [])
        else:
            ch_title = ch.chapter_title.strip()
            ch_start = ch.start_page
            sections = ch.sections

        sec_items = []
        for s in sections:
            if isinstance(s, dict):
                s_title = s.get("title", "").strip()
                s_start = s.get("start_page")
            else:
                s_title = s.title.strip()
                s_start = s.start_page
            if s_title and s_start is not None:
                sec_items.append((s_title, int(s_start)))

        # Nếu chương không có section con, lấy chính chapter_title làm section
        if not sec_items and ch_start is not None:
            sec_items.append((ch_title, int(ch_start)))

        if sec_items:
            sec_items.sort(key=lambda x: x[1])
            # Nếu chương có ch_start và mục con đầu tiên bắt đầu sau ch_start:
            # Tự động chèn mục mở đầu chương để bao trọn khoảng bìa/mở đầu (ví dụ: trang 8 đến 9)
            if ch_start is not None and int(ch_start) < sec_items[0][1]:
                sec_items.insert(0, ("Mở đầu chương", int(ch_start)))

            parsed_chapters.append(
                {
                    "chapter_title": ch_title,
                    "chapter_start": int(ch_start) if ch_start is not None else sec_items[0][1],
                    "sections": sec_items,
                }
            )

    result_dict: dict[str, list[tuple[str, tuple[int, int]]]] = {}

    for ch_idx, ch_info in enumerate(parsed_chapters):
        ch_title = ch_info["chapter_title"]
        secs = ch_info["sections"]
        built_secs = []

        for s_idx, (s_title, s_start) in enumerate(secs):
            if s_idx + 1 < len(secs):
                next_start = secs[s_idx + 1][1]
                end_page = max(s_start, next_start - 1)
            else:
                if ch_idx + 1 < len(parsed_chapters):
                    next_ch = parsed_chapters[ch_idx + 1]
                    next_start = next_ch["chapter_start"] or next_ch["sections"][0][1]
                    end_page = max(s_start, next_start - 1)
                else:
                    end_page = max(s_start, total_logical_pages)

            built_secs.append((s_title, (s_start, end_page)))

        result_dict[ch_title] = built_secs

    return result_dict


def lookup_section_for_page(
    toc_dict: dict,
    chapter: str,
    page_num: int,
) -> tuple[str, str]:
    """
    Tra cứu số trang logic vào cấu trúc TOC để tìm Ý chính (section) và Chương (chapter).
    Ưu tiên tra cứu theo khoảng trang (start <= page_num <= end) trong toàn bộ TOC,
    bởi vì số trang logic được in cố định trên sách/báo cáo và phản ánh cấu trúc tuyệt đối,
    kể cả ở những trang bìa/phân cách không có header PyMuPDF.
    Trả về: (tên_ý_chính, tên_chương).
    """
    if not toc_dict or page_num is None:
        return chapter or "Nội dung", chapter or "Nội dung"

    # 1. Quét tìm khoảng trang chứa page_num trong toàn bộ TOC
    for ch_key, sections in toc_dict.items():
        for sec_title, (start, end) in sections:
            if start <= page_num <= end:
                return sec_title, ch_key

    # 2. Fallback an toàn nếu trang nằm ngoài phạm vi TOC
    return chapter or "Nội dung", chapter or "Nội dung"


def sanitize_content_headings(content: str) -> str:
    """
    Chuẩn hóa các đề mục trong nội dung thân trang:
    - Ép tất cả các đề mục # (H1) và ## (H2) thành ### (H3).
    - Giữ nguyên các đề mục từ ### trở xuống (###, ####, ...).
    - Không làm ảnh hưởng đến code block (```...```).
    """
    if not content:
        return content

    lines = content.split("\n")
    in_code_block = False
    result_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result_lines.append(line)
            continue

        if not in_code_block:
            m = re.match(r"^(#{1,2})\s+(.+)$", line)
            if m:
                rest = m.group(2)
                line = f"### {rest}"

        result_lines.append(line)

    return "\n".join(result_lines)


def parse_toc_page(
    client: genai.Client,
    image_bytes: bytes,
    model: str,
    total_logical_pages: int = 400,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict[str, list[tuple[str, tuple[int, int]]]]:
    """
    Gửi ảnh trang mục lục tới Gemini để trích xuất TOC structured JSON và
    tính toán các khoảng trang (start, end) cho từng ý quan trọng.
    """
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image_part, TOC_USER_PROMPT],
                config=types.GenerateContentConfig(
                    system_instruction=TOC_SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=TOCExtractionOutput,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            raw_text = response.text.strip() if response.text else "{}"
            parsed_json = clean_and_parse_json(raw_text)
            parsed_toc = TOCExtractionOutput.model_validate(parsed_json)
            return build_toc_ranges(parsed_toc, total_logical_pages=total_logical_pages)
        except Exception as e:
            print(f"    [Cảnh báo] Lỗi khi trích xuất mục lục (lần thử {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                sleep_time = retry_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)
            else:
                raise

    return {}


def map_blocks_to_logical_pages(
    blocks: list[dict],
    plan: dict,
) -> list[dict]:
    """
    Ánh xạ các blocks (location, content) thành các đối tượng có logical_page là list:
    - location == 'all'   -> 1 đối tượng DUY NHẤT với logical_page: [left_logical_page, right_logical_page]
                             (hoặc [1] nếu là bìa).
    - location == 'left'  -> 1 đối tượng với logical_page: [left_logical_page].
    - location == 'right' -> 1 đối tượng với logical_page: [right_logical_page].
    """
    pdf_page = plan["pdf_page"]
    chapter = plan["chapter"]

    if plan["is_cover"]:
        all_content = "\n\n".join(b["content"].strip() for b in blocks if b.get("content", "").strip())
        if not all_content:
            return []
        return [
            {
                "logical_page": [1],
                "pdf_page": 1,
                "chapter": chapter,
                "content": all_content,
            }
        ]

    left_page = plan["left_logical_page"]
    right_page = plan["right_logical_page"]

    left_parts = []
    right_parts = []
    all_parts = []

    for b in blocks:
        loc = b.get("location", "all")
        content = b.get("content", "").strip()
        if not content:
            continue

        if loc == "left":
            left_parts.append(content)
        elif loc == "right":
            right_parts.append(content)
        elif loc == "all":
            all_parts.append(content)

    results = []

    # 1. Nếu có phần location = 'all', chỉ lưu vào 1 đối tượng duy nhất có logical_page = [left, right]
    if all_parts:
        pages = []
        if left_page is not None:
            pages.append(left_page)
        if right_page is not None:
            pages.append(right_page)

        results.append(
            {
                "logical_page": pages,
                "pdf_page": pdf_page,
                "chapter": chapter,
                "content": "\n\n".join(all_parts),
            }
        )

    # 2. Phần location = 'left' -> logical_page = [left]
    if left_parts and left_page is not None:
        results.append(
            {
                "logical_page": [left_page],
                "pdf_page": pdf_page,
                "chapter": chapter,
                "content": "\n\n".join(left_parts),
            }
        )

    # 3. Phần location = 'right' -> logical_page = [right]
    if right_parts and right_page is not None:
        results.append(
            {
                "logical_page": [right_page],
                "pdf_page": pdf_page,
                "chapter": chapter,
                "content": "\n\n".join(right_parts),
            }
        )

    return results


def extract_pdf_with_gemini(
    input_path: Path,
    output_path: Path,
    start_page: int = 1,
    end_page: int | None = None,
    dpi: int = 200,
    min_words: int = 50,
    delay_seconds: float = 1.0,
    save_images_dir: Path | None = None,
    resume: bool = False,
    unassigned_action: Optional[str] = None,
) -> None:
    """
    Xử lý file PDF:
    1. Preprocessing: Quét metadata (header, số trang logic, tên chương) và lập plan cho từng trang PDF.
    2. Render toàn bộ trang PDF thực tế (không cắt đôi).
    3. Gửi ảnh sang Gemini OCR lấy danh sách [{location, content}].
    4. Ánh xạ (map) nội dung vào từng trang logic (location == 'all' gán vào cả 2 trang logic).
    5. Xuất kết quả ra file Markdown và file JSON.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF tại: {input_path}")

    api_key, model = get_gemini_config()
    client = init_gemini_client(api_key=api_key)

    start_time = time.time()
    doc = pymupdf.open(str(input_path))
    total_pdf_pages = len(doc)

    if total_pdf_pages == 0:
        print("Cảnh báo: File PDF không có trang nào.")
        return

    start_idx = max(0, start_page - 1)
    end_idx = min(total_pdf_pages, end_page) if end_page else total_pdf_pages

    if start_idx >= end_idx:
        raise ValueError(f"Khoảng trang không hợp lệ: trang {start_page} đến {end_page}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path = output_path.with_suffix(".json")
    toc_output_path = output_path.parent / "toc.json"

    if save_images_dir:
        save_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Bắt đầu tiền xử lý và lập Parse Plan: {input_path.name} ===")
    plans = build_parsing_plan(doc, start_idx=start_idx, end_idx=end_idx)
    print(f"- Tổng số trang PDF cần xử lý: {len(plans)}")
    print(f"- Model Gemini (từ .env): {model}")
    print(f"- DPI ảnh gửi OCR: {dpi}")
    print(f"- File kết quả Markdown: {output_path.resolve()}")
    print(f"- File kết quả JSON: {json_output_path.resolve()}")
    print(f"- File mục lục TOC: {toc_output_path.resolve()}\n")

    processed_logical_count = 0
    skipped_count = 0
    all_logical_pages_data: list[dict] = []
    layout_tracker = LayoutTracker()
    toc_dict: dict = {}
    last_written_chapter: Optional[str] = None
    last_written_section: Optional[str] = None

    # Nếu resume, nạp dữ liệu cũ đã lưu từ json_output_path
    if resume and json_output_path.exists():
        try:
            with open(json_output_path, "r", encoding="utf-8") as jf:
                loaded_data = json.load(jf)
            # Giữ lại các trang đã xử lý trước start_page
            all_logical_pages_data = [d for d in loaded_data if d.get("pdf_page", 0) < (start_idx + 1)]
            processed_logical_count = len(all_logical_pages_data)
            print(
                f"[i] Chế độ Resume: Đã khôi phục {processed_logical_count} trang logic đã lưu từ {json_output_path.name}"
            )
            if all_logical_pages_data:
                last_item = all_logical_pages_data[-1]
                last_written_chapter = last_item.get("chapter")
                last_written_section = last_item.get("section")
                layout_tracker.set_context(chapter=last_written_chapter, section=last_written_section)
                if last_item.get("current_headings"):
                    layout_tracker.current_h3 = last_item["current_headings"][-1]
                print(
                    f"    Ngữ cảnh kế thừa: Chương '{last_written_chapter}' | Ý chính '{last_written_section}' | H3 '{layout_tracker.current_h3}'"
                )
        except Exception as e:
            print(f"[!] Cảnh báo không thể nạp file JSON cũ để resume: {e}")

    # Nếu đã có file toc.json từ trước và start_page >= 3, tự động nạp
    if toc_output_path.exists():
        try:
            with open(toc_output_path, "r", encoding="utf-8") as tf:
                toc_dict = json.load(tf)
                print(f"[i] Đã nạp cấu trúc mục lục có sẵn từ: {toc_output_path.name}")
        except Exception:
            pass

    # Nếu bắt đầu từ trang >= 3 nhưng chưa có toc.json: tự động trích xuất trang 2 trước
    if not toc_dict and start_idx >= 2 and len(doc) >= 2:
        print("[i] Chưa tìm thấy mục lục (toc.json). Đang tự động trích xuất Mục lục từ PDF Trang 2 trước...")
        try:
            page2 = doc[1]
            img_b2 = render_page_to_png_bytes(page2, is_cover=False, dpi=dpi)
            toc_dict = parse_toc_page(
                client=client,
                image_bytes=img_b2,
                model=model,
                total_logical_pages=total_pdf_pages * 2,
            )
            with open(toc_output_path, "w", encoding="utf-8") as tf:
                json.dump(toc_dict, tf, ensure_ascii=False, indent=2)
            print(f"    [✓] Đã tạo thành công {toc_output_path.name} ({len(toc_dict)} phần/chương)")
        except Exception as e:
            print(f"    [!] Cảnh báo không thể tự động trích xuất mục lục: {e}")

    # Đồng bộ file markdown khi resume: loại bỏ các trang từ start_page trở đi để tránh trùng lặp
    if resume and output_path.exists() and start_idx > 0:
        pattern = rf"(?m)(?:\n+# [^\n]+)*(?:\n+## [^\n]+)*\n+<!-- Trang logic [^()]*\(PDF Trang {start_idx + 1}\) -->"
        try:
            old_md = output_path.read_text(encoding="utf-8")
            match = re.search(pattern, old_md)
            if match:
                truncated_md = old_md[: match.start()].rstrip() + "\n\n"
                output_path.write_text(truncated_md, encoding="utf-8")
                print(f"[i] Chế độ Resume: Đã đồng bộ file Markdown, cắt bỏ dữ liệu cũ từ PDF Trang {start_idx + 1}")
        except Exception as e:
            print(f"[!] Cảnh báo không thể đồng bộ Markdown cũ khi resume: {e}")

    file_mode = "a" if (resume and output_path.exists()) else "w"
    with open(output_path, file_mode, encoding="utf-8") as f:
        if file_mode == "w":
            # Header tài liệu
            f.write(f"# {input_path.stem}\n\n")
            f.write(f"- **Nguồn file:** `{input_path.name}`\n")
            f.write(f"- **Thời gian xử lý:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Phạm vi trang PDF gốc:** {start_idx + 1} -> {end_idx} (Tổng: {total_pdf_pages} trang)\n")
            f.write(f"- **Model OCR (từ .env):** `{model}`\n\n")
            f.write("---\n\n")
            f.flush()

        for plan in plans:
            pdf_page_num = plan["pdf_page"]
            is_cover = plan["is_cover"]
            left_page = plan["left_logical_page"]
            right_page = plan["right_logical_page"]
            chapter = plan["chapter"]

            # 1. Bỏ qua việc parse trang bìa (PDF Trang 1)
            if pdf_page_num == 1 or is_cover:
                print(f"[-] Bỏ qua trang bìa: PDF Trang {pdf_page_num} (theo cấu hình)")
                skipped_count += 1
                continue

            # 2. Xử lý riêng trang mục lục (PDF Trang 2)
            if pdf_page_num == 2:
                print("[+] Đang trích xuất Trang Mục lục (PDF Trang 2)...")
                t_page = time.time()
                page = doc[pdf_page_num - 1]
                img_bytes = render_page_to_png_bytes(page, is_cover=False, dpi=dpi)

                if save_images_dir:
                    img_filename = save_images_dir / f"pdf_page_{pdf_page_num}_toc.png"
                    with open(img_filename, "wb") as img_file:
                        img_file.write(img_bytes)

                total_logical_pages = total_pdf_pages * 2
                try:
                    toc_dict = parse_toc_page(
                        client=client,
                        image_bytes=img_bytes,
                        model=model,
                        total_logical_pages=total_logical_pages,
                    )
                    with open(toc_output_path, "w", encoding="utf-8") as tf:
                        json.dump(toc_dict, tf, ensure_ascii=False, indent=2)
                    print(
                        f"    [✓] Đã bóc tách mục lục thành công ({len(toc_dict)} phần/chương) -> Lưu tại: {toc_output_path.name}"
                    )
                except Exception as e:
                    print(f"    [!] Lỗi khi bóc tách mục lục: {e}")

                page_elapsed = time.time() - t_page
                print(f"    [✓] Hoàn thành PDF Trang Mục lục trong {page_elapsed:.1f}s")
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                continue

            # 3. Các trang nội dung từ Trang 3 trở đi
            if not plan["has_content"]:
                print(f"[-] Bỏ qua: PDF Trang {pdf_page_num} (Trang trắng / không có nội dung thân trang)")
                skipped_count += 1
                continue

            # Xác định Ý chính (section) cho trang hiện tại dựa vào TOC và PyMuPDF
            sample_page = left_page if left_page is not None else right_page
            current_section, current_chapter = lookup_section_for_page(toc_dict, chapter=chapter, page_num=sample_page)
            active_chapter = current_chapter if current_chapter and current_chapter != "Nội dung" else chapter

            # Bỏ qua các trang mở đầu chương (chỉ chứa bìa chương / danh mục bài viết)
            if current_section == "Mở đầu chương":
                print(
                    f"[-] Bỏ qua trang mở đầu chương: PDF Trang {pdf_page_num} (Trang logic {left_page} - {right_page} | {active_chapter})"
                )
                skipped_count += 1
                continue

            page_desc = f"Trang logic {left_page} - {right_page}"
            print(
                f"[+] Đang xử lý: PDF Trang {pdf_page_num} ({page_desc} | Chương: {active_chapter} | Ý chính: {current_section})..."
            )
            t_page = time.time()

            page = doc[pdf_page_num - 1]

            # Render toàn bộ trang PDF thực tế
            img_bytes = render_page_to_png_bytes(page, is_cover=False, dpi=dpi)

            if save_images_dir:
                img_filename = save_images_dir / f"pdf_page_{pdf_page_num}.png"
                with open(img_filename, "wb") as img_file:
                    img_file.write(img_bytes)

            # Cập nhật ngữ cảnh phân cấp Chương và Ý chính vào layout_tracker
            layout_tracker.set_context(chapter=active_chapter, section=current_section)
            layout_ctx = layout_tracker.get_prompt_context()

            prompt_parts = [USER_PROMPT]
            if layout_ctx:
                prompt_parts.append(layout_ctx)

            # Kiểm tra nếu trang có bảng thì append chỉ dẫn chuyên sâu cho bảng từ hình ảnh
            try:
                tabs = page.find_tables()
                has_table = len(tabs.tables) > 0
            except Exception:
                has_table = False

            if has_table:
                prompt_parts.append(TABLE_INSTRUCTION_PROMPT)

            current_user_prompt = "\n\n".join(prompt_parts)

            try:
                raw_blocks = call_gemini_ocr_with_retry(
                    client=client,
                    image_bytes=img_bytes,
                    model=model,
                    user_prompt_text=current_user_prompt,
                    system_prompt_text=SYSTEM_PROMPT,
                )
            except Exception as e:
                print(f"    [!] Lỗi trích xuất PDF Trang {pdf_page_num}: {e}")
                raw_blocks = [{"location": "all", "content": f"Lỗi OCR khi gọi Gemini: {e}"}]

            # Chuẩn hóa đề mục: ép mọi đề mục # hoặc ## thành ###
            for b in raw_blocks:
                b["content"] = sanitize_content_headings(b.get("content", ""))

            # Chuẩn hóa bảng: nếu bảng có header đa cấp từ PyMuPDF hoặc có bảng unassigned
            for b in raw_blocks:
                has_gemini_table = "<table" in b.get("content", "")
                w, h = page.rect.width, page.rect.height
                loc = b.get("location", "all")
                scope = page.rect
                if loc == "left":
                    scope = pymupdf.Rect(0, 0, w / 2 + 5, h)
                elif loc == "right":
                    scope = pymupdf.Rect(w / 2 - 5, 0, w, h)
                has_pymupdf_tables = any(scope.intersects(pymupdf.Rect(t.bbox)) for t in page.find_tables().tables)

                if has_gemini_table or has_pymupdf_tables:
                    b["content"] = enhance_table_with_pymupdf_headers(
                        b.get("content", ""),
                        page,
                        location=loc,
                        pdf_page_num=pdf_page_num,
                        unassigned_action=unassigned_action,
                    )

            # Ánh xạ nội dung vào các trang logic
            mapped_pages = map_blocks_to_logical_pages(raw_blocks, plan)

            for lp in mapped_pages:
                lp_first_page = lp["logical_page"][0] if lp.get("logical_page") else None
                sec, ch = lookup_section_for_page(toc_dict, chapter=active_chapter, page_num=lp_first_page)
                lp["chapter"] = ch
                lp["section"] = sec

                # Cập nhật trạng thái H3 độc lập và chính xác cho từng trang logic (chỉ xét ###)
                current_h3_list = layout_tracker.update_for_logical_page(
                    content=lp.get("content", ""),
                    chapter=ch,
                    section=sec,
                )
                lp["current_headings"] = current_h3_list

                # Ghi đề mục Chương nếu chuyển sang chương mới
                if ch != last_written_chapter:
                    f.write(f"# {ch}\n\n")
                    last_written_chapter = ch
                    last_written_section = None

                # Ghi đề mục Ý chính nếu chuyển sang ý chính mới
                if sec != last_written_section:
                    f.write(f"## {sec}\n\n")
                    last_written_section = sec

                l_pages_str = ", ".join(str(p) for p in lp["logical_page"])
                c_text = lp["content"]

                # Chèn ghi chú trang logic dưới dạng comment để không làm xáo trộn phân cấp Markdown
                f.write(f"<!-- Trang logic {l_pages_str} (PDF Trang {pdf_page_num}) -->\n\n")
                f.write(c_text if c_text else "*[Không có nội dung văn bản]*")
                f.write("\n\n---\n\n")
                f.flush()

                all_logical_pages_data.append(lp)
                processed_logical_count += 1

            # Lưu file JSON định kỳ sau mỗi trang để bảo toàn tiến trình
            with open(json_output_path, "w", encoding="utf-8") as jf:
                json.dump(all_logical_pages_data, jf, ensure_ascii=False, indent=2)

            page_elapsed = time.time() - t_page
            print(
                f"    [✓] Hoàn thành PDF Trang {pdf_page_num} trong {page_elapsed:.1f}s -> Tạo {len(mapped_pages)} trang logic"
            )

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    # Đảm bảo ghi file JSON hoàn chỉnh cuối cùng
    with open(json_output_path, "w", encoding="utf-8") as jf:
        json.dump(all_logical_pages_data, jf, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("\n=== Hoàn tất quá trình trích xuất! ===")
    print(f"- Số trang logic đã tạo: {processed_logical_count}")
    print(f"- Số trang PDF bỏ qua (trống): {skipped_count}")
    print(f"- File kết quả Markdown: {output_path.resolve()}")
    print(f"- File kết quả JSON: {json_output_path.resolve()}")
    print(f"- Tổng thời gian: {elapsed:.2f} giây")


def main():
    default_dir = Path(__file__).resolve().parent
    default_pdf = default_dir / "techcombank-bao-cao-thuong-nien-2025-vie-update.pdf"
    default_md = default_dir / "output_gemini.md"

    parser = argparse.ArgumentParser(
        description="Đọc toàn bộ trang PDF thực tế, dùng Gemini OCR trích xuất các khối {location, content} "
        "và ánh xạ vào trang logic tương ứng."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=default_pdf,
        help=f"Đường dẫn file PDF đầu vào (mặc định: {default_pdf.name})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_md,
        help=f"Đường dẫn file MD đầu ra (mặc định: {default_md.name})",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Trang bắt đầu của PDF gốc (1-indexed, mặc định: 1)",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Trang kết thúc của PDF gốc (1-indexed, mặc định: hết file)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI khi render ảnh gửi Gemini (mặc định: 200)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Thời gian nghỉ (giây) giữa các request để tránh rate-limit (mặc định: 1.0)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Số từ tối thiểu để tạo placeholder cho đoạn văn bản (mặc định: 50)",
    )
    parser.add_argument(
        "--save-images-dir",
        type=Path,
        default=None,
        help="Thư mục tùy chọn để lưu các ảnh render phục vụ kiểm tra trực quan",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục chạy nối tiếp từ dữ liệu đã có (append vào file JSON và Markdown hiện tại)",
    )
    parser.add_argument(
        "--unassigned-action",
        type=str,
        choices=["1", "2", "3"],
        default=None,
        help="Lựa chọn xử lý khi phát hiện bảng chưa gán (1: để nguyên, 2: dùng Gemini gốc, 3: dùng tất cả bảng PyMuPDF; mặc định: hỏi người dùng)",
    )

    args = parser.parse_args()

    extract_pdf_with_gemini(
        input_path=args.input,
        output_path=args.output,
        start_page=args.start_page,
        end_page=args.end_page,
        dpi=args.dpi,
        min_words=args.min_words,
        delay_seconds=args.delay,
        save_images_dir=args.save_images_dir,
        resume=args.resume,
        unassigned_action=args.unassigned_action,
    )


if __name__ == "__main__":
    main()
