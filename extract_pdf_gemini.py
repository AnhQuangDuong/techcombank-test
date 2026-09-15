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
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from bs4 import BeautifulSoup

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


def check_page_has_meaningful_content(page: pymupdf.Page, header_height: float = 65.0) -> bool:
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
    for img in page.get_image_info(xrefs=True):
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
        top_left = pymupdf.Rect(0, 0, mid_x, 60)
        tl_text = page.get_text(clip=top_left).strip()

        # Header góc trên bên phải (right logical page number + chapter name)
        top_right = pymupdf.Rect(mid_x, 0, w, 60)
        tr_text = page.get_text(clip=top_right).strip()

        fallback_left = idx * 2
        fallback_right = idx * 2 + 1

        # Trích xuất số trang logic bên trái (bỏ qua số năm 2025)
        left_num = fallback_left
        for line in tl_text.split("\n"):
            line = line.strip()
            if line.isdigit() and line != "2025":
                left_num = int(line)
                break

        # Trích xuất số trang logic bên phải và tên chương
        right_num = fallback_right
        detected_chapter = None
        for line in tr_text.split("\n"):
            line = line.strip()
            if line.isdigit():
                right_num = int(line)
            else:
                cleaned = re.sub(r"^[|\-–\s\d]+|[|\-–\s\d]+$", "", line).strip()
                if cleaned and len(cleaned) > 2 and cleaned.lower() != "báo cáo thường niên":
                    detected_chapter = cleaned

        if detected_chapter:
            current_chapter = detected_chapter

        # Kiểm tra trang có nội dung thực sự hay không (loại trừ header)
        has_content = check_page_has_meaningful_content(page, header_height=65.0)

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


def get_clean_headers_from_pymupdf(page: pymupdf.Page) -> list[list[str]]:
    """
    Trích xuất danh sách các tiêu đề cột chuẩn từ bảng hình học của PyMuPDF.
    CHỈ áp dụng cho các bảng có header đa cấp / merged header (chứa ô None ở dòng 0).
    Tự động kết hợp nhóm cha và cột con (ví dụ: 'Các quỹ - Quỹ dự phòng...') để đảm bảo
    mỗi cột đều có tên đầy đủ và số lượng cột khớp 100% với số ô dữ liệu.
    """
    try:
        tabs = page.find_tables()
        if not tabs.tables:
            return []

        headers_list = []
        for tab in tabs.tables:
            df = tab.extract()
            if len(df) < 2:
                continue
            row0 = df[0]
            # CHỈ xử lý các bảng có ô gộp ở dòng tiêu đề đầu tiên (chứa None)
            if not any(c is None for c in row0):
                continue

            row1 = df[1]

            parents = []
            curr = ""
            for c in row0:
                if c is None:
                    parents.append(curr)
                elif c and c.strip():
                    curr = c.strip().replace("\n", " ")
                    parents.append(curr)
                else:
                    curr = ""
                    parents.append("")

            unified = []
            for idx in range(tab.col_count):
                p = parents[idx] if idx < len(parents) else ""
                c = row1[idx].strip().replace("\n", " ") if idx < len(row1) and row1[idx] else ""
                if p and c and p != c:
                    unified.append(f"{p} - {c}")
                elif c:
                    unified.append(c)
                elif p:
                    unified.append(p)
                else:
                    unified.append("")
            headers_list.append(unified)
        return headers_list
    except Exception:
        return []


def enhance_table_with_pymupdf_headers(content: str, page: pymupdf.Page) -> str:
    """
    Tự động chuẩn hóa và cân bằng cột cho các bảng HTML trong nội dung:
    - Xác định số cột thực tế N_cols từ các dòng dữ liệu <td>.
    - Đối chiếu với các tiêu đề chuẩn từ PyMuPDF có cùng N_cols.
    - Cập nhật lại <thead> với danh sách cột chuẩn xác 100%, loại bỏ lỗi lệch cột do colspan.
    """
    if "<table" not in content:
        return content

    pymupdf_headers = get_clean_headers_from_pymupdf(page)
    if not pymupdf_headers:
        return content

    try:
        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return content

        for table in tables:
            tbody = table.find("tbody") or table
            data_rows = [tr for tr in tbody.find_all("tr") if tr.find_all("td")]
            if not data_rows:
                continue

            col_counts = [len(tr.find_all("td")) for tr in data_rows]
            n_cols = Counter(col_counts).most_common(1)[0][0]

            # Tìm header từ PyMuPDF có đúng n_cols cột
            matching_header = None
            for h in pymupdf_headers:
                if len(h) == n_cols:
                    matching_header = h
                    break

            if matching_header:
                new_thead = soup.new_tag("thead")
                new_tr = soup.new_tag("tr")
                for col_name in matching_header:
                    th = soup.new_tag("th")
                    th.string = col_name
                    new_tr.append(th)
                new_thead.append(new_tr)

                # Xóa toàn bộ header cũ (trong <thead> hoặc các <tr> chỉ chứa <th>)
                old_thead = table.find("thead")
                if old_thead:
                    old_thead.decompose()

                for tr in list(table.find_all("tr")):
                    if tr.find_all("th") and not tr.find_all("td"):
                        tr.decompose()

                # Chèn new_thead vào đầu bảng
                table.insert(0, new_thead)

        return str(soup)
    except Exception:
        return content


def render_page_to_png_bytes(page: pymupdf.Page, is_cover: bool = False, dpi: int = 200) -> bytes:
    """
    Render toàn bộ trang PDF thực tế thành dữ liệu ảnh PNG (bytes).
    Nếu không phải trang bìa (is_cover=False), vẽ một đường kẻ xám siêu mảnh (0.5 pt)
    ở chính giữa nếp gấp (x = width / 2) để hỗ trợ Gemini OCR phân định left / right / all.
    """
    if not is_cover:
        mid_x = page.rect.width / 2.0
        page.draw_line(
            p1=pymupdf.Point(mid_x, 0),
            p2=pymupdf.Point(mid_x, page.rect.height),
            color=(0.75, 0.75, 0.75),  # Xám nhạt
            width=0.5,  # Siêu mảnh (~1 pixel ở 200 DPI)
        )
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


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

            # Chuẩn hóa bảng: nếu bảng có header đa cấp từ PyMuPDF thì cập nhật cột
            for b in raw_blocks:
                if "<table" in b.get("content", ""):
                    b["content"] = enhance_table_with_pymupdf_headers(b["content"], page)

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
    )


if __name__ == "__main__":
    main()
