#!/usr/bin/env python3
"""
chunk_post_processor.py

Script xử lý hậu kỳ cho các chunk dữ liệu:
1. Đọc số trang của 'Danh mục thuật ngữ viết tắt' từ toc.json.
2. Trích xuất bảng thuật ngữ từ output_gemini.json, tạo từ điển abbreviations.json.
3. Duyệt qua tất cả các chunk trong output_chunks.json:
   - Thay thế từ viết tắt thành: Từ_viết_tắt (Định_nghĩa).
   - Chỉ thay thế ở lần đầu tiên xuất hiện trong mỗi chunk.
   - Không thay thế nếu từ đã nằm trong ngoặc đơn hoặc đã có giải nghĩa liền sau.
4. Ghi kết quả sang pp_output_chunks.json (giữ nguyên metadata gốc).
"""

import argparse
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_abbreviation_logical_pages(toc_path: str) -> List[int]:
    """Tìm danh sách số trang logical của Danh mục thuật ngữ viết tắt từ toc.json."""
    with open(toc_path, "r", encoding="utf-8") as f:
        toc_data = json.load(f)

    target_pages: List[int] = []
    for chapter_title, sections in toc_data.items():
        if "thuật ngữ" in chapter_title.lower() and "viết tắt" in chapter_title.lower():
            for section in sections:
                if len(section) >= 2 and isinstance(section[1], list):
                    page_range = section[1]
                    if len(page_range) == 2:
                        target_pages.extend(range(page_range[0], page_range[1] + 1))
                    elif len(page_range) == 1:
                        target_pages.append(page_range[0])

    if not target_pages:
        # Fallback nếu cấu trúc key khác
        logger.warning(
            "Không tìm thấy chương 'Danh mục thuật ngữ viết tắt' theo key. Sử dụng trang mặc định [386, 387]."
        )
        target_pages = [386, 387]

    return sorted(list(set(target_pages)))


def parse_term_entry(raw_term: str, raw_defn: str) -> List[Tuple[str, str]]:
    """
    Chuẩn hóa cặp thuật ngữ - định nghĩa.
    Xử lý trường hợp đặc biệt như: 'Doanh nghiệp vừa và nhỏ (SME)' -> ('SME', 'Doanh nghiệp vừa và nhỏ')
    """
    term = raw_term.strip()
    defn = raw_defn.strip()
    if not term:
        return []

    # Kiểm tra dạng: Tiếng Việt (Viết tắt)
    match = re.match(r"^(.*?)\s*\(([^)]+)\)$", term)
    if match:
        full_name = match.group(1).strip()
        acronym = match.group(2).strip()
        # Định nghĩa cho từ viết tắt ưu tiên lấy full_name ngắn gọn
        return [(acronym, full_name)]

    # Cặp chuẩn: Viết tắt -> Định nghĩa
    # Nếu định nghĩa rỗng, bỏ qua
    if not defn:
        return []

    return [(term, defn)]


def extract_abbreviations_from_gemini(
    toc_path: str = "toc.json",
    gemini_path: str = "output_gemini.json",
    save_dict_path: Optional[str] = "abbreviations.json",
) -> Dict[str, str]:
    """
    Trích xuất từ điển viết tắt từ output_gemini.json dựa trên các trang chỉ định trong toc.json.
    """
    target_pages = find_abbreviation_logical_pages(toc_path)
    logger.info(f"Trang thuật ngữ viết tắt cần parse: {target_pages}")

    with open(gemini_path, "r", encoding="utf-8") as f:
        gemini_pages = json.load(f)

    abbr_dict: Dict[str, str] = {}

    for page in gemini_pages:
        logical_pages = page.get("logical_page", [])
        chapter = page.get("chapter", "")

        is_target_page = any(p in target_pages for p in logical_pages)
        is_target_chapter = "thuật ngữ" in chapter.lower() and "viết tắt" in chapter.lower()

        if is_target_page or is_target_chapter:
            content = page.get("content", "")
            soup = BeautifulSoup(content, "html.parser")
            tables = soup.find_all("table")

            for table in tables:
                tbody = table.find("tbody") or table
                for row in tbody.find_all("tr"):
                    cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                    # Các cột theo cặp (Thuật ngữ, Định nghĩa)
                    for i in range(0, len(cells) - 1, 2):
                        raw_term = cells[i]
                        raw_defn = cells[i + 1]
                        if raw_term == "Thuật ngữ" or raw_defn == "Định nghĩa":
                            continue
                        parsed_pairs = parse_term_entry(raw_term, raw_defn)
                        for k, v in parsed_pairs:
                            abbr_dict[k] = v

    logger.info(f"Đã trích xuất thành công {len(abbr_dict)} thuật ngữ viết tắt.")

    if save_dict_path:
        with open(save_dict_path, "w", encoding="utf-8") as f:
            json.dump(abbr_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã lưu từ điển viết tắt vào: {save_dict_path}")

    return abbr_dict


def build_pattern(term: str) -> re.Pattern:
    r"""
    Tạo regex pattern để match từ viết tắt:
    - Không đi liền sau chữ cái/chữ số hoặc dấu mở ngoặc: (?<![\w\(])
    - Không đi liền trước chữ cái/chữ số hoặc dấu đóng ngoặc: (?![\w\)])
    - Không được đi liền trước khoảng trắng và dấu mở ngoặc (tránh lặp giải nghĩa): (?!\s*\()
    """
    escaped = re.escape(term)
    return re.compile(rf"(?<![\w\(]){escaped}(?![\w\)])(?!\s*\()")


def expand_abbreviations_in_text(
    text: str, abbr_dict: Dict[str, str], compiled_patterns: Optional[List[Tuple[str, str, re.Pattern]]] = None
) -> str:
    """
    Thay thế từ viết tắt trong văn bản bằng: Term (Meaning).
    Chỉ thay thế lần đầu tiên xuất hiện của mỗi từ viết tắt trong text.
    """
    if compiled_patterns is None:
        # Sắp xếp các từ theo độ dài giảm dần để ưu tiên từ dài trước
        sorted_terms = sorted(abbr_dict.items(), key=lambda x: -len(x[0]))
        compiled_patterns = [(term, defn, build_pattern(term)) for term, defn in sorted_terms]

    result = text
    for term, defn, pattern in compiled_patterns:
        # Thay thế duy nhất 1 lần (count=1)
        # Sử dụng lambda hoặc string format an toàn
        replacement = f"{term} ({defn})"
        result = pattern.sub(replacement, result, count=1)

    return result


def process_chunks(
    input_chunks_path: str = "output_chunks.json",
    output_chunks_path: str = "pp_output_chunks.json",
    abbr_dict: Optional[Dict[str, str]] = None,
) -> int:
    """
    Duyệt qua tất cả các chunk trong file input, làm giàu content và ghi sang file output.
    """
    if abbr_dict is None:
        abbr_dict = extract_abbreviations_from_gemini()

    # Pre-compile patterns theo thứ tự độ dài giảm dần
    sorted_terms = sorted(abbr_dict.items(), key=lambda x: -len(x[0]))
    compiled_patterns = [(term, defn, build_pattern(term)) for term, defn in sorted_terms]

    with open(input_chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    modified_count = 0
    total_chunks = len(chunks)
    processed_chunks = []

    for chunk in chunks:
        # Tạo bản sao chunk để giữ nguyên toàn bộ metadata
        chunk_copy = dict(chunk)
        chapter = chunk.get("chapter", "")

        # Bỏ qua không sửa đổi các chunk thuộc chính chương Danh mục thuật ngữ viết tắt
        if "thuật ngữ" in chapter.lower() and "viết tắt" in chapter.lower():
            processed_chunks.append(chunk_copy)
            continue

        original_content = chunk.get("content", "")
        enriched_content = expand_abbreviations_in_text(
            original_content, abbr_dict, compiled_patterns=compiled_patterns
        )

        if enriched_content != original_content:
            modified_count += 1
            chunk_copy["content"] = enriched_content

        processed_chunks.append(chunk_copy)

    # Ghi ra file output mới
    with open(output_chunks_path, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Hoàn thành xử lý: {modified_count}/{total_chunks} chunks đã được bổ sung giải nghĩa thuật ngữ.")
    logger.info(f"File kết quả được lưu tại: {output_chunks_path}")

    return modified_count


def main():
    parser = argparse.ArgumentParser(
        description="Post-process chunks by expanding abbreviations with their definitions."
    )
    parser.add_argument("--toc", default="toc.json", help="Đường dẫn file toc.json (mặc định: toc.json)")
    parser.add_argument(
        "--gemini",
        default="output_gemini.json",
        help="Đường dẫn file output_gemini.json (mặc định: output_gemini.json)",
    )
    parser.add_argument(
        "--input", default="output_chunks.json", help="Đường dẫn file chunks gốc (mặc định: output_chunks.json)"
    )
    parser.add_argument(
        "--output",
        default="pp_output_chunks.json",
        help="Đường dẫn file chunks kết quả (mặc định: pp_output_chunks.json)",
    )
    parser.add_argument(
        "--abbreviations",
        default="abbreviations.json",
        help="Đường dẫn lưu file từ điển viết tắt (mặc định: abbreviations.json)",
    )

    args = parser.parse_args()

    # 1. Trích xuất từ điển
    abbr_dict = extract_abbreviations_from_gemini(
        toc_path=args.toc, gemini_path=args.gemini, save_dict_path=args.abbreviations
    )

    # 2. Xử lý chunks
    process_chunks(input_chunks_path=args.input, output_chunks_path=args.output, abbr_dict=abbr_dict)


if __name__ == "__main__":
    main()
