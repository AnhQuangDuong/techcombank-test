"""
layout_tracker.py - Quản lý và theo dõi cây phân cấp đề mục Markdown qua các trang tài liệu.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


class LayoutTracker:
    """
    Theo dõi trạng thái các đề mục Markdown giữa các trang để sinh ngữ cảnh
    phân cấp truyền cho LLM ở trang tiếp theo.
    - Cấp 1 (#) cố định cho Chương (chapter)
    - Cấp 2 (##) cố định cho Ý chính (section)
    - Cấp 3+ (###, ####) dành cho các tiểu mục bóc tách từ thân trang.
    """

    def __init__(self) -> None:
        self.stack: List[Tuple[int, str]] = []
        self.current_chapter: Optional[str] = None
        self.current_section: Optional[str] = None
        self.current_h3: Optional[str] = None
        self.last_heading: Optional[Tuple[int, str]] = None

    def reset(self) -> None:
        """Xóa sạch toàn bộ trạng thái phân cấp tiểu mục."""
        self.stack.clear()
        self.current_h3 = None
        self.last_heading = None


    def set_context(self, chapter: Optional[str] = None, section: Optional[str] = None) -> None:
        """Cập nhật ngữ cảnh Chương và Ý chính. Reset stack nếu có sự thay đổi."""
        chapter_changed = bool(chapter and self.current_chapter and chapter != self.current_chapter)
        section_changed = bool(section and self.current_section and section != self.current_section)

        if chapter_changed or section_changed:
            self.reset()

        if chapter is not None:
            self.current_chapter = chapter
        if section is not None:
            self.current_section = section

    def _extract_headings(self, markdown_text: str) -> List[Tuple[int, str]]:
        """Bóc tách danh sách các dòng heading từ chuỗi Markdown, bỏ qua bảng HTML và code block."""
        headings: List[Tuple[int, str]] = []
        if not markdown_text:
            return headings

        # Tạm thời loại bỏ các khối <table>...</table> và code block ```...```
        cleaned_text = re.sub(r"<table[\s\S]*?</table>", "", markdown_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"```[\s\S]*?```", "", cleaned_text)

        for line in cleaned_text.splitlines():
            line = line.strip()
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                # Loại bỏ các định dạng bold bao quanh tiêu đề nếu có: **Tiêu đề** -> Tiêu đề
                title = re.sub(r"^\*\*|\*\*$", "", title).strip()
                if title:
                    # Nếu đang ở chế độ kiểm soát section, ép mức tối thiểu là H3
                    if self.current_section is not None and level < 3:
                        level = 3
                    headings.append((level, title))
        return headings

    def update_from_blocks(
        self,
        raw_blocks: list[dict],
        chapter: Optional[str] = None,
        section: Optional[str] = None,
    ) -> None:
        """
        Cập nhật ngăn xếp đề mục từ danh sách raw_blocks trả về từ Gemini OCR.
        Nếu chapter hoặc section thay đổi so với trang trước, tự động reset ngăn xếp tiểu mục.
        """
        self.set_context(chapter=chapter, section=section)

        all_new_headings: List[Tuple[int, str]] = []
        for block in raw_blocks:
            content = block.get("content", "")
            all_new_headings.extend(self._extract_headings(content))

        if not all_new_headings:
            # Giữ nguyên stack hiện tại nếu không có đề mục mới
            return

        for level, title in all_new_headings:
            # Pop tất cả đề mục có cấp sâu hơn hoặc ngang cấp
            while self.stack and self.stack[-1][0] >= level:
                self.stack.pop()

            self.stack.append((level, title))
            self.last_heading = (level, title)

    def get_prompt_context(self) -> str:
        """
        Sinh chuỗi ngữ cảnh truyền vào prompt của trang tiếp theo.
        - Nếu đã thiết lập section (chế độ phân cấp theo mục lục): sinh hướng dẫn chặt chẽ cấm '#' và '##'.
        - Nếu chưa thiết lập section (chế độ tự do): sinh ngữ cảnh đề mục thông thường.
        """
        if self.current_section is not None:
            lines = ["[NGỮ CẢNH PHÂN CẤP TÀI LIỆU]:"]
            if self.current_chapter:
                lines.append(f"- Chương (#): {self.current_chapter}")
            lines.append(f"- Ý chính (##): {self.current_section}")

            h3_items = [t for lvl, t in self.stack if lvl == 3]
            if h3_items:
                lines.append(f"- Tiểu mục cấp 3 (###) đang mở từ trang trước: {h3_items[-1]}")

            if self.stack:
                h_parts = []
                if self.current_chapter:
                    h_parts.append("H1 (#)")
                h_parts.append("H2 (##)")
                h_parts.extend([f"H{lvl}" for lvl, _ in self.stack])
                lines.append(f"- Cây phân cấp hiện tại: {' > '.join(h_parts)}")

                last_lvl, last_txt = self.last_heading if self.last_heading else self.stack[-1]
                lines.append(f"- Đề mục gần nhất vừa hoàn thành: {'#' * last_lvl} {last_txt}")

            lines.extend([
                "* HƯỚNG DẪN ĐỀ MỤC BẮT BUỘC:",
                "  + TUYỆT ĐỐI KHÔNG dùng '#' hoặc '##' (đã dành riêng cho Chương và Ý chính).",
                "  + Nếu nội dung trang này là phần viết tiếp của tiểu mục trước: tiếp tục trình bày nội dung, không lặp lại đề mục '###'.",
                "  + Nếu xuất hiện ý nhỏ mới trong trang: bắt buộc dùng '###' (hoặc '####' cho phân cấp sâu hơn).",
            ])
            return "\n".join(lines)

        # Chế độ tự do (khi chưa cấu hình section từ TOC)
        if not self.stack:
            return ""

        h1_item = next((item for item in self.stack if item[0] == 1), None)
        h1_text = f"# {h1_item[1]}" if h1_item else "Không có"
        hierarchy_str = " > ".join(f"H{level}" for level, _ in self.stack)
        last_lvl, last_txt = self.last_heading if self.last_heading else self.stack[-1]
        last_heading_str = f"{'#' * last_lvl} {last_txt}"

        lines = [
            "[NGỮ CẢNH ĐỀ MỤC TỪ CÁC TRANG TRƯỚC]:",
            f"- Đề mục cấp 1 (#) đang mở: {h1_text}",
            f"- Cây phân cấp hiện tại: {hierarchy_str}",
            f"- Đề mục gần nhất vừa hoàn thành: {last_heading_str}",
            "* Hướng dẫn kế thừa phân cấp:",
            "  + Nếu trang hiện tại là nội dung tiếp nối, hãy gán cấp độ đề mục phù hợp tương ứng theo phân cấp trên.",
            "  + Nếu trang hiện tại bắt đầu một chủ đề lớn mới hoặc chương mới, hãy sử dụng đề mục # cô đọng độc lập tương ứng.",
        ]
        return "\n".join(lines)


    def extract_h3_headings(self, markdown_text: str) -> List[str]:
        """Bóc tách các tiêu đề chỉ ở cấp ### (bỏ qua #### trở đi, bỏ qua table và code block)."""
        if not markdown_text:
            return []

        cleaned = re.sub(r"<table[\s\S]*?</table>", "", markdown_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)

        h3_list = []
        for line in cleaned.splitlines():
            line = line.strip()
            m = re.match(r"^###\s+([^#\s].*)$", line)
            if m:
                title = m.group(1).strip()
                title = re.sub(r"^\*\*|\*\*$", "", title).strip()
                if title:
                    h3_list.append(title)
        return h3_list

    def update_for_logical_page(
        self,
        content: str,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[str]:
        """
        Cập nhật trạng thái H3 độc lập cho từng trang logic:
        - Nếu trang có tiêu đề ###: lấy tiêu đề ### cuối cùng làm current_h3.
        - Nếu trang không có tiêu đề ###: kế thừa current_h3 từ trang trước (nếu cùng section).
        - Bỏ qua toàn bộ các đề mục từ #### trở đi.
        - Trả về list chứa tiêu đề ### hiện tại: [current_h3] hoặc [].
        """
        self.set_context(chapter=chapter, section=section)
        page_h3s = self.extract_h3_headings(content)
        if page_h3s:
            self.current_h3 = page_h3s[-1]
            self.last_heading = (3, self.current_h3)
            while self.stack and self.stack[-1][0] >= 3:
                self.stack.pop()
            self.stack.append((3, self.current_h3))

        return [self.current_h3] if self.current_h3 else []

    def get_active_subsections(self) -> list[str]:
        """Trả về danh sách tiêu đề cấp ### đang có hiệu lực (chỉ xét ###, bỏ qua #### trở đi)."""
        if self.current_h3:
            return [self.current_h3]
        return [title for lvl, title in self.stack if lvl == 3]


