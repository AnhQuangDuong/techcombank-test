"""
prompts.py - Định nghĩa Prompt chuẩn cho Gemini OCR trích xuất trang tài liệu dạng đôi (spread).
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# ==========================================
# Cấu hình Pydantic & Prompt cho Trang Mục lục (TOC)
# ==========================================

class TOCSubSection(BaseModel):
    title: str = Field(description="Tên ý quan trọng / mục con")
    start_page: int = Field(description="Số trang bắt đầu in trên mục lục")


class TOCChapter(BaseModel):
    chapter_title: str = Field(description="Tên chương hoặc tên mục lớn (ví dụ: 'CHƯƠNG 01: ...', 'Chúng tôi là ai', 'Danh mục thuật ngữ viết tắt')")
    start_page: Optional[int] = Field(default=None, description="Số trang bắt đầu của chương nếu có")
    sections: List[TOCSubSection] = Field(default_factory=list, description="Danh sách các ý chính / mục con trực thuộc chương đó")


class TOCExtractionOutput(BaseModel):
    chapters: List[TOCChapter] = Field(default_factory=list, description="Danh sách các chương hoặc phần trong mục lục theo đúng thứ tự xuất hiện")


TOC_SYSTEM_PROMPT = """\
Đọc hình ảnh trang mục lục của tài liệu và trích xuất cấu trúc mục lục sang định dạng JSON:
- Giữ nguyên tên gốc của từng chương hoặc từng phần (ví dụ: 'Chúng tôi là ai', 'CHƯƠNG 01: ...', 'CHƯƠNG 02: ...', 'Danh mục thuật ngữ viết tắt', 'Phụ lục GRI'). Không tự ý đổi tên hay nhóm lại.
- Với mỗi chương, trích xuất tất cả các ý chính/mục con trực thuộc cùng số trang bắt đầu in trên mục lục.
- Nếu một phần độc lập không có mục con (ví dụ: 'Chúng tôi là ai', 'CHƯƠNG 06: ...', 'Danh mục thuật ngữ viết tắt'), ghi nhận tên phần đó và số trang bắt đầu tương ứng.
- Đảm bảo giữ đúng thứ tự từ trên xuống dưới theo trang tài liệu.
"""

TOC_USER_PROMPT = "Hãy trích xuất toàn bộ mục lục trong hình ảnh thành danh sách các chương/phần và số trang bắt đầu theo đúng cấu trúc JSON."


# ==========================================
# Cấu hình Prompt cho Các Trang Nội dung (Content Pages)
# ==========================================

SYSTEM_PROMPT = """\
Đọc hình ảnh trang đôi (spread) của tài liệu PDF và trích xuất nội dung sang định dạng JSON:
{
  "blocks": [
    {
      "location": "left" | "right" | "all",
      "content": "string"
    }
  ]
}

Quy tắc phân định vị trí (location):
1. "location":
   - "left": Nội dung nằm ở nửa trang bên trái.
   - "right": Nội dung nằm ở nửa trang bên phải.
   - "all": CHỈ dùng cho nội dung thực sự trải dài hoặc chiếm trọn cả 2 trang (ví dụ: hình ảnh panorama khổ lớn trải ngang cả 2 trang, hoặc 1 bảng biểu duy nhất kéo dài liền mạch từ mép trái sang mép phải qua nếp gấp giữa).
   - QUAN TRỌNG: Nếu trang bên trái và trang bên phải có các phần nội dung riêng biệt hoặc hai bảng riêng biệt (dù cùng chủ đề hay là bảng tiếp theo), BẮT BUỘC phải tách thành các block riêng với location "left" và "right", TUYỆT ĐỐI KHÔNG gộp chung thành "all".

Quy tắc định dạng nội dung (content):
2. "content":
   - Trình bày toàn bộ nội dung theo ĐÚNG THỨ TỰ TRÌNH BÀY TỰ NHIÊN từ trên xuống dưới, theo từng nửa trang (left / right).
   - Văn bản ở dạng Markdown chuẩn:
     + QUY TẮC PHÂN BIỆT ĐỀ MỤC (###) VÀ CHỮ IN ĐẬM (**...**):
       * CẤM DÙNG # VÀ ##: Dấu `#` (H1) và `##` (H2) đã được hệ thống cố định cho Chương và Ý chính. Nội dung trang BẮT BUỘC chỉ dùng từ cấp `###` trở xuống.
       * Dấu `###` CHỈ DÙNG cho tiêu đề mục cấu trúc (ngắn gọn, tên đề tài, hoặc có số thứ tự như '1.', '23.2.'). Tuyệt đối KHÔNG chứa thẻ HTML (như <span>, <small>, <br>) trong đề mục.
       * MỌI CHỮ IN ĐẬM KHÁC (câu văn mở đầu đoạn, câu trích dẫn/slogan, tên người/chức danh dưới ảnh hoặc cuối thư) BẮT BUỘC dùng định dạng in đậm `**...**`, TUYỆT ĐỐI KHÔNG gán dấu `###`.
     + Kế thừa phân cấp: Nếu nhận được chỉ dẫn ngữ cảnh đề mục từ trang trước trong yêu cầu, BẮT BUỘC tuân thủ phân cấp kế thừa khi nội dung trang là phần tiếp nối.
   - Bảng biểu (Tables): BẮT BUỘC 100% trình bày ở dạng HTML với thẻ <table>...</table> (bao gồm <thead>, <tbody>, <tr>, <th>, <td>). TUYỆT ĐỐI KHÔNG dùng bảng Markdown pipe (| ... |).
     + Trích xuất đầy đủ 100% tất cả các dòng và các cột, không tóm tắt, không bỏ sót dòng nào.
     + Tổng số cột ở Header (tính cả colspan) BẮT BUỘC phải khớp chính xác 100% với số lượng thẻ <td> trong mỗi dòng dữ liệu. Nhóm cột cha có bao nhiêu cột con bên dưới thì BẮT BUỘC đặt colspan bằng đúng bấy nhiêu. Tuyệt đối không để thừa/thiếu làm lệch cột.
      + ĐỘ CHÍNH XÁC VÀ CẤU TRÚC BẢNG BIỂU TỪ HÌNH ẢNH:
        * Dùng hình ảnh để xác định chính xác bố cục bảng biểu, cột, dòng, tiêu đề, colspan và rowspan.
        * Nhìn kỹ từng con số, dấu phân cách hàng nghìn (dấu chấm), dấu thập phân (dấu phẩy), số âm trong ngoặc đơn, đơn vị tiền tệ và các ký hiệu ghi chú footnote.
        * Đảm bảo tính nhất quán về mặt toán học giữa các số liệu trong bảng và khớp hoàn toàn với tài liệu.
   - Biểu đồ, đồ thị (Charts / Graphs): BẮT BUỘC trích xuất ĐẦY ĐỦ 100% dữ liệu sang dạng văn bản, tuyệt đối không tóm tắt chung chung hay bỏ qua số liệu:
     + Trích xuất toàn bộ các điểm dữ liệu theo từng mốc/danh mục: gồm giá trị số, đơn vị, cùng mọi nhãn phụ đi kèm (thứ hạng/Top, tỷ lệ tăng trưởng, huy hiệu nhãn).
     + Trình bày rõ ràng: Tiêu đề biểu đồ, chú giải các chuỗi dữ liệu (nếu có nhiều chuỗi) và danh sách chi tiết từng mốc (`Mốc: Giá trị [Nhãn/Thứ hạng kèm theo]`) hoặc bảng số liệu.
   - Danh sách, gạch đầu dòng: Giữ nguyên định dạng bullet (- hoặc •) hoặc số thứ tự (1., 2.).
   - Nếu một nửa trang không có nội dung, trả về chuỗi rỗng "".
   - Bỏ qua dòng header/footer cố định ở mép trên cùng hoặc mép dưới cùng (số trang, tên chương, tên báo cáo) và hình ảnh chỉ mang tính trang trí.
"""

USER_PROMPT = "Hãy đọc toàn bộ hình ảnh trang tài liệu và trích xuất tất cả các khối nội dung sang định dạng JSON theo đúng quy tắc trên."

TABLE_INSTRUCTION_PROMPT = """\
--- CHỈ DẪN TRÍCH XUẤT BẢNG BIỂU TỪ HÌNH ẢNH ---
Trang này có chứa bảng biểu số liệu. Hãy tập trung quan sát kỹ hình ảnh để trích xuất bảng chính xác 100%:
1. Đọc đúng vị trí: Bảng nằm ở nửa trang nào thì đặt trong block có location tương ứng ("left" hoặc "right"). Nếu có nhiều bảng tách rời trên cùng một trang, giữ nguyên từng bảng HTML riêng biệt theo thứ tự trên ảnh, tuyệt đối không tự ý gộp bảng.
2. Cấu trúc HTML hoàn chỉnh: Trình bày dạng <table>...</table> với <thead> và <tbody>. Số cột ở <th> (tính cả colspan) phải khớp chính xác với số <td> ở mỗi dòng.
3. Độ chính xác số liệu: Nhìn kỹ từng con số, dấu phân cách thập phân và hàng nghìn, số âm trong ngoặc đơn (nếu có), đơn vị tiền tệ (ví dụ: Tỷ đồng, Triệu đồng) và các ký hiệu ghi chú footnote (1, 2, *).
4. Thứ tự đọc: Đặt bảng vào đúng thứ tự tự nhiên so với các đoạn văn bản xung quanh trên ảnh.
--- HẾT CHỈ DẪN BẢNG ---"""
