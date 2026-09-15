# Submission

## How to run
- **Clean machine**:
  ```bash
  # Prerequisites: Python >= 3.10, uv (hoặc pip)
  git clone <repo-url>
  cd techcombank

  # Cài đặt toàn bộ dependencies tự động qua uv
  uv sync

  # Thiết lập cấu hình biến môi trường
  cp .env.example .env # và điền GEMINI_API_KEY vào .env

  # Chạy chatbot giao diện dòng lệnh tương tác (sử dụng index và vector DB đã đóng gói sẵn)
  uv run python chatbot.py
  ```
- **Different key or local model**:
  - **Dùng API key hoặc model khác**: Cấu hình tập trung trong file `.env`:
    - Điền API key mới: `GEMINI_API_KEY=your_api_key_here`
    - Đổi mô hình LLM generation khác của Google: `GEMINI_MODEL=gemini-2.5-flash` (hoặc `gemini-1.5-pro`, mặc định: `gemini-3.5-flash-lite`).
    - *Khuyến nghị về Embedding Model*: Khuyến khích sử dụng model miễn phí `GEMINI_EMBEDDING_MODEL=gemini-embedding-001` từ Google AI Studio. Toàn bộ chỉ mục vector và metadata đóng gói sẵn (`qdrant_storage/` và `output_indexs.json`) đã được xây dựng chuẩn xác trên model này (vector 3072 chiều).
  - **Dùng Local Model hoặc Provider khác (OpenAI / Ollama / vLLM)**:
    - *LLM Generation (Answer Generator & Query Rewriter)*: Có thể trỏ sang endpoint local bằng cách thay thế client trong `chatbot.py` bằng OpenAI-compatible client (ví dụ `openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")`). Cách này vẫn tái sử dụng trọn vẹn 100% kho vector Qdrant đóng gói sẵn.
    - *Local Embedding Model*: Nếu đổi sang mô hình embedding cục bộ khác (như `bge-m3`), **bắt buộc phải re-index lại toàn bộ tài liệu** (`uv run python index_chunks.py`) do khác biệt về số chiều và không gian vector nhúng.
- **Batch run**:
  - **Sinh câu trả lời hàng loạt tự động** (trả lời tuần tự 10 câu hỏi độc lập trong `sample_questions.json` qua `chatbot.py`, tự động reset session giữa các câu để đảm bảo tính độc lập tuyệt đối, đo thời gian phản hồi `⏱️`, và nghỉ 4 giây giữa các câu để không vượt Rate Limit 15 RPM của Gemini Free Tier):
    ```bash
    uv run python chatbot.py --batch sample_questions.json
    ```
  - **Đánh giá hiệu năng truy xuất (Retrieval Evaluation)** qua `eval.py`:
    *(Lưu ý: Script này chỉ thực hiện đánh giá giai đoạn Retrieval — tính toán Recall@k, nDCG@k, MAP@k bằng cách đối chiếu số trang của các chunk tìm được với `gold_printed_pages`, **không** gọi LLM để sinh câu trả lời tự động)*:
    ```bash
    uv run python eval.py --questions sample_questions.json --output eval_results.json
    ```
- **Environment variables**:
  - `GEMINI_API_KEY`: Khóa API của Google Gemini dùng cho cả Embedding (`RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`) và Text Generation (Query Rewriting, Grounded Answer Generation).
  - `GEMINI_MODEL`: Tên mô hình LLM chính dùng để sinh câu trả lời và viết lại câu hỏi hội thoại (mặc định: `gemini-3.5-flash-lite`).
  - `GEMINI_EMBEDDING_MODEL`: Tên mô hình tạo vector nhúng ngữ nghĩa cho tài liệu và câu hỏi (mặc định: `gemini-embedding-001`).
- **Shipped index**:
  - Vector Database & Metadata đã được build và lưu trữ sẵn tại:
    - Metadata & Dense Embeddings: [`output_indexs.json`](file:///home/quanganh/techcombank/output_indexs.json) (52.6 MB, gồm 842 chunks hoàn chỉnh với vector 3072 chiều).
    - Local Vector Database: thư mục [`qdrant_storage/`](file:///home/quanganh/techcombank/qdrant_storage) (collection `techcombank_chunks` sẵn sàng truy vấn không cần build lại).
    - Chunks đã qua hậu xử lý thuật ngữ: [`pp_output_chunks.json`](file:///home/quanganh/techcombank/pp_output_chunks.json).
    - Từ điển thuật ngữ viết tắt: [`abbreviations.json`](file:///home/quanganh/techcombank/abbreviations.json).

---

## Depth track(s) chosen

Dự án quyết định tập trung chuyên sâu duy nhất vào track **Document intelligence**.

### **Document intelligence**
- **Lý do lựa chọn**:
  - Báo cáo thường niên Techcombank 2025 là tài liệu tài chính đồ họa phức tạp với định dạng trang đôi (spread - 2 trang logic ghép trên 1 trang PDF vật lý), nếp gấp ở giữa cắt qua các bảng Báo cáo Tài chính Kiểm toán nhiều cột và biểu đồ chỉ số.
  - Trong hệ thống RAG tài chính, nếu tầng Ingestion và Chunking làm hỏng cấu trúc tài liệu (cắt đôi dòng bảng, làm mất tiêu đề cột, mất số trang in thực tế, hoặc chia vụn số liệu), thì bất kỳ mô hình Retrieval hay LLM nào ở phía sau cũng sẽ bị *"Garbage In, Garbage Out"*. Do đó, dự án dồn toàn lực giải quyết bài toán thị giác máy tính kết hợp Layout State Machine để khôi phục cấu trúc tài liệu chuẩn xác 100%.

- **Những giải pháp kỹ thuật chuyên sâu đã triển khai**:
  1. **Seam Recovery & Layout-aware Ingestion ([extract_pdf_gemini.py](file:///home/quanganh/techcombank/extract_pdf_gemini.py))**:
     - Thay vì cắt đôi ảnh PDF bằng tọa độ hình học thuần túy (vốn sẽ xé toạc các bảng biểu và dòng chữ nằm đè lên nếp gấp giữa hai trang), hệ thống giữ nguyên trang đôi vật lý và vẽ rãnh đệm mảnh (0.5 pt center seam buffer) ở nếp gấp để định hướng thị giác.
     - Gemini Vision nhận diện và gắn nhãn vị trí trực quan cho từng khối nội dung:
       - `left`: Thuộc nửa trang trái $\rightarrow$ tự động ánh xạ vào trang in logic bên trái (ví dụ: `[tr 18]`).
       - `right`: Thuộc nửa trang phải $\rightarrow$ tự động ánh xạ vào trang in logic bên phải (ví dụ: `[tr 19]`).
       - `all`: Bảng biểu tài chính nhiều cột hoặc biểu đồ lớn nằm tràn qua nếp gấp $\rightarrow$ **được giữ nguyên vẹn 100% không cắt rời**, đồng thời gán nhãn thuộc cả 2 trang (ví dụ: `[tr 18, 19]`).
     - Nhờ vậy, hệ thống khôi phục hoàn chỉnh 185 trang PDF đôi thành đúng **370 trang in logic thực tế** của báo cáo, đảm bảo chatbot luôn trích dẫn số trang `[tr x, y]` trùng khớp tuyệt đối với bản in sách giấy.
  2. **Bảo toàn nguyên vẹn bảng biểu (Atomic Table Protection - (chunker.py))**:
     - Xây dựng cơ chế `extract_atomic_blocks` coi mỗi thẻ `<table>...</table>` của báo cáo tài chính kiểm toán là một khối nguyên tử không thể chia cắt.
     - Dù bảng tài chính dài vượt quá ngưỡng kích thước chunk thông thường (2000+ ký tự), bảng vẫn được giữ nguyên vẹn cùng toàn bộ header cột, tuyệt đối không bị cắt ngang giữa các dòng `<tr>`/`<td>`.
  3. **Đọc số liệu biểu đồ & Lọc bỏ hình ảnh trang trí**:
     - *Biểu đồ, đồ thị*: Chỉ đạo Gemini Vision nhận diện các biểu đồ tài chính để trích xuất tiêu đề, chú giải và các mốc số liệu sang dạng danh sách hoặc bảng.
     - *Lọc ảnh trang trí*: Chủ động bỏ qua các hình ảnh minh họa thương hiệu, ảnh stock chụp nhân viên/tòa nhà để tránh sinh ra dữ liệu rác làm nhiễu ngữ cảnh.
  4. **Layout State Machine & Breadcrumb Hierarchy ([layout_tracker.py](file:///home/quanganh/techcombank/layout_tracker.py))**:
     - Xây dựng State Machine theo dõi ngăn xếp tiêu đề (`#`, `##`, `###`) liên trang.
     - Giải quyết triệt để tình trạng "chunk mồ côi" (orphan chunks): khi một trang nối tiếp chỉ chứa bảng số liệu mà không lặp lại heading, hệ thống tự động kế thừa heading của trang trước, đảm bảo mỗi chunk luôn có đầy đủ đường dẫn ngữ cảnh phân cấp (Chương > Mục > Tiêu đề).

---

## Architecture

Toàn bộ quá trình từ file PDF thô đến câu trả lời hoàn chỉnh diễn ra qua 5 giai đoạn:

```
+--------------------------------------------------------------------------------------------------+
| 1. EXTRACTION & SEAM RECOVERY (extract_pdf_gemini.py + layout_tracker.py + prompts.py)           |
| - PyMuPDF render trang đôi (spread) với Center Seam Buffer mỏng (tránh mất chữ ở nếp gấp).       |
| - Gemini OCR trích xuất JSON blocks [left, right, all] + Table HTML + Biểu đồ số liệu đầy đủ.    |
| - LayoutTracker theo dõi ngăn xếp tiêu đề (#, ##, ###) liên trang, truyền context cho trang sau.  |
| - Ánh xạ về đúng trang in logic thực tế của báo cáo.                                             |
| ➔ Outputs: toc.json (cây mục lục), output_gemini.json (370 trang logic), output_gemini.md.       |
+--------------------------------------------------------------------------------------------------+
                                                 │
                                                 ▼
+--------------------------------------------------------------------------------------------------+
| 2. ATOMIC CHUNKING & ENRICHMENT (chunker.py + chunk_post_processor.py)                           |
| - Atomic Block Protection: Giữ nguyên vẹn 100% khối <table>...</table>, không bao giờ cắt vụn.   |
| - State Machine merge heading rỗng: Tránh tạo chunk mồ côi, tạo đường dẫn phân cấp đầy đủ.       |
| - Trích xuất phụ lục viết tắt ngân hàng thành từ điển tra cứu.                                   |
| - Bung từ viết tắt lần đầu trong mỗi chunk (Abbreviation Expansion) & chuẩn hóa văn bản.          |
| ➔ Outputs: output_chunks.json (chunks thô), abbreviations.json (từ điển),                         |
|            pp_output_chunks.json (842 enriched chunks đã làm giàu).                              |
+--------------------------------------------------------------------------------------------------+
                                                 │
                                                 ▼
+--------------------------------------------------------------------------------------------------+
| 3. CONTEXTUAL INDEXING (index_chunks.py)                                                         |
| - Tạo Contextual Text: "Chương: ... | Mục: ... | Tiêu đề: ...\n\n{content}".                     |
| - Gemini Embedding API (task_type=RETRIEVAL_DOCUMENT) theo batch 50 + Adaptive Split.            |
| - Đóng gói vector 3072 chiều cùng metadata phân cấp vào JSON và lưu trữ vào Qdrant Local.        |
| ➔ Outputs: output_indexs.json (metadata + dense vectors), qdrant_storage/ (Vector DB).           |
+--------------------------------------------------------------------------------------------------+
                                                 │
                                                 ▼
+--------------------------------------------------------------------------------------------------+
| 4. DOMAIN-AWARE HYBRID RETRIEVAL (search_qdrant.py)                                              |
| - BM25Okapi với Custom Regex Tokenizer (bảo toàn tiếng Việt có dấu, số thập phân 53,4; tỷ lệ %)  |
| - Gemini Embedding API (task_type=RETRIEVAL_QUERY) + Qdrant Vector Cosine Search.                |
| - Min-Max Scaling + Score Fusion: Final = 0.4 * Lexical + 0.6 * Semantic.                        |
| ➔ Output: Top K chunks liên quan nhất kèm điểm số tổng hợp và phân rã lexical/semantic.         |
+--------------------------------------------------------------------------------------------------+
                                                 │
                                                 ▼
+--------------------------------------------------------------------------------------------------+
| 5. CONVERSATIONAL RAG & STRICT GROUNDING (chatbot.py + eval.py)                                  |
| - Multi-turn Query Rewriter: Viết lại câu hỏi nối tiếp thành truy vấn độc lập (đầy đủ thực thể).  |
| - Grounded Generator: Ép System Instruction nghiêm ngặt, bắt buộc trích dẫn [tr x, y].          |
| - Safe Refusal: Tự động trả về câu từ chối chuẩn hóa nếu tài liệu không đủ thông tin.            |
| ➔ Outputs: Câu trả lời có nguồn trích dẫn; eval_results.json (khi chạy kịch bản đánh giá).      |
+--------------------------------------------------------------------------------------------------+
```

---

## What I tried that did not work
1. **Cắt đôi trang PDF vật lý bằng tọa độ hình học thuần túy (`width / 2`)**:
   - *Vấn đề*: Trong báo cáo thường niên, nhiều bảng biểu số liệu (như Báo cáo Kết quả Hoạt động kinh doanh hợp nhất) và ảnh chụp đồ họa trải ngang (panorama) nằm đè trực tiếp qua nếp gấp giữa hai trang. Việc cắt cứng bằng tọa độ làm xé toạc các dòng dữ liệu của bảng, khiến OCR nhận diện sai hoàn toàn các con số ở cột giữa.
   - *Giải pháp thay thế*: Giữ nguyên trang đôi vật lý, chỉ vẽ một vệt ngăn cách mảnh (0.5 pt) ở nếp gấp, yêu cầu Gemini OCR phân loại các block theo `left`, `right`, hoặc `all`. Sau đó mới dùng logic ánh xạ sang trang in logic.
2. **Cắt chunk dựa trên độ dài ký tự/token thông thường (Recursive Character Splitter)**:
   - *Vấn đề*: Bộ chia văn bản thông thường cắt ngang giữa các hàng của bảng HTML hoặc Markdown pipe, phá vỡ thẻ `<tr>`/`<td>`, làm mất hoàn toàn ngữ cảnh tiêu đề cột của bảng số liệu tài chính.
   - *Giải pháp thay thế*: Xây dựng `extract_atomic_blocks()` trong `chunker.py` coi mỗi thẻ `<table>...</table>` là một khối nguyên tử (atomic block) không thể chia cắt. Bảng dài hơn 2000 ký tự vẫn được giữ nguyên khối.
3. **Chỉ sử dụng Semantic Search (Vector Embedding thuần túy)**:
   - *Vấn đề*: Vector dense embedding thường hiểu tốt ý nghĩa ngữ nghĩa nhưng gặp khó khăn khi tìm kiếm chính xác các mã viết tắt ngân hàng cụ thể (ví dụ: "RBG", "CASA") hoặc các số liệu phần trăm đặc thù ("16,5%"), dẫn đến các câu hỏi thuộc category `terminology` bị tụt hạng ngoài Top 10.
   - *Giải pháp thay thế*: Triển khai mô hình Hybrid Search kết hợp BM25 (với bộ tokenizer tiếng Việt tùy biến giữ lại dấu thập phân, dấu phẩy, `%`, mã acronym) và Qdrant vector search với trọng số `0.4 * BM25 + 0.6 * Vector`.

---

## Evaluation
- **Method**:
  - Đánh giá tự động hiệu năng truy xuất (Retrieval) bằng script [`eval.py`](eval.py) trên tập 10 câu hỏi chuẩn (`sample_questions.json`) bao gồm đầy đủ các khía cạnh: hồ sơ doanh nghiệp (`company_profile`), số liệu tài chính cốt lõi (`key_figures`), phân khúc khách hàng (`segment_performance`), thuật ngữ viết tắt (`terminology`), và câu hỏi không thể trả lời (`unanswerable`).
  - Đối chiếu số trang logic của các chunk được truy xuất với `gold_printed_pages` thực tế (đánh giá khả năng tìm đúng tài liệu nguồn, không bao gồm bước sinh câu trả lời bằng LLM).
  - Sử dụng các thước đo chuẩn Information Retrieval (IR): **Recall@k**, **nDCG@k** và **MAP@k** với $k \in \{1, 3, 5, 10, 20\}$.
- **Results on the 10 published questions**:
  *(Đo lường trên 9 câu hỏi có đáp án trong tài liệu; 1 câu hỏi `unanswerable` được hệ thống từ chối thành công theo đúng yêu cầu)*

  | Metric | @1 | @3 | @5 | @10 | @20 |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Recall** | **0.3333** | **0.4444** | **0.7778** | **0.8889** | **0.8889** |
  | **nDCG** | 0.3333 | 0.3445 | 0.4003 | 0.3901 | 0.3856 |
  | **MAP** | 0.3333 | 0.3148 | 0.3294 | 0.3177 | 0.3152 |

  - Chi tiết từng câu hỏi:
    - **Top-1 Exact Hits**: `sq-03` (Tổng tài sản - Trang 5, Hit #1), `sq-04` (CASA - Trang 5, Hit #1), `sq-06` (Thu nhập hoạt động & CAGR - Trang 5, Hit #1).
    - **Top-5 Hits**: `sq-01` (Mạng lưới chi nhánh - Trang 4, Hit #5), `sq-02` (Xếp hạng tín nhiệm - Trang 4, Hit #5), `sq-05` (Tỷ lệ nợ xấu - Trang 5, Hit #4), `sq-07` (Dư nợ Bán lẻ - Trang 59, Hit #4).
    - **Terminology Hits**: `sq-08` (RBG - Trang 387, Hit #8), `sq-09` (CASA - Trang 386, Hit #7).
    - **Unanswerable Case (`sq-10`)**: Hệ thống nhận diện không có tài liệu liên quan phù hợp và kích hoạt cơ chế Safe Refusal: *"Tôi không đủ thông tin để trả lời câu hỏi này."*

---

## Cost and latency
- **Ingestion**:
  - *Wall-clock*: ~18 phút (chạy toàn bộ 370 trang logic từ PDF gốc sang markdown, chunking, hậu xử lý thuật ngữ và embedding toàn bộ 842 chunks).
  - *Cost*: **$0.00** Miễn phí do sử dụng gói Free Tier của Google AI Studio.
- **Per query**:
  - *Độ trễ trung vị (P50 Latency)*: **~2.4 giây** (Embed câu hỏi: ~350ms; Qdrant + BM25 Hybrid Search: ~15ms; Gemini sinh câu trả lời: ~1.5s; đo lường thực tế trên tập 10 câu hỏi chuẩn).
  - *Cost*: **$0.00** Miễn phí qua Google AI Studio Free Tier.
- **Models used**:
  - *Generation & Query Rewriting*: `gemini-3.5-flash-lite`.
  - *Embedding*: `gemini-embedding-001` (vector dimension: 3072).

---

## What I deliberately did not build
- **Mô hình Cross-Encoder Reranker (như Cohere Rerank hay BGE-Reranker-Large)**:
  - *Lý do*: Việc triển khai thêm một model neural reranker nặng đòi hỏi tài nguyên GPU riêng biệt hoặc chi phí API bên thứ 3. Do giới hạn tài nguyên của project nên việc này chưa được triển khai.

---

## With 10x time and budget
1. **Cây mục lục phân cấp & Định tuyến lọc metadata thông minh (Hierarchical Layout Tree & LLM-Guided Pre-Retrieval Routing)**:
   - *Bối cảnh & Điểm đau*: Qdrant hiện đã hỗ trợ metadata filtering theo trường, nhưng hệ thống đang tìm kiếm phẳng trên toàn bộ 842 chunks của tài liệu. Điều này dễ dẫn đến việc các chỉ số tài chính ở những chương giới thiệu chung làm loãng hoặc tranh chấp vị trí với số liệu kiểm toán chi tiết ở chương Báo cáo Tài chính.
   - *Quy trình thực hiện*:
     - **Giai đoạn Ingestion (Offline)**: Tận dụng cấu trúc phân cấp tự nhiên của báo cáo để dựng cây mục lục hoàn chỉnh: **Chương $\rightarrow$ Đề mục $\rightarrow$ Tiểu mục**. Chạy một pipeline LLM duyệt qua các đoạn đã parse của từng trang logic để sinh bản tóm tắt ngắn gọn (executive summary) cho từng Tiểu mục.
     - **Giai đoạn Suy luận (Online - Pre-retrieval Routing)**: Ngay sau bước chuẩn hóa truy vấn (Query Rewriter) và trước khi thực hiện tìm kiếm, câu hỏi sẽ được chuyển qua một LLM Router nhẹ kèm nội dung cây Layout và tóm tắt tiểu mục. LLM sẽ dự đoán danh sách metadata filter phù hợp (ví dụ: chỉ giới hạn trong `Chapter 05: Báo cáo tài chính` hoặc `Section: Báo cáo của Ban Điều hành`). Bộ lọc này được nạp trực tiếp vào Qdrant/Hybrid Search để thu hẹp không gian tìm kiếm, loại bỏ hoàn toàn nhiễu từ các chương khác và đảm bảo 7 chunks context được nạp vào bước sinh câu trả lời có độ chính xác cao nhất.
     - **Khả năng mở rộng đa tài liệu (Multi-Document Scaling)**: Khi hệ thống mở rộng nạp nhiều tài liệu (báo cáo thường niên qua các năm 2020–2025, Báo cáo ESG, Báo cáo Quản trị độc lập), hệ thống Ingestion sẽ tổng hợp từ layout thành tóm tắt cấp tài liệu. LLM Router cấp 1 sẽ lọc ra các tài liệu có liên quan nhất, sau đó Router cấp 2 tiếp tục lọc sâu vào từng phân mục cụ thể của tài liệu đó trước khi truy xuất.
2. **Kiến trúc Two-Stage Retrieval với Neural Re-ranking (High-Recall Candidate Retrieval & Cross-Encoder Re-ranking)**:
   - *Bối cảnh & Điểm đau*: Hiện tại hệ thống đang dùng Single-Stage Retrieval và chỉ lấy trực tiếp top K nhỏ (~7 chunks) nạp vào LLM để tối ưu latency và chi phí. Tuy nhiên, đối với các câu hỏi phức tạp hoặc bảng biểu tài chính nằm rải rác, việc chỉ lấy top 7 ở bước đầu dễ dẫn đến hiện tượng bỏ sót thông tin quan trọng (Recall chưa tối ưu) nếu biểu diễn vector hoặc từ khóa ban đầu bị lệch ngữ cảnh.
   - *Quy trình thực hiện*:
     - **Stage 1 - Mở rộng Retrieval Window (High-Recall Filtering)**: Tăng $k$ ứng viên trả về từ hybrid search (Dense + Sparse) lên quy mô lớn (khoảng 100 chunks liên quan nhất). Giai đoạn này đóng vai trò như một lưới quét rộng nhằm tối đa hóa tỷ lệ thu hồi (Recall), đảm bảo bao quát toàn bộ các đoạn văn bản, bảng số liệu hay chú thích kiểm toán tiềm năng.
     - **Stage 2 - Neural Re-ranking (High-Precision Scoring)**: Triển khai mô hình Re-ranker chuyên dụng (Cross-Encoder như `BGE-Reranker-Large`, `Cohere Rerank 3` hoặc mô hình late-interaction như `ColBERT`). Mô hình này tính toán tương tác chéo sâu (cross-attention) giữa query và từng chunk trong 100 ứng viên để chấm điểm lại mức độ liên quan ngữ nghĩa thực tế.
     - **Top-10 Selection & Context Packing**: Chọn lọc ra **Top 10** chunks có điểm re-ranking cao nhất, áp dụng kỹ thuật sắp xếp vị trí tối ưu ngữ cảnh (giảm thiểu hiện tượng Lost-in-the-Middle) trước khi đưa vào context window của LLM để sinh câu trả lời chính xác và đầy đủ nhất.

---

## Demo video
- Link: `https://drive.google.com/file/d/13LD_RYQ5UwcnotNKtgMJFLgsHKHrS_-8/view?usp=sharing` *(Video 3-5 phút, không chỉnh sửa, demo trả lời trực tiếp 10 câu hỏi mẫu và hiển thị trích dẫn số trang [tr x, y])*.
