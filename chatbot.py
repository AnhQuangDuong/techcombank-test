#!/usr/bin/env python3
"""
chatbot.py

Hệ thống Chatbot RAG đa lượt (Multi-turn Conversational RAG) tra cứu Báo cáo thường niên Techcombank:
1. Multi-turn Query Rewriter: Viết lại câu hỏi nối tiếp thành câu truy vấn độc lập.
2. Hybrid Retrieval: Tìm kiếm Top K chunks liên quan nhất từ Qdrant Local + BM25.
3. Grounded Answer Generation: Gọi Gemini LLM với System Instruction nghiêm ngặt:
   - Chỉ dùng tài liệu được cung cấp.
   - Nếu không đủ thông tin -> "Tôi không đủ thông tin để trả lời câu hỏi này."
   - Bắt buộc đính kèm trích dẫn số trang [tr x, y] ở các ý khẳng định.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from search_qdrant import HybridSearchEngine, SearchResult

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_chatbot_config() -> Tuple[str, str, str]:
    """
    Đọc cấu hình API key, LLM model và embedding model từ .env.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Biến môi trường GEMINI_API_KEY không tồn tại trong .env!")

    llm_model = os.getenv("GEMINI_MODEL")
    if not llm_model:
        raise ValueError("Biến môi trường GEMINI_MODEL không tồn tại trong .env!")

    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("Biến môi trường GEMINI_EMBEDDING_MODEL không tồn tại trong .env!")

    return api_key, llm_model, embedding_model


def format_retrieved_context(chunks: List[SearchResult]) -> str:
    """
    Định dạng danh sách các chunks được retrieve thành ngữ cảnh văn bản cho LLM,
    tập trung vào số trang logic để LLM trích dẫn chính xác [tr x, y].
    """
    context_parts = []
    for chunk in chunks:
        pages_str = ", ".join(str(p) for p in chunk.logical_page)
        header = f"[Trang {pages_str} | Chương: {chunk.chapter} | Tiêu đề: {chunk.current_heading}]"
        content_text = chunk.content.strip()
        context_parts.append(f"{header}\n{content_text}")

    delimiter = "\n\n" + ("=" * 50) + "\n\n"
    return delimiter + ("\n\n" + ("-" * 40) + "\n\n").join(context_parts) + delimiter


def build_rephrase_prompt(history: List[Dict[str, str]], latest_query: str) -> str:
    """
    Tạo prompt yêu cầu LLM viết lại câu hỏi nối tiếp thành câu truy vấn độc lập.
    """
    history_lines = []
    for turn in history[-4:]:  # Lấy tối đa 4 lượt hội thoại gần nhất
        role_label = "Người dùng" if turn.get("role") == "user" else "Trợ lý"
        history_lines.append(f"{role_label}: {turn.get('text', '')}")

    formatted_history = "\n".join(history_lines)

    return f"""Dưới đây là lịch sử hội thoại gần nhất giữa Người dùng và Trợ lý, cùng với câu hỏi mới của Người dùng.

[LỊCH SỬ HỘI THOẠI]:
{formatted_history}

[CÂU HỎI MỚI CỦA NGƯỜI DÙNG]:
"{latest_query}"

Nhiệm vụ:
- Nếu câu hỏi mới có từ ngữ tham chiếu, nối tiếp ý (ví dụ: "còn năm 2024 thì sao?", "ông ấy là ai?", "tại sao?"), hãy viết lại câu hỏi thành MỘT CÂU TRUY VẤN ĐỘC LẬP bằng tiếng Việt, đầy đủ chủ ngữ, vị ngữ và thực thể (như ngân hàng Techcombank, năm liên quan...) để phục vụ tìm kiếm tài liệu.
- Nếu câu hỏi mới đã đầy đủ nghĩa và độc lập, hãy giữ nguyên câu hỏi.
- CHỈ TRẢ VỀ DUY NHẤT CÂU TRUY VẤN ĐƯỢC VIẾT LẠI, không thêm lời giải thích nào khác."""


def rephrase_query_if_needed(client: genai.Client, model: str, history: List[Dict[str, str]], query: str) -> str:
    """
    Nếu chưa có lịch sử chat (lượt 1), trả về câu hỏi gốc.
    Nếu đã có lịch sử chat (lượt 2+), gọi Gemini LLM để viết lại câu hỏi thành standalone query.
    """
    cleaned_query = query.strip()
    if not history:
        return cleaned_query

    prompt = build_rephrase_prompt(history, cleaned_query)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )
        rephrased = response.text.strip()
        # Loại bỏ dấu ngoặc kép thừa nếu có
        rephrased = rephrased.strip("\"'")
        return rephrased if rephrased else cleaned_query
    except Exception as e:
        logger.warning(f"Lỗi khi rephrase query: {e}. Sử dụng query gốc.")
        return cleaned_query


def build_answer_system_instruction() -> str:
    """
    System Instruction nghiêm ngặt cho Answer Generator:
    - Bắt buộc tuân thủ Grounding (chỉ dùng tài liệu được cung cấp).
    - Không đủ thông tin -> 'Tôi không đủ thông tin để trả lời câu hỏi này.'
    - Gắn trích dẫn [tr x, y] ở các ý khẳng định.
    """
    return """Bạn là trợ lý AI chuyên nghiệp phụ trách tra cứu thông tin chính thức từ Báo cáo Thường niên của Ngân hàng TMCP Kỹ Thương Việt Nam (Techcombank).

Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa DUY NHẤT vào các đoạn văn bản được cung cấp trong mục [NGỮ CẢNH TÀI LIỆU].

CÁC NGUYÊN TẮC BẮT BUỘC:
1. CHỈ DỰA VÀO TÀI LIỆU ĐƯỢC CẤP: Tuyệt đối không suy đoán, không sử dụng kiến thức bên ngoài tài liệu được cung cấp. Mọi thông tin, số liệu bạn đưa ra phải có nguồn gốc rõ ràng từ các đoạn tài liệu.
2. QUY TẮC TỪ CHỐI KHI THIẾU THÔNG TIN: Nếu các đoạn tài liệu được cung cấp KHÔNG CHỨA THÔNG TIN hoặc KHÔNG ĐỦ CƠ SỞ để trả lời câu hỏi, bạn BẮT BUỘC phải trả lời chính xác câu sau:
   "Tôi không đủ thông tin để trả lời câu hỏi này."
   (Không tự tiện suy diễn hoặc cố gắng trả lời một phần khi không chắc chắn).
3. QUY TẮC TRÍCH DẪN NGUỒN TRANG [tr x, y]:
   - Sau mỗi ý khẳng định, nhận định hoặc số liệu cụ thể lấy từ tài liệu, bạn BẮT BUỘC phải ghi rõ số trang tương ứng ngay cạnh ý đó theo định dạng [tr x] hoặc [tr x, y].
   - Ví dụ: "Tỷ lệ CASA năm 2025 của Techcombank đạt 40,4% [tr 5, 53], tiếp tục dẫn đầu toàn ngành ngân hàng [tr 5]."
4. VĂN PHONG: Khách quan, trung thực, gãy gọn, chuẩn xác theo thuật ngữ và số liệu tài chính."""


def generate_answer_with_rag(
    client: genai.Client, model: str, context: str, history: List[Dict[str, str]], user_query: str
) -> str:
    """
    Tạo câu trả lời RAG dựa trên context được retrieve và lịch sử hội thoại.
    """
    system_prompt = build_answer_system_instruction()

    # Xây dựng contents kèm lịch sử hội thoại gần nhất
    contents: List[Any] = []
    for turn in history[-6:]:  # Giữ tối đa 6 lượt chat gần nhất
        role = turn.get("role", "user")
        text = turn.get("text", "")
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    # Lượt hiện tại đính kèm ngữ cảnh tài liệu vừa retrieve
    current_prompt = f"""[NGỮ CẢNH TÀI LIỆU ĐƯỢC CUNG CẤP]:
{context}

[CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG]:
{user_query}

Hãy trả lời câu hỏi trên dựa trên [NGỮ CẢNH TÀI LIỆU ĐƯỢC CUNG CẤP] theo đúng các quy tắc bắt buộc (đính kèm trích dẫn [tr x, y] hoặc từ chối nếu không đủ thông tin):"""

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=current_prompt)]))

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini generate_content: {e}")
        return f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời: {e}"


class ChatbotSession:
    """
    Quản lý phiên hội thoại đa lượt với RAG pipeline.
    """

    def __init__(
        self,
        index_file: str = "output_indexs.json",
        qdrant_path: str = "./qdrant_storage",
        collection_name: str = "techcombank_chunks",
        top_k: int = 7,
    ):
        self.api_key, self.llm_model, self.embedding_model = get_chatbot_config()
        self.client = genai.Client(api_key=self.api_key)
        self.top_k = top_k
        self.history: List[Dict[str, str]] = []

        # Khởi tạo engine tìm kiếm lai
        self.engine = HybridSearchEngine(
            index_file=index_file, qdrant_path=qdrant_path, collection_name=collection_name
        )

    def reset(self):
        """Xóa lịch sử hội thoại, bắt đầu phiên mới."""
        self.history.clear()

    def ask(self, user_query: str) -> Tuple[str, List[SearchResult], str]:
        """
        Xử lý một câu hỏi của người dùng trong phiên hội thoại:
        1. Viết lại câu hỏi (Rephrase Query) nếu có lịch sử.
        2. Tìm kiếm Top K chunks bằng Hybrid Search.
        3. Định dạng context.
        4. Gọi Gemini LLM tạo câu trả lời.
        5. Cập nhật lịch sử hội thoại.
        """
        # 1. Rephrase query nếu cần
        search_query = rephrase_query_if_needed(self.client, self.llm_model, self.history, user_query)

        # 2. Retrieve tài liệu liên quan
        retrieved_chunks = self.engine.search(query=search_query, top_k=self.top_k)

        # 3. Định dạng ngữ cảnh
        context_str = format_retrieved_context(retrieved_chunks)

        # 4. Sinh câu trả lời
        answer = generate_answer_with_rag(
            client=self.client, model=self.llm_model, context=context_str, history=self.history, user_query=user_query
        )

        # 5. Lưu vào lịch sử hội thoại
        self.history.append({"role": "user", "text": user_query})
        self.history.append({"role": "model", "text": answer})

        return answer, retrieved_chunks, search_query


def run_cli():
    parser = argparse.ArgumentParser(
        description="Interactive Chatbot with Conversational RAG for Techcombank Annual Report."
    )
    parser.add_argument(
        "--index-file", default="output_indexs.json", help="Đường dẫn file index (mặc định: output_indexs.json)"
    )
    parser.add_argument(
        "--qdrant-path", default="./qdrant_storage", help="Thư mục Qdrant Local (mặc định: ./qdrant_storage)"
    )
    parser.add_argument("--top-k", type=int, default=7, help="Số lượng chunks retrieve (mặc định: 7)")
    parser.add_argument(
        "--query", type=str, default=None, help="Câu hỏi chạy một lần (nếu không truyền sẽ mở chế độ chat tương tác)"
    )
    parser.add_argument(
        "--batch", type=str, default=None, help="Đường dẫn file JSON câu hỏi chạy batch (ví dụ: sample_questions.json)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Khoảng nghỉ (giây) giữa các câu hỏi trong batch để tránh rate limit (mặc định: 4.0s)",
    )

    args = parser.parse_args()

    session = ChatbotSession(index_file=args.index_file, qdrant_path=args.qdrant_path, top_k=args.top_k)

    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            questions = json.load(f)

        print("\n" + "=" * 75)
        print(f"BẮT ĐẦU CHẠY BATCH {len(questions)} CÂU HỎI TỪ: {args.batch}")
        print("=" * 75)

        for idx, item in enumerate(questions, 1):
            q_text = item.get("question", "")
            print(f"\n[{idx}/{len(questions)}] Câu hỏi: {q_text}", flush=True)
            print("Đang tra cứu và sinh câu trả lời...", flush=True)

            t_start = time.time()
            answer, _, _ = session.ask(q_text)
            elapsed = time.time() - t_start
            session.reset()

            print(f"Trợ lý: {answer}")
            gold_ans = item.get("gold_answer")
            if gold_ans:
                gold_pages = item.get("gold_printed_pages")
                pages_str = f" [Trang {', '.join(map(str, gold_pages))}]" if gold_pages else ""
                print(f"🎯 Gold Answer: {gold_ans}{pages_str}")
            print(f"⏱️  [Thời gian phản hồi: {elapsed:.2f}s]", flush=True)

            if idx < len(questions) and args.delay > 0:
                print(f"(Nghỉ {args.delay}s để đảm bảo Rate Limit 15 RPM...)", flush=True)
                time.sleep(args.delay)

        print("\n" + "=" * 75)
        print("HOÀN TẤT CHẠY BATCH!")
        print("=" * 75 + "\n")
        return

    if args.query:
        t_start = time.time()
        answer, retrieved, search_q = session.ask(args.query)
        elapsed = time.time() - t_start
        print(f"\nBạn: {args.query}")
        print(f"Trợ lý: {answer}")
        print(f"⏱️  [Thời gian phản hồi: {elapsed:.2f}s]\n")
        return

    print("\n" + "=" * 70)
    print("  CHATBOT TRA CỨU BÁO CÁO THƯỜNG NIÊN TECHCOMBANK 2025")
    print("=" * 70)
    print("Các lệnh điều khiển:")
    print("  - new / reset : Bắt đầu phiên trò chuyện mới (xóa lịch sử)")
    print("  - sources     : Bật/tắt hiển thị nguồn tài liệu trích dẫn")
    print("  - exit / quit : Thoát chương trình")
    print("=" * 70 + "\n")

    show_sources = False

    while True:
        try:
            user_input = input("\nBạn: ").strip()
            if not user_input:
                continue

            lower = user_input.lower()
            if lower in ["exit", "quit", "q"]:
                print("\nCảm ơn bạn đã sử dụng. Tạm biệt!\n")
                break

            if lower in ["reset", "new"]:
                session.reset()
                print("\n[Đã làm mới phiên trò chuyện. Lịch sử đã được xóa.]")
                continue

            if lower == "sources":
                show_sources = not show_sources
                status = "BẬT" if show_sources else "TẮT"
                print(f"\n[Chế độ hiển thị nguồn tài liệu: {status}]")
                continue

            print("\nĐang tra cứu và tổng hợp câu trả lời...", end="\r", flush=True)

            t_start = time.time()
            answer, retrieved, search_q = session.ask(user_input)
            elapsed = time.time() - t_start

            # Xóa dòng 'Đang tra cứu...'
            sys.stdout.write("\033[K")

            if search_q != user_input:
                print(f"[Query tìm kiếm độc lập: '{search_q}']")

            print(f"Trợ lý: {answer}")
            print(f"⏱️  [Thời gian phản hồi: {elapsed:.2f}s]\n")

            if show_sources:
                print("-" * 50)
                print("TÀI LIỆU NGUỒN ĐƯỢC RETRIEVE:")
                for r in retrieved[:3]:
                    print(f"• [Trang {r.logical_page}] (Score: {r.score:.4f}) {r.chapter} > {r.current_heading}")
                print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break
        except Exception as e:
            print(f"\nLỗi: {e}")


if __name__ == "__main__":
    run_cli()
