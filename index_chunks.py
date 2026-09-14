#!/usr/bin/env python3
"""
index_chunks.py

Pipeline tạo embedding cho các chunk bằng Gemini Embedding API:
1. Đọc pp_output_chunks.json.
2. Tạo contextual text: {chapter} > {section} > {current_heading}\n\n{content}.
3. Gọi Gemini Embedding API theo batch (50 chunks/batch) với task_type="RETRIEVAL_DOCUMENT".
4. Giữ nguyên 100% metadata gốc, bổ sung trường 'embedding'.
5. Lưu kết quả ra output_indexs.json (hỗ trợ checkpoint resume).
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_gemini_config(dotenv_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Đọc cấu hình API key và model từ biến môi trường.
    Bắt buộc phải có trong .env, không dùng fallback ngầm.
    """
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Biến môi trường GEMINI_API_KEY không tồn tại trong .env hoặc bị rỗng!")

    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("Biến môi trường GEMINI_EMBEDDING_MODEL không tồn tại trong .env hoặc bị rỗng!")

    return api_key, embedding_model


def get_gemini_client(api_key: str) -> genai.Client:
    """Khởi tạo Gemini client."""
    return genai.Client(api_key=api_key)


def prepare_contextual_text(chunk: Dict[str, Any]) -> str:
    """
    Tạo văn bản contextual để gửi vào embedding model.
    Đính kèm Chương, Mục và Tiêu đề để giữ trọn vẹn ngữ cảnh của đoạn nội dung.
    """
    chapter = chunk.get("chapter", "")
    section = chunk.get("section", "")
    heading = chunk.get("current_heading", "")
    content = chunk.get("content", "")

    prefix = f"Chương: {chapter} | Mục: {section} | Tiêu đề: {heading}"
    return f"{prefix}\n\n{content}"


def embed_batch(
    client: genai.Client,
    model: str,
    texts: List[str],
    max_retries: int = 6,
    initial_backoff: float = 3.0
) -> List[List[float]]:
    """
    Gửi batch text tới Gemini Embedding API với cơ chế Exponential Backoff và Adaptive Split:
    - Nếu batch quá lớn bị lỗi quota/size (429), tự động chia đôi batch để gửi.
    - Tự động trích xuất retryDelay khi gặp lỗi 429 RESOURCE_EXHAUSTED.
    """
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            embeddings = [emb.values for emb in response.embeddings]
            return embeddings
        except Exception as e:
            err_str = str(e)
            logger.warning(
                f"[Lần thử {attempt}/{max_retries}] Gặp lỗi khi embed batch {len(texts)} chunks: {err_str[:160]}..."
            )

            # Nếu batch nhiều hơn 10 phần tử và dính lỗi, chia đôi batch để thử
            if len(texts) > 10 and ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()):
                mid = len(texts) // 2
                logger.info(f"Tự động chia nhỏ batch ({len(texts)} chunks -> {mid} và {len(texts)-mid} chunks)...")
                left_vectors = embed_batch(client, model, texts[:mid], max_retries=max_retries)
                time.sleep(1.0)
                right_vectors = embed_batch(client, model, texts[mid:], max_retries=max_retries)
                return left_vectors + right_vectors

            if attempt == max_retries:
                raise RuntimeError(
                    f"Đã thử {max_retries} lần nhưng không thể embed batch: {e}"
                ) from e

            # Kiểm tra xem có retryDelay cụ thể từ Google API không
            wait_time = backoff
            retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
            delay_match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s?", err_str, re.IGNORECASE)
            if retry_match:
                wait_time = float(retry_match.group(1)) + 2.0
            elif delay_match:
                wait_time = float(delay_match.group(1)) + 2.0
            elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = max(backoff, 25.0)

            logger.info(f"Tạm nghỉ {wait_time:.1f}s trước khi thử lại batch...")
            time.sleep(wait_time)
            backoff *= 2.0

    return []


def run_indexing(
    input_path: str = "pp_output_chunks.json",
    output_path: str = "output_indexs.json",
    batch_size: int = 25,
    sleep_delay: float = 2.0,
    limit: Optional[int] = None
) -> int:
    """
    Chạy pipeline indexing:
    - Hỗ trợ resume từ checkpoint nếu output_path đã có dữ liệu dở dang.
    - Hỗ trợ tham số limit để giới hạn số chunk cần embed khi test/tiết kiệm quota.
    - Lưu file theo từng batch để đảm bảo an toàn dữ liệu.
    """
    api_key, model = get_gemini_config()
    client = get_gemini_client(api_key)

    with open(input_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    if limit is not None and limit > 0:
        chunks = chunks[:limit]
        logger.info(f"Đã áp dụng giới hạn: chỉ xử lý {len(chunks)} chunks để tiết kiệm quota.")

    total_chunks = len(chunks)
    logger.info(f"Tổng số chunk cần xử lý: {total_chunks}")
    logger.info(f"Sử dụng model: {model} | Batch size: {batch_size}")

    # Kiểm tra checkpoint nếu file output đã tồn tại
    existing_records: List[Dict[str, Any]] = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    existing_records = loaded
                    logger.info(
                        f"Phát hiện file checkpoint '{output_path}' với {len(existing_records)} bản ghi."
                    )
        except Exception as e:
            logger.warning(f"Không thể đọc file checkpoint cũ: {e}. Bắt đầu mới.")
            existing_records = []

    # Map các chunk đã có embedding
    processed_count = 0
    results: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        if idx < len(existing_records) and "embedding" in existing_records[idx]:
            # Đã có embedding hợp lệ từ checkpoint
            results.append(existing_records[idx])
            processed_count += 1
        else:
            # Chưa có embedding, cần xử lý
            break

    if processed_count > 0:
        logger.info(f"Đã phục hồi {processed_count}/{total_chunks} chunk từ checkpoint.")

    # Xử lý các chunk còn lại theo từng batch
    remaining_start = processed_count
    for batch_start in range(remaining_start, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        current_batch_chunks = chunks[batch_start:batch_end]

        batch_texts = [
            prepare_contextual_text(c) for c in current_batch_chunks
        ]

        t0 = time.time()
        batch_vectors = embed_batch(client, model, batch_texts)
        duration = time.time() - t0

        logger.info(
            f"Embed thành công batch {batch_start + 1}-{batch_end}/{total_chunks} "
            f"({len(batch_vectors)} chunks) trong {duration:.2f}s."
        )

        for chunk_item, vector in zip(current_batch_chunks, batch_vectors):
            chunk_copy = dict(chunk_item)
            chunk_copy["embedding"] = vector
            results.append(chunk_copy)

        # Lưu checkpoint sau mỗi batch hoàn thành
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if batch_end < total_chunks:
            time.sleep(sleep_delay)

    logger.info(
        f"Hoàn thành toàn bộ pipeline indexing! {len(results)}/{total_chunks} "
        f"chunks đã được lưu vào '{output_path}'."
    )
    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Index document chunks using Gemini embedding model."
    )
    parser.add_argument(
        "--input",
        default="pp_output_chunks.json",
        help="Đường dẫn file chunks đầu vào (mặc định: pp_output_chunks.json)"
    )
    parser.add_argument(
        "--output",
        default="output_indexs.json",
        help="Đường dẫn file kết quả (mặc định: output_indexs.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Kích thước batch gửi tới Gemini API (mặc định: 50)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Khoảng nghỉ (giây) giữa các batch (mặc định: 2.0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số lượng chunk cần embed để tiết kiệm quota (mặc định: None - toàn bộ)"
    )

    args = parser.parse_args()

    run_indexing(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        sleep_delay=args.sleep,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
