#!/usr/bin/env python3
"""
search_qdrant.py

Hệ thống Hybrid Search kết hợp Qdrant (Semantic Search) và BM25 (Lexical Search):
1. Ingestion: Nạp vector embedding và metadata từ output_indexs.json vào Qdrant Local.
2. Lexical: Khởi tạo BM25Okapi trên contextual text hỗ trợ tiếng Việt & số liệu tài chính.
3. Query Embedding: Embed câu hỏi với task_type="RETRIEVAL_QUERY".
4. Filtering: Hỗ trợ lọc theo metadata (chapter, section, current_heading).
5. Scoring: Min-Max Normalization -> Final Score = 0.4 * Lexical + 0.6 * Semantic.
6. CLI: Giao diện dòng lệnh tương tác trả về Top 10 kết quả liên quan nhất.
"""

import argparse
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*Payload indexes have no effect in the local Qdrant.*")

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: int
    score: float
    lexical_score: float
    semantic_score: float
    chapter: str
    section: str
    current_heading: str
    logical_page: List[int]
    pdf_page: int
    content: str


def tokenize_vi_financial(text: str) -> List[str]:
    """
    Bộ tách từ (tokenizer) tối ưu cho tiếng Việt và thuật ngữ/số liệu báo cáo tài chính:
    - Giữ lại các từ tiếng Việt có dấu.
    - Giữ lại số thập phân (53,4 hoặc 16.5).
    - Giữ lại tỷ lệ phần trăm (16,5%, 40,4%).
    - Giữ lại các từ viết tắt tài chính (CASA, NIM, ROE, ROA, SME...).
    """
    pattern = re.compile(
        r"[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+"
        r"(?:[,.][a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ%]+)*%?",
        re.UNICODE,
    )
    tokens = pattern.findall(text.lower())
    return [t.strip() for t in tokens if t.strip()]


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Chuẩn hóa danh sách điểm số về dải [0, 1] sử dụng Min-Max Scaling.
    Nếu tất cả điểm bằng nhau, trả về 1.0 cho tất cả phần tử.
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [1.0] * len(scores)

    diff = max_s - min_s
    return [(s - min_s) / diff for s in scores]


def compute_hybrid_scores(
    lexical_scores: List[float], semantic_scores: List[float], alpha: float = 0.4, beta: float = 0.6
) -> List[float]:
    """
    Tính điểm tổng hợp: Final Score = alpha * Lexical_norm + beta * Semantic_norm.
    """
    norm_lex = normalize_scores(lexical_scores)
    norm_sem = normalize_scores(semantic_scores)

    return [(alpha * l) + (beta * s) for l, s in zip(norm_lex, norm_sem)]


def get_gemini_query_client() -> Tuple[genai.Client, str]:
    """
    Đọc API key và model từ .env và trả về client cùng tên model.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Biến môi trường GEMINI_API_KEY không tồn tại trong .env!")

    model = os.getenv("GEMINI_EMBEDDING_MODEL")
    if not model:
        raise ValueError("Biến môi trường GEMINI_EMBEDDING_MODEL không tồn tại trong .env!")

    client = genai.Client(api_key=api_key)
    return client, model


def embed_query(client: genai.Client, model: str, query: str) -> List[float]:
    """
    Tạo vector embedding cho câu hỏi tìm kiếm sử dụng task_type='RETRIEVAL_QUERY'.
    """
    response = client.models.embed_content(
        model=model, contents=[query], config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return response.embeddings[0].values


class HybridSearchEngine:
    """
    Động cơ tìm kiếm kết hợp Qdrant Local + BM25 tiếng Việt.
    """

    def __init__(
        self,
        index_file: str = "output_indexs.json",
        qdrant_path: str = "./qdrant_storage",
        collection_name: str = "techcombank_chunks",
        force_reindex: bool = False,
    ):
        self.index_file = index_file
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.client, self.embedding_model = get_gemini_query_client()

        # 1. Nạp dữ liệu chunks từ file index
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {index_file}")

        with open(index_file, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)

        if not self.chunks:
            raise ValueError(f"File {index_file} không có dữ liệu!")

        first_emb = self.chunks[0].get("embedding")
        if not first_emb:
            raise ValueError("Dữ liệu chunks không có trường 'embedding'!")

        self.vector_dim = len(first_emb)
        logger.info(f"Đã nạp {len(self.chunks)} chunks. Vector dimension: {self.vector_dim}")

        # 2. Khởi tạo Qdrant Local
        self.qdrant = QdrantClient(path=self.qdrant_path)
        self._setup_qdrant(force_reindex)

        # 3. Khởi tạo BM25 Engine
        self._setup_bm25()

    def _setup_qdrant(self, force_reindex: bool):
        """Khởi tạo collection và nạp dữ liệu vào Qdrant."""
        collections = [c.name for c in self.qdrant.get_collections().collections]

        needs_ingestion = force_reindex or (self.collection_name not in collections)
        if not needs_ingestion:
            info = self.qdrant.get_collection(self.collection_name)
            if info.points_count != len(self.chunks):
                logger.info(
                    f"Collection hiện có {info.points_count} points, khác số chunk {len(self.chunks)}. Tiến hành nạp lại."
                )
                needs_ingestion = True

        if needs_ingestion:
            if self.collection_name in collections:
                self.qdrant.delete_collection(self.collection_name)

            logger.info(f"Tạo collection '{self.collection_name}' (dim={self.vector_dim}, Cosine)...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

            # Tạo Payload Index cho các trường metadata phục vụ filter
            for field in ["chapter", "section", "current_heading"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        self.qdrant.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field,
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                    except Exception:
                        pass

            # Nạp points theo batch
            batch_size = 100
            points = []
            for idx, c in enumerate(self.chunks):
                payload = {
                    "logical_page": c.get("logical_page", []),
                    "pdf_page": c.get("pdf_page", 0),
                    "chapter": c.get("chapter", ""),
                    "section": c.get("section", ""),
                    "current_heading": c.get("current_heading", ""),
                    "content": c.get("content", ""),
                    "chunk_id": idx,
                }
                points.append(PointStruct(id=idx, vector=c["embedding"], payload=payload))

            for i in range(0, len(points), batch_size):
                self.qdrant.upsert(collection_name=self.collection_name, points=points[i : i + batch_size])
            logger.info(f"Đã nạp thành công {len(points)} points vào Qdrant Local.")
        else:
            logger.info(f"Collection '{self.collection_name}' đã sẵn sàng ({len(self.chunks)} points).")

    def _setup_bm25(self):
        """Xây dựng BM25 index trên contextual text của toàn bộ chunks."""
        corpus_tokens = []
        for c in self.chunks:
            prefix = f"Chương: {c.get('chapter', '')} | Mục: {c.get('section', '')} | Tiêu đề: {c.get('current_heading', '')}"
            full_text = f"{prefix}\n\n{c.get('content', '')}"
            tokens = tokenize_vi_financial(full_text)
            corpus_tokens.append(tokens)

        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Đã khởi tạo BM25 index cho {len(corpus_tokens)} chunks.")

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None,
        top_k: int = 10,
        alpha: float = 0.4,
        beta: float = 0.6,
    ) -> List[SearchResult]:
        """
        Thực hiện tìm kiếm lai (Hybrid Search):
        1. Embed câu hỏi với task_type='RETRIEVAL_QUERY'.
        2. Query Semantic Search trên Qdrant với metadata filter (nếu có).
        3. Tính điểm BM25 cho các ứng viên.
        4. Chuẩn hóa và tổng hợp điểm: 0.4 * Lexical + 0.6 * Semantic.
        5. Trả về Top K kết quả cao nhất.
        """
        # 1. Embed câu hỏi
        query_vector = embed_query(self.client, self.embedding_model, query)

        # 2. Xây dựng Filter Qdrant nếu có
        qdrant_filter = None
        if filters:
            conditions = []
            for field, val in filters.items():
                if val:
                    conditions.append(FieldCondition(key=field, match=MatchValue(value=val)))
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # 3. Lấy kết quả từ Qdrant
        # Lấy tối đa toàn bộ số points thỏa mãn điều kiện filter để tính hybrid score
        limit_candidates = len(self.chunks)
        query_res = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit_candidates,
            with_payload=True,
        )

        candidate_points = query_res.points
        if not candidate_points:
            return []

        # 4. Tính điểm BM25 cho các ứng viên
        query_tokens = tokenize_vi_financial(query)
        all_bm25_scores = self.bm25.get_scores(query_tokens)

        candidate_chunk_ids = [p.id for p in candidate_points]
        semantic_scores = [float(p.score) for p in candidate_points]
        lexical_scores = [float(all_bm25_scores[cid]) for cid in candidate_chunk_ids]

        # 5. Tính Hybrid Score
        hybrid_scores = compute_hybrid_scores(
            lexical_scores=lexical_scores, semantic_scores=semantic_scores, alpha=alpha, beta=beta
        )

        # 6. Gom kết quả và sắp xếp giảm dần
        results: List[SearchResult] = []
        for point, final_s, lex_s, sem_s in zip(candidate_points, hybrid_scores, lexical_scores, semantic_scores):
            payload = point.payload or {}
            results.append(
                SearchResult(
                    chunk_id=point.id,
                    score=final_s,
                    lexical_score=lex_s,
                    semantic_score=sem_s,
                    chapter=payload.get("chapter", ""),
                    section=payload.get("section", ""),
                    current_heading=payload.get("current_heading", ""),
                    logical_page=payload.get("logical_page", []),
                    pdf_page=payload.get("pdf_page", 0),
                    content=payload.get("content", ""),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


def print_search_results(results: List[SearchResult], query: str):
    """Hiển thị kết quả tìm kiếm đẹp mắt trên terminal."""
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ TÌM KIẾM CHO: '{query}'")
    print(f"Tổng số kết quả trả về: {len(results)}")
    print("=" * 80)

    for rank, res in enumerate(results, start=1):
        print(
            f"\n[#{rank}] Score: {res.score:.4f} (Semantic: {res.semantic_score:.4f} | Lexical: {res.lexical_score:.2f})"
        )
        print(f"  • Vị trí: Trang logic {res.logical_page} (Trang PDF: {res.pdf_page})")
        print(f"  • Phân cấp: {res.chapter} > {res.section} > {res.current_heading}")
        print("  • Nội dung:")
        # In snippet nội dung
        content_lines = res.content.strip().split("\n")
        snippet = "\n    ".join(content_lines[:6])
        if len(content_lines) > 6:
            snippet += "\n    ..."
        print(f"    {snippet}")
        print("-" * 80)


def interactive_cli(engine: HybridSearchEngine):
    """Vòng lặp CLI tìm kiếm tương tác."""
    print("\n========================================================")
    print("  HỆ THỐNG HYBRID SEARCH TECHCOMBANK (QDRANT + BM25)")
    print("========================================================")
    print("Hướng dẫn:")
    print("  - Nhập câu hỏi bình thường để tìm kiếm.")
    print("  - Đặt bộ lọc metadata:  filter: chapter=Tên_chương")
    print("                          filter: section=Tên_mục")
    print("  - Xem bộ lọc hiện tại:  filters")
    print("  - Xóa toàn bộ bộ lọc:   clear")
    print("  - Thoát chương trình:   exit / quit")
    print("========================================================\n")

    active_filters: Dict[str, str] = {}

    while True:
        try:
            prompt = "Search> "
            if active_filters:
                filter_str = ", ".join([f"{k}='{v}'" for k, v in active_filters.items()])
                prompt = f"Search [{filter_str}]> "

            user_input = input(prompt).strip()
            if not user_input:
                continue

            lower_input = user_input.lower()
            if lower_input in ["exit", "quit", "q"]:
                print("Tạm biệt!")
                break

            if lower_input == "clear":
                active_filters.clear()
                print("Đã xóa toàn bộ bộ lọc.")
                continue

            if lower_input == "filters":
                print("Bộ lọc hiện tại:", active_filters if active_filters else "Không có (tìm trên toàn bộ tài liệu)")
                continue

            if user_input.startswith("filter:"):
                filter_expr = user_input[len("filter:") :].strip()
                if "=" in filter_expr:
                    k, v = filter_expr.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    active_filters[k] = v
                    print(f"Đã kích hoạt bộ lọc: {k} = '{v}'")
                else:
                    print("Cú pháp không hợp lệ. Ví dụ: filter: chapter=Chúng tôi là ai")
                continue

            # Thực hiện tìm kiếm
            results = engine.search(query=user_input, filters=active_filters if active_filters else None, top_k=10)
            print_search_results(results, user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search on Techcombank Annual Report using Qdrant and BM25.")
    parser.add_argument(
        "--index-file",
        default="output_indexs.json",
        help="Đường dẫn file output_indexs.json (mặc định: output_indexs.json)",
    )
    parser.add_argument(
        "--qdrant-path", default="./qdrant_storage", help="Đường dẫn lưu trữ Qdrant Local (mặc định: ./qdrant_storage)"
    )
    parser.add_argument(
        "--collection", default="techcombank_chunks", help="Tên Qdrant collection (mặc định: techcombank_chunks)"
    )
    parser.add_argument("--reindex", action="store_true", help="Bắt buộc nạp lại dữ liệu vào Qdrant")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Câu hỏi tìm kiếm một lần qua CLI (nếu không truyền sẽ mở interactive mode)",
    )

    args = parser.parse_args()

    engine = HybridSearchEngine(
        index_file=args.index_file,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        force_reindex=args.reindex,
    )

    if args.query:
        results = engine.search(query=args.query, top_k=10)
        print_search_results(results, args.query)
    else:
        interactive_cli(engine)


if __name__ == "__main__":
    main()
