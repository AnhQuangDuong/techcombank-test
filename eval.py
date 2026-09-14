#!/usr/bin/env python3
"""
eval.py

Script đánh giá hiệu năng hệ thống Hybrid Search trên tập câu hỏi kiểm thử:
- Đọc tập test từ sample_questions.json.
- Chạy tìm kiếm với từng câu hỏi để lấy ra Top 20 chunks.
- So khớp với gold_printed_pages để xác định tính liên quan (relevance).
- Tính toán và báo cáo các chỉ số chuẩn:
  + Recall@k (k = 1, 3, 5, 10, 20)
  + nDCG@k  (k = 1, 3, 5, 10, 20)
  + MAP@k   (k = 1, 3, 5, 10, 20)
- Xuất báo cáo chi tiết ra màn hình và file eval_results.json.
"""

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from search_qdrant import HybridSearchEngine, SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def compute_recall_at_k(binary_relevance: List[int], k: int) -> float:
    """
    Recall@k: 1.0 nếu có ít nhất 1 chunk liên quan xuất hiện trong Top k, ngược lại 0.0.
    """
    return 1.0 if any(binary_relevance[:k]) else 0.0


def compute_ndcg_at_k(binary_relevance: List[int], k: int, total_relevant: int) -> float:
    """
    Normalized Discounted Cumulative Gain tại k (nDCG@k).
    """
    if total_relevant <= 0:
        return 0.0

    # DCG@k = sum( rel_i / log2(i + 1) ) với i tính từ 1 (tức log2(idx + 2))
    dcg = sum(
        rel / math.log2(idx + 2)
        for idx, rel in enumerate(binary_relevance[:k])
    )

    # IDCG@k: Xếp tất cả các phần tử liên quan (tối đa min(total_relevant, k)) lên đầu
    ideal_k = min(total_relevant, k)
    idcg = sum(
        1.0 / math.log2(idx + 2)
        for idx in range(ideal_k)
    )

    return dcg / idcg if idcg > 0.0 else 0.0


def compute_ap_at_k(binary_relevance: List[int], k: int, total_relevant: int) -> float:
    """
    Average Precision tại k (AP@k).
    """
    if total_relevant <= 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for idx, rel in enumerate(binary_relevance[:k]):
        if rel:
            hits += 1
            sum_precisions += hits / (idx + 1)

    norm_factor = min(total_relevant, k)
    return sum_precisions / norm_factor if norm_factor > 0.0 else 0.0


def evaluate_dataset(
    questions_file: str = "sample_questions.json",
    index_file: str = "output_indexs.json",
    qdrant_path: str = "./qdrant_storage",
    collection_name: str = "techcombank_chunks",
    top_k: int = 20,
    output_file: str = "eval_results.json"
) -> Dict[str, Any]:
    """
    Chạy đánh giá toàn bộ tập test và tính các chỉ số Recall@k, nDCG@k, MAP@k.
    """
    with open(questions_file, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    engine = HybridSearchEngine(
        index_file=index_file,
        qdrant_path=qdrant_path,
        collection_name=collection_name
    )

    k_list = [1, 3, 5, 10, 20]
    # Lọc danh sách k không vượt quá top_k yêu cầu
    k_list = [k for k in k_list if k <= top_k]

    per_query_results = []
    eval_queries_count = 0

    metric_sums = {
        f"recall@{k}": 0.0 for k in k_list
    }
    metric_sums.update({
        f"ndcg@{k}": 0.0 for k in k_list
    })
    metric_sums.update({
        f"map@{k}": 0.0 for k in k_list
    })

    print("\n" + "=" * 90)
    print(f"BẮT ĐẦU ĐÁNH GIÁ TRÊN {len(questions)} CÂU HỎI TỪ: {questions_file}")
    print("=" * 90)

    for item in questions:
        qid = item["id"]
        q_text = item["question"]
        gold_pages = item.get("gold_printed_pages", [])
        gold_ans = item.get("gold_answer", "")
        answerable = item.get("answerable", True)

        # Đếm tổng số chunk thực tế trong toàn bộ corpus thuộc về gold_pages
        total_relevant_in_corpus = 0
        if gold_pages:
            total_relevant_in_corpus = sum(
                1 for c in engine.chunks
                if any(p in gold_pages for p in c.get("logical_page", []))
            )

        # Retrieve Top K chunks
        retrieved_chunks = engine.search(query=q_text, top_k=top_k)

        # Xác định binary relevance cho từng chunk được retrieve
        binary_rel = []
        retrieved_info = []
        first_hit_rank = None

        for rank, res in enumerate(retrieved_chunks, start=1):
            is_relevant = 1 if (gold_pages and any(p in gold_pages for p in res.logical_page)) else 0
            binary_rel.append(is_relevant)

            if is_relevant and first_hit_rank is None:
                first_hit_rank = rank

            retrieved_info.append({
                "rank": rank,
                "chunk_id": res.chunk_id,
                "score": round(res.score, 4),
                "semantic_score": round(res.semantic_score, 4),
                "lexical_score": round(res.lexical_score, 2),
                "logical_page": res.logical_page,
                "pdf_page": res.pdf_page,
                "chapter": res.chapter,
                "heading": res.current_heading,
                "is_relevant": bool(is_relevant),
                "content_snippet": res.content[:200]
            })

        query_metrics = {}
        if answerable and gold_pages:
            eval_queries_count += 1
            for k in k_list:
                r_k = compute_recall_at_k(binary_rel, k)
                n_k = compute_ndcg_at_k(binary_rel, k, total_relevant_in_corpus)
                ap_k = compute_ap_at_k(binary_rel, k, total_relevant_in_corpus)

                query_metrics[f"recall@{k}"] = round(r_k, 4)
                query_metrics[f"ndcg@{k}"] = round(n_k, 4)
                query_metrics[f"ap@{k}"] = round(ap_k, 4)

                metric_sums[f"recall@{k}"] += r_k
                metric_sums[f"ndcg@{k}"] += n_k
                metric_sums[f"map@{k}"] += ap_k

            status_str = f"First Hit: #{first_hit_rank}" if first_hit_rank else "MISS"
            print(f"[{qid}] {q_text[:70]}...")
            print(f"  • Gold Pages: {gold_pages} | {status_str} | Chunks liên quan trong corpus: {total_relevant_in_corpus}")
            print(f"  • Recall@1={query_metrics['recall@1']} | Recall@5={query_metrics['recall@5']} | Recall@20={query_metrics['recall@20']}")
            print(f"  • nDCG@10={query_metrics['ndcg@10']} | nDCG@20={query_metrics['ndcg@20']} | AP@20={query_metrics['ap@20']}")
            print("-" * 90)
        else:
            print(f"[{qid}] (Unanswerable/No gold pages) {q_text[:70]}...")
            print(f"  • Max retrieved score: {retrieved_chunks[0].score:.4f} (Chunk ID: {retrieved_chunks[0].chunk_id})")
            print("-" * 90)

        per_query_results.append({
            "id": qid,
            "question": q_text,
            "category": item.get("category", ""),
            "answerable": answerable,
            "gold_printed_pages": gold_pages,
            "gold_answer": gold_ans,
            "total_relevant_in_corpus": total_relevant_in_corpus,
            "first_hit_rank": first_hit_rank,
            "metrics": query_metrics,
            "retrieved_top_k": retrieved_info
        })

    # Tính trung bình các chỉ số trên các câu hỏi answerable
    aggregated_metrics = {}
    if eval_queries_count > 0:
        for k in k_list:
            aggregated_metrics[f"Recall@{k}"] = round(metric_sums[f"recall@{k}"] / eval_queries_count, 4)
            aggregated_metrics[f"nDCG@{k}"] = round(metric_sums[f"ndcg@{k}"] / eval_queries_count, 4)
            aggregated_metrics[f"MAP@{k}"] = round(metric_sums[f"map@{k}"] / eval_queries_count, 4)

    # Hiển thị bảng tổng kết
    print("\n" + "=" * 90)
    print(f"TỔNG KẾT HIỆU NĂNG TÌM KIẾM (TRÊN {eval_queries_count} CÂU HỎI CÓ ĐÁP ÁN TRONG TÀI LIỆU)")
    print("=" * 90)
    print(f"{'Metric':<15} | {'@1':<10} | {'@3':<10} | {'@5':<10} | {'@10':<10} | {'@20':<10}")
    print("-" * 75)
    for metric_name in ["Recall", "nDCG", "MAP"]:
        row = f"{metric_name:<15} | "
        for k in k_list:
            key = f"{metric_name}@{k}"
            val = aggregated_metrics.get(key, 0.0)
            row += f"{val:<10.4f} | "
        print(row)
    print("=" * 90 + "\n")

    # Ghi file kết quả JSON
    output_data = {
        "dataset": questions_file,
        "total_questions": len(questions),
        "evaluated_questions": eval_queries_count,
        "aggregated_metrics": aggregated_metrics,
        "per_query_details": per_query_results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Đã lưu báo cáo đánh giá chi tiết vào file: {output_file}")
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid Search retrieval on sample_questions.json."
    )
    parser.add_argument(
        "--questions",
        default="sample_questions.json",
        help="File chứa tập câu hỏi kiểm thử (mặc định: sample_questions.json)"
    )
    parser.add_argument(
        "--index-file",
        default="output_indexs.json",
        help="File dữ liệu index (mặc định: output_indexs.json)"
    )
    parser.add_argument(
        "--qdrant-path",
        default="./qdrant_storage",
        help="Thư mục Qdrant Local (mặc định: ./qdrant_storage)"
    )
    parser.add_argument(
        "--collection",
        default="techcombank_chunks",
        help="Tên collection Qdrant (mặc định: techcombank_chunks)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Số lượng chunks retrieve cho mỗi câu hỏi (mặc định: 20)"
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="File lưu kết quả đánh giá (mặc định: eval_results.json)"
    )

    args = parser.parse_args()

    evaluate_dataset(
        questions_file=args.questions,
        index_file=args.index_file,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        top_k=args.top_k,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
