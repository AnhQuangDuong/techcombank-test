import re
from typing import List, Tuple, Optional, Dict, Any

def apply_page_continuation(
    content: str,
    current_headings: Optional[List[str]],
    last_seen_heading: Optional[str],
    fallback_heading: str
) -> Tuple[str, List[str], str]:
    """
    Checks if content starts with '###'. If not, prepends:
    f"### {inherited_heading} (Tiếp tục)\n\n"
    and updates current_headings if empty.
    """
    normalized_content = content.replace("\r\n", "\n")
    stripped = normalized_content.strip()

    if stripped.startswith("###"):
        headings = current_headings or []
        first_line = stripped.split("\n", 1)[0]
        match = re.match(r"^###\s+(.+)$", first_line)
        active = match.group(1).strip() if match else (headings[0] if headings else fallback_heading)
        return normalized_content, headings, active

    inherited = last_seen_heading
    if not inherited and current_headings:
        inherited = current_headings[-1]
    if not inherited:
        inherited = fallback_heading

    prepended_content = f"### {inherited} (Tiếp tục)\n\n" + normalized_content
    updated_headings = current_headings if (current_headings and len(current_headings) > 0) else [inherited]
    return prepended_content, updated_headings, inherited

def parse_h3_blocks(content: str) -> List[Tuple[str, str]]:
    r"""
    Splits markdown content into a list of (heading_title, body_text).
    Uses regex to find all '^###\s+(.+)$' boundaries.
    """
    normalized = content.replace("\r\n", "\n")
    matches = list(re.finditer(r"(?m)^###\s+(.+)$", normalized))
    if not matches:
        return [("", normalized.strip())]

    blocks: List[Tuple[str, str]] = []
    
    # Text before first heading (if any)
    first_start = matches[0].start()
    if first_start > 0:
        orphan = normalized[:first_start].strip()
        if orphan:
            blocks.append(("", orphan))

    for i, m in enumerate(matches):
        heading_title = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
        body_text = normalized[body_start:body_end].strip()
        blocks.append((heading_title, body_text))

    return blocks

def clean_heading_title(title: str) -> str:
    """Removes trailing ' (Tiếp tục)' note for metadata normalization."""
    return re.sub(r"\s*\(Tiếp tục\)\s*$", "", title).strip()

def merge_heading_blocks(blocks: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    Implements State Machine to merge empty consecutive headings into subsequent heading blocks.
    Deduplicates identical consecutive headings.
    Returns list of dicts with keys: 'current_heading', 'heading_prefix', 'body_text'.
    """
    merged: List[Dict[str, str]] = []
    pending_headings: List[str] = []
    pending_raw: List[str] = []

    for heading_title, body_text in blocks:
        h_title = heading_title.strip()
        b_text = body_text.strip()

        if not h_title and not b_text:
            continue

        if not b_text:
            # Empty body: accumulate into pending
            if h_title:
                clean_h = clean_heading_title(h_title)
                if not pending_headings or pending_headings[-1] != clean_h:
                    pending_headings.append(clean_h)
                    pending_raw.append(f"### {h_title}")
            continue

        # Non-empty body
        clean_h = clean_heading_title(h_title) if h_title else ""
        raw_h = f"### {h_title}" if h_title else ""

        # Dedup if h_title is identical to the last pending heading
        if pending_headings and clean_h and pending_headings[-1] == clean_h:
            pending_headings.pop()
            if pending_raw:
                pending_raw.pop()

        if pending_headings:
            if clean_h:
                final_heading = f"{' > '.join(pending_headings)} > {clean_h}"
                final_prefix = "\n\n".join(pending_raw + [raw_h])
            else:
                final_heading = " > ".join(pending_headings)
                final_prefix = "\n\n".join(pending_raw)
        else:
            final_heading = clean_h
            final_prefix = raw_h

        merged.append({
            "current_heading": final_heading,
            "heading_prefix": final_prefix,
            "body_text": b_text
        })

        pending_headings = []
        pending_raw = []

    # If any pending headings left at the end without body
    if pending_headings and merged:
        # Attach to the last merged item
        extra_h = " > ".join(pending_headings)
        merged[-1]["current_heading"] += f" > {extra_h}"
        merged[-1]["heading_prefix"] += "\n\n" + "\n\n".join(pending_raw)
    elif pending_headings and not merged:
        merged.append({
            "current_heading": " > ".join(pending_headings),
            "heading_prefix": "\n\n".join(pending_raw),
            "body_text": ""
        })

    return merged

def extract_atomic_blocks(text: str) -> List[str]:
    """
    Separates HTML tables <table>...</table> as atomic blocks
    and normal text paragraphs.
    """
    pattern = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)
    parts = pattern.split(text)
    blocks: List[str] = []
    for p in parts:
        p_str = p.strip()
        if p_str:
            blocks.append(p_str)
    return blocks

def is_valid_sentence_split(prev_line: str, next_line: str) -> bool:
    """
    Checks if splitting between prev_line and next_line respects sentence/list boundaries.
    """
    prev_s = prev_line.strip()
    next_s = next_line.strip()
    if not prev_s or not next_s:
        return True
    
    # Previous line ends with terminal punctuation
    if re.search(r"[.:;!?]$", prev_s):
        return True
    # Next line starts with bullet / list marker
    if re.match(r"^[-*•]|\d+\.", next_s):
        return True
    return False

def split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    """
    Splits a single paragraph by '\n' at valid sentence boundaries.
    Does NOT split mid-sentence.
    """
    lines = [l.strip() for l in paragraph.split("\n") if l.strip()]
    if not lines:
        return []

    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = 0

    for i, line in enumerate(lines):
        line_len = len(line) + (1 if current_lines else 0)
        if current_len + line_len > max_chars and current_lines:
            if is_valid_sentence_split(current_lines[-1], line):
                chunks.append("\n".join(current_lines))
                current_lines = [line]
                current_len = len(line)
                continue
        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks

def split_body_text(body_text: str, max_body_chars: int) -> List[str]:
    """
    Splits body text into segments within max_body_chars.
    Protects <table>...</table> blocks.
    Prioritizes \n\n, falls back to sentence-valid \n.
    Overlap = 0.
    """
    if len(body_text) <= max_body_chars:
        return [body_text]

    atomic_blocks = extract_atomic_blocks(body_text)
    result_segments: List[str] = []

    for block in atomic_blocks:
        # If block is a table, keep intact even if larger than max_body_chars
        if block.lower().startswith("<table") and block.lower().endswith("</table>"):
            result_segments.append(block)
            continue

        paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
        curr_paras: List[str] = []
        curr_len = 0

        for p in paragraphs:
            if len(p) > max_body_chars:
                if curr_paras:
                    result_segments.append("\n\n".join(curr_paras))
                    curr_paras = []
                    curr_len = 0
                sub_lines = split_long_paragraph(p, max_body_chars)
                result_segments.extend(sub_lines)
                continue

            added_len = len(p) + (2 if curr_paras else 0)
            if curr_len + added_len > max_body_chars and curr_paras:
                result_segments.append("\n\n".join(curr_paras))
                curr_paras = [p]
                curr_len = len(p)
            else:
                curr_paras.append(p)
                curr_len += added_len

        if curr_paras:
            result_segments.append("\n\n".join(curr_paras))

    return result_segments

def create_subchunks(
    heading_prefix: str,
    current_heading: str,
    body_text: str,
    max_chars: int
) -> List[Tuple[str, str]]:
    """
    Constructs subchunks with full heading prefix prepended to each content.
    Returns list of (current_heading, content).
    """
    prefix = heading_prefix.strip()
    prefix_overhead = len(prefix) + 2 if prefix else 0  # for '\n\n'
    max_body = max(max_chars - prefix_overhead, 1)

    if not body_text.strip():
        return [(current_heading, prefix)]

    body_segments = split_body_text(body_text, max_body)
    subchunks: List[Tuple[str, str]] = []

    for seg in body_segments:
        if prefix:
            full_content = f"{prefix}\n\n{seg}"
        else:
            full_content = seg
        subchunks.append((current_heading, full_content))

    return subchunks

def chunk_document(pages: List[Dict[str, Any]], max_chars: int = 2000) -> List[Dict[str, Any]]:
    """
    Processes a list of logical page objects sequentially.
    Applies page continuation, H3 block splitting, consecutive heading merging,
    subchunking, and produces list of enriched chunk objects.
    """
    all_chunks: List[Dict[str, Any]] = []
    last_seen_heading: Optional[str] = None

    for page in pages:
        content = page.get("content", "")
        if not content or not content.strip():
            continue

        curr_headings = page.get("current_headings")
        chapter = page.get("chapter", "")
        section = page.get("section", "")
        fallback = section or chapter or "Tài liệu"

        # 1. Page Continuation
        prepended_content, updated_headings, inherited = apply_page_continuation(
            content=content,
            current_headings=curr_headings,
            last_seen_heading=last_seen_heading,
            fallback_heading=fallback
        )

        # 2. Parse H3 Blocks
        blocks = parse_h3_blocks(prepended_content)

        # 3. Merge Consecutive / Empty Headings & Deduplicate
        merged_blocks = merge_heading_blocks(blocks)

        # 4. Subchunking & Emit Chunks
        for mb in merged_blocks:
            prefix = mb["heading_prefix"]
            heading = mb["current_heading"]
            body = mb["body_text"]

            subchunks = create_subchunks(
                heading_prefix=prefix,
                current_heading=heading,
                body_text=body,
                max_chars=max_chars
            )

            for sub_heading, sub_content in subchunks:
                chunk_obj = {
                    "logical_page": page.get("logical_page"),
                    "pdf_page": page.get("pdf_page"),
                    "chapter": page.get("chapter"),
                    "section": page.get("section"),
                    "current_heading": sub_heading,
                    "content": sub_content
                }
                all_chunks.append(chunk_obj)
                if sub_heading:
                    last_seen_heading = clean_heading_title(sub_heading.split(" > ")[-1])

    return all_chunks

def run_chunker_file(input_file: str, output_file: str, max_chars: int = 2000) -> Dict[str, Any]:
    import json
    from pathlib import Path

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = chunk_document(pages, max_chars=max_chars)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    stats = {
        "input_pages": len(pages),
        "output_chunks": len(chunks),
        "output_file": str(output_path)
    }
    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chunk documents from JSON by H3 headings.")
    parser.add_argument("--input", default="/home/quanganh/techcombank/output_gemini.json", help="Path to input JSON")
    parser.add_argument("--output", default="/home/quanganh/techcombank/output_chunks.json", help="Path to output JSON")
    parser.add_argument("--max-chars", type=int, default=2000, help="Max character size per chunk")

    args = parser.parse_args()
    print(f"Loading from: {args.input}")
    stats = run_chunker_file(args.input, args.output, max_chars=args.max_chars)
    print(f"Done! Processed {stats['input_pages']} pages -> generated {stats['output_chunks']} chunks.")
    print(f"Saved to: {stats['output_file']}")

if __name__ == "__main__":
    main()



