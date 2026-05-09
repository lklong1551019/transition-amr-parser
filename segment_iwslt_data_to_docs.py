"""
segment_iwslt_data_to_docs.py

This script processes parallel IWSLT dataset files (English-Vietnamese) to extract
document-level text files and corresponding AMR Penman files.

Key Features:
1.  **Parallel Extraction**: Processes .tags and .xml files to extract aligned documents.
2.  **AMR-Guided Segmentation**: Uses the transition-amr-parser to detect if a sentence
    should be split further. It follows a multi-level splitting strategy:
    - Level 0: Parse the line as is.
    - Level 1: Regex-based split on punctuation and conjunctions.
    - Level 2: Aggressive punctuation-based split.
3.  **Parallel Alignment**: Ensures that English and Vietnamese sentences remain 1:1 aligned.
    If a split results in a count mismatch, the entire parallel line is dropped.
4.  **Unified Parsing**: Reuses the AMR parser output from the segmentation phase to
    generate Penman AMR files directly, saving 50% of computational time.
5.  **Vocabulary Generation**: Collects all unique AMR relations and concepts (frames) 
    encountered during parsing and saves them to amrs_token.json and amrs_token_simple.json.

Output Structure:
- output_dataset_doc/   : Aligned .en and .vi text documents.
- output_dataset_amr/   : Corresponding Penman AMR blocks.
- amrs_token.json       : Unique relation and concept vocabulary.
- amrs_token_simple.json: Simplified vocabulary (numbers/frames removed).
"""

import os
import sys
import re
import shutil
import json
from typing import List, Set, Tuple, Optional, Any
from transition_amr_parser.parse import AMRParser


# ---------------------------------------------------------------------------
# Global sets to collect all unique relations and concepts across all documents
# ---------------------------------------------------------------------------
all_unique_relations: Set[str] = set()
all_unique_concepts: Set[str] = set()
all_ignored_concepts: Set[str] = set()


# ---------------------------------------------------------------------------
# Load the parser once (expensive – do NOT reload inside loops)
# ---------------------------------------------------------------------------
print("Loading AMR parser model (AMR3-structbart-L)...")
parser = AMRParser.from_pretrained('AMR3-structbart-L')
print("Parser ready.\n")


# ---------------------------------------------------------------------------
# Load BERT vocab to filter out common words
# ---------------------------------------------------------------------------
bert_vocab: Set[str] = set()
bert_vocab_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi-bert-base-cased-vocab.txt")
if os.path.exists(bert_vocab_file):
    print(f"Loading BERT vocab from {os.path.basename(bert_vocab_file)}...")
    with open(bert_vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            bert_vocab.add(line.strip())
    print(f"Loaded {len(bert_vocab)} vocab items.\n")


def is_number(s: str) -> bool:
    return bool(re.match(r'^-?\d+(?:\.\d+)?$', s))


def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using regex.
    Splits on . ! ? ONLY if followed by an uppercase letter (with or without whitespace).
    Also splits on ; and -- (with optional whitespace).
    Avoids splitting after single-letter acronyms (e.g., U.S.A.) when whitespace is missing.
    Also strips leading ellipses (...) from sentences.
    """
    # Uses multiple fixed-width look-behinds to keep punctuation with the sentence.
    # Added branches for ; and -- to split sentences.
    # Also allows for sentences starting with quotes (look-ahead).
    # Added Vietnamese uppercase characters to look-ahead to handle non-English sentences.
    upper = r'[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỴỶỸĐ]'
    pattern = (
        fr'(?<=[.!?])\s+(?=["\']?{upper})|'
        fr'(?<=[.!?]["\'])\s+(?=["\']?{upper})|'
        fr'(?<!\b[A-Z][.!?])(?<=[.!?])(?=["\']?{upper})|'
        fr'(?<!\b[A-Z][.!?])(?<=[.!?]["\'])(?=["\']?{upper})|'
        r'(?<=;)\s*|'
        r'(?<=--)\s*'
    )
    sents = re.split(pattern, text)
    
    cleaned = []
    for s in sents:
        s = s.strip()
        # Strip leading ellipses (...) and optional space
        s = re.sub(r'^\.\.\.\s*|^\.\.\s*', '', s)
        if s:
            cleaned.append(s)
    return cleaned


def aggressive_split(text: str) -> List[str]:
    """
    Even more aggressive sentence splitter.
    Splits strictly on . ? ! " ] followed by any whitespace.
    """
    # Split on punctuation followed by whitespace
    sents = re.split(r'(?<=[.?!"\]])\s+', text)
    
    cleaned = []
    for s in sents:
        s = s.strip()
        if s:
            cleaned.append(s)
    return cleaned


def parse_and_check(text: str) -> Tuple[bool, str, Any]:
    """
    Parses a single sentence string and checks for multi-sentence issues.
    
    Args:
        text: The raw sentence string to parse.
        
    Returns:
        Tuple containing:
        - is_multi_sentence (bool): True if the parser detected multiple sentences.
        - penman_string (str): The generated Penman AMR block.
        - amr_object (Any): The raw AMR object for further processing.
    """
    if not text.strip():
        return False, "", None
    
    tokenized = parser.tokenize(text)[0]
    try:
        _, machines = parser.parse_sentences([tokenized])
        amr = machines[0].get_amr()
        penman = amr.to_penman(jamr=True, isi=False)
        is_multi = "multi-sentence" in penman
    except Exception as e:
        # If the parser generates a completely broken graph (e.g. missing root)
        # or fails internally during parse_sentences, treat it as a failed parse.
        return True, "", None
    
    # If not multi-sentence, collect vocab
    if not is_multi:
        # 1. Collect relations
        for _, rel, _ in amr.edges:
            all_unique_relations.add(rel)
        # 2. Collect concepts
        for concept in amr.nodes.values():
            if not concept or (concept.startswith('"') and concept.endswith('"')):
                continue
            if is_number(concept):
                continue
            
            is_frame = bool(re.search(r'-[0-9]{2,}$', concept))
            is_structural = concept in ['person', 'thing', 'date-entity', 'government-organization', 'amr-unknown']
            
            if is_frame or is_structural:
                all_unique_concepts.add(concept)
            else:
                all_ignored_concepts.add(concept)
                
    return is_multi, penman, amr


def split_and_report(line: str, title: str, line_idx: int) -> List[str]:
    """
    Splits a line into sentences and prints a report if multiple sentences are found.
    Now includes the line index in the report.
    """
    sents = split_sentences(line)
    if len(sents) > 1:
        print(f"\n[MULTI-SENTENCE] Document: {title} (Line {line_idx})")
        print(f"Original line: {line}")
        for i, s in enumerate(sents):
            print(f"  Sent {i+1}: {s}")
    return sents


MAX_LINES = 250
MAX_WORDS = 5000
SKIPPED_DOCS = set()  # Tracks documents to skip: "prefix:doc_idx"

TOTAL_ORIG_DOCS = 0
TOTAL_ORIG_LINES = 0


def get_output_dir_name(basename):
    if basename == "train.tags.en-vi.en" or basename == "train.tags.en-vi.vi":
        return "train_en-vi." + basename.split('.')[-1]
    
    # For XML files or others
    # IWSLT15.TED.tst2015.en-vi.en.xml -> tst2015.en-vi.en
    if basename.endswith(".xml"):
        name_no_ext = os.path.splitext(basename)[0]
        parts = name_no_ext.split(".", 2)
        if len(parts) == 3 and parts[1].upper() == "TED":
            return parts[2]
        return name_no_ext
    
    return basename.replace(".tags.", "_")


def process_parallel_lines(line_en: str, line_vi: str, title: str, line_idx: Any, filename_en: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits parallel English and Vietnamese lines into aligned sentences using 
    recursive AMR-parser feedback.
    
    Strategy:
    1. Level 0: Try parsing the English line as a single unit. If valid, return it.
    2. Level 1: If 'multi-sentence' is detected, split both EN and VI using regex.
    3. Alignment Check: If split counts mismatch, drop the whole line.
    4. Level 2: If any EN segment still has 'multi-sentence', split aggressively on punctuation.
    5. Final Check: If multi-sentence persists after Level 2, drop the line.
    
    Args:
        line_en: Raw English text line.
        line_vi: Raw Vietnamese text line.
        title: Document title (for logging).
        line_idx: Index of the line within the document.
        filename_en: Name of the source file.
        
    Returns:
        Tuple of three lists: (final_en_sentences, final_vi_sentences, final_amr_blocks).
    """
    # Clean inputs: replace newlines/multiple spaces with single space
    line_en = " ".join(line_en.split())
    line_vi = " ".join(line_vi.split())

    if not line_en:
        return [], [], []

    # Level 0: Try parsing as is
    is_multi, penman, _ = parse_and_check(line_en)
    if not is_multi:
        return [line_en], [line_vi], [penman]

    # Level 1: Split using current regex
    sents_en = split_sentences(line_en)
    sents_vi = split_sentences(line_vi)

    if len(sents_en) != len(sents_vi):
        # Mismatch after regex split
        print(f"\n[DROPPED] Mismatch after Regex split: {filename_en} (Line {line_idx})")
        print(f"  EN ({len(sents_en)} sents): {line_en}")
        print(f"  VI ({len(sents_vi)} sents): {line_vi}")
        return [], [], []

    final_en = []
    final_vi = []
    final_amr = []

    for s_en, s_vi in zip(sents_en, sents_vi):
        # Check if Level 1 sentences still have multi-sentence issues
        is_multi_1, penman_1, _ = parse_and_check(s_en)
        if not is_multi_1:
            final_en.append(s_en)
            final_vi.append(s_vi)
            final_amr.append(penman_1)
        else:
            # Level 2: Aggressive Split
            agg_en = aggressive_split(s_en)
            agg_vi = aggressive_split(s_vi)

            if len(agg_en) != len(agg_vi):
                print(f"\n[DROPPED] Mismatch after Aggressive split: {filename_en} (Line {line_idx})")
                print(f"  EN Sub: {s_en}")
                print(f"  VI Sub: {s_vi}")
                return [], [], []

            for a_en, a_vi in zip(agg_en, agg_vi):
                is_multi_2, penman_2, _ = parse_and_check(a_en)
                if is_multi_2:
                    # FAILURE: Still multi-sentence after aggressive split - Drop the whole line
                    print(f"\n[DROPPED] 'multi-sentence' persists after aggressive split: {filename_en} (Line {line_idx})")
                    print(f"  Offending EN segment: {a_en}")
                    return [], [], []
                
                final_en.append(a_en)
                final_vi.append(a_vi)
                final_amr.append(penman_2)

    return final_en, final_vi, final_amr


def extract_documents_parallel(filepath_en: str, filepath_vi: str) -> None:
    """
    Extracts documents from parallel IWSLT .tags files.
    
    This function reads the entire raw content of the .tags files, splits them 
    into <talk> blocks, and processes each talk as a separate document. 
    It handles title/description/transcript extraction and writes results to 
    the documented output folders.
    """
    basename_en = os.path.basename(filepath_en)
    basename_vi = os.path.basename(filepath_vi)
    
    out_dir_en_name = get_output_dir_name(basename_en)
    out_dir_vi_name = get_output_dir_name(basename_vi)
    
    prefix = re.sub(r'\.(en|vi)$', '', out_dir_en_name)
    
    output_dir_en = os.path.join("output_dataset_doc", out_dir_en_name)
    output_dir_vi = os.path.join("output_dataset_doc", out_dir_vi_name)
    
    # Clear existing files to avoid stale data
    for d in [output_dir_en, output_dir_vi]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    with open(filepath_en, 'r', encoding='utf-8') as f:
        content_en = f.read()
    with open(filepath_vi, 'r', encoding='utf-8') as f:
        content_vi = f.read()

    talks_en = content_en.split('<url>')
    talks_vi = content_vi.split('<url>')

    if len(talks_en) != len(talks_vi):
        print(f"Warning: Different number of talks in {basename_en} and {basename_vi}")

    global TOTAL_ORIG_DOCS, TOTAL_ORIG_LINES
    
    total_docs = 0
    total_skipped = 0
    
    for doc_idx, (talk_en, talk_vi) in enumerate(zip(talks_en[1:], talks_vi[1:]), start=1):
        TOTAL_ORIG_DOCS += 1
        # Extract titles/desc
        t_match_en = re.search(r'<title>(.*?)</title>', talk_en, re.DOTALL)
        t_match_vi = re.search(r'<title>(.*?)</title>', talk_vi, re.DOTALL)
    for doc_idx, (talk_en, talk_vi) in enumerate(zip(talks_en[1:], talks_vi[1:]), start=1):
        TOTAL_ORIG_DOCS += 1
        # Extract title
        t_match_en = re.search(r'<title>(.*?)</title>', talk_en, re.DOTALL)
        title_en = t_match_en.group(1).strip() if t_match_en else f"doc-{doc_idx}"
        
        # Split talk into lines, ignoring blank lines and XML tags
        lines_en = [l.strip() for l in talk_en.split('\n') if l.strip() and not l.strip().startswith('<')]
        lines_vi = [l.strip() for l in talk_vi.split('\n') if l.strip() and not l.strip().startswith('<')]

        parts_en = []
        parts_vi = []
        parts_amr = []

        if len(lines_en) == len(lines_vi):
            TOTAL_ORIG_LINES += len(lines_en)
            for i, (l_en, l_vi) in enumerate(zip(lines_en, lines_vi), start=1):
                s_en, s_vi, s_amr = process_parallel_lines(l_en, l_vi, title_en, i, basename_en)
                parts_en.extend(s_en)
                parts_vi.extend(s_vi)
                parts_amr.extend(s_amr)
        else:
            print(f"Warning: Doc {doc_idx} has mismatched lines (EN:{len(lines_en)}, VI:{len(lines_vi)}). Skipping.")

        doc_en = "\n".join(parts_en)
        doc_vi = "\n".join(parts_vi)
        doc_amr = "\n\n".join(p.strip() for p in parts_amr if p.strip())

        if doc_en or doc_vi:
            skip_key = f"{prefix}:{doc_idx}"
            
            # Check limits on EN
            line_count = len(doc_en.split('\n'))
            word_count = sum(len(line.split()) for line in doc_en.split('\n'))
            if line_count > MAX_LINES or word_count > MAX_WORDS:
                print(f"Dropped oversized document: {skip_key} (Lines: {line_count}, Words: {word_count})")
                total_skipped += 1
                continue

            with open(os.path.join(output_dir_en, f"doc-{doc_idx}.txt"), 'w', encoding='utf-8') as out:
                out.write(doc_en)
            with open(os.path.join(output_dir_vi, f"doc-{doc_idx}.txt"), 'w', encoding='utf-8') as out:
                out.write(doc_vi)
            
            # Mirror to output_dataset_amr
            amr_out_folder = os.path.join("output_dataset_amr", out_dir_en_name)
            os.makedirs(amr_out_folder, exist_ok=True)
            with open(os.path.join(amr_out_folder, f"doc-{doc_idx}.txt"), 'w', encoding='utf-8') as out:
                out.write(doc_amr + "\n\n" if doc_amr.strip() else "")

            total_docs += 1

    print(f"Total for '{prefix}': {total_docs} documents created, {total_skipped} documents dropped.")


def extract_xml_documents_parallel(filepath_en: str, filepath_vi: str) -> None:
    """
    Extracts documents from parallel IWSLT .xml files.
    
    This function processes XML-formatted datasets (like dev/test sets).
    It identifies <doc> blocks and processes the contained text lines 
    using the unified segmentation and parsing pipeline.
    """
    basename_en = os.path.basename(filepath_en)
    basename_vi = os.path.basename(filepath_vi)
    
    out_dir_en_name = get_output_dir_name(basename_en)
    out_dir_vi_name = get_output_dir_name(basename_vi)
    
    prefix = re.sub(r'\.(en|vi)$', '', out_dir_en_name)
    
    output_dir_en = os.path.join("output_dataset_doc", out_dir_en_name)
    output_dir_vi = os.path.join("output_dataset_doc", out_dir_vi_name)

    # Clear existing files to avoid stale data
    for d in [output_dir_en, output_dir_vi]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    with open(filepath_en, 'r', encoding='utf-8') as f:
        content_en = f.read()
    with open(filepath_vi, 'r', encoding='utf-8') as f:
        content_vi = f.read()

    doc_blocks_en = re.findall(r'<doc\b[^>]*>(.*?)</doc>', content_en, re.DOTALL)
    doc_blocks_vi = re.findall(r'<doc\b[^>]*>(.*?)</doc>', content_vi, re.DOTALL)

    if len(doc_blocks_en) != len(doc_blocks_vi):
        print(f"Warning: Different number of docs in {basename_en} and {basename_vi}")

    global TOTAL_ORIG_DOCS, TOTAL_ORIG_LINES
    
    total_docs = 0
    total_skipped = 0

    for doc_idx, (block_en, block_vi) in enumerate(zip(doc_blocks_en, doc_blocks_vi), start=1):
        TOTAL_ORIG_DOCS += 1
        title_en = (re.search(r'<title>(.*?)</title>', block_en, re.DOTALL) or type('m', (), {'group': lambda x: ""})()).group(1).strip()
        
        # Extract text from <seg> tags
        lines_en = [s.strip() for s in re.findall(r'<seg[^>]*>(.*?)</seg>', block_en, re.DOTALL) if s.strip()]
        lines_vi = [s.strip() for s in re.findall(r'<seg[^>]*>(.*?)</seg>', block_vi, re.DOTALL) if s.strip()]

        parts_en = []
        parts_vi = []
        parts_amr = []

        if len(lines_en) == len(lines_vi):
            TOTAL_ORIG_LINES += len(lines_en)
            for i, (l_en, l_vi) in enumerate(zip(lines_en, lines_vi), start=1):
                s_en, s_vi, s_amr = process_parallel_lines(l_en, l_vi, title_en, i, basename_en)
                parts_en.extend(s_en)
                parts_vi.extend(s_vi)
                parts_amr.extend(s_amr)

        doc_en = "\n".join(parts_en)
        doc_vi = "\n".join(parts_vi)
        doc_amr = "\n\n".join(p.strip() for p in parts_amr if p.strip())

        if doc_en or doc_vi:
            skip_key = f"{prefix}:{doc_idx}"
            
            # Check limits on EN
            line_count = len(doc_en.split('\n'))
            word_count = sum(len(line.split()) for line in doc_en.split('\n'))
            if line_count > MAX_LINES or word_count > MAX_WORDS:
                print(f"Dropped oversized document: {skip_key} (Lines: {line_count}, Words: {word_count})")
                total_skipped += 1
                continue

            with open(os.path.join(output_dir_en, f"doc-{doc_idx}.txt"), 'w', encoding='utf-8') as out:
                out.write(doc_en)
            with open(os.path.join(output_dir_vi, f"doc-{doc_idx}.txt"), 'w', encoding='utf-8') as out:
                out.write(doc_vi)

            # Mirror to output_dataset_amr
            amr_out_folder = os.path.join("output_dataset_amr", out_dir_en_name)
            os.makedirs(amr_out_folder, exist_ok=True)
            with open(os.path.join(amr_out_folder, f"doc-{doc_idx}.txt"), 'w', encoding='utf-8') as out:
                out.write(doc_amr + "\n\n")

            total_docs += 1

    print(f"Total for '{prefix}': {total_docs} documents created, {total_skipped} documents dropped.")


def check_line_counts():
    """
    Checks if parallel documents have the same number of lines.
    If not, deletes both and reports.
    Returns (total_remaining_docs, total_remaining_lines)
    """
    base_dir = "output_dataset_doc"
    if not os.path.exists(base_dir):
        return 0, 0

    dirs = sorted(os.listdir(base_dir))
    en_dirs = [d for d in dirs if d.endswith(".en")]
    
    print("\n--- Running Line Count Parity Check ---")
    
    total_final_docs = 0
    total_final_lines = 0

    for en_dir in en_dirs:
        # Correctly pair by replacing only the trailing .en with .vi
        vi_dir = en_dir[:-3] + ".vi"
        if vi_dir not in dirs:
            continue
        
        en_path = os.path.join(base_dir, en_dir)
        vi_path = os.path.join(base_dir, vi_dir)
        
        en_files = sorted(os.listdir(en_path))
        for f in en_files:
            if not f.startswith("doc-"):
                continue
            
            f_en = os.path.join(en_path, f)
            f_vi = os.path.join(vi_path, f)
            
            if not os.path.exists(f_vi):
                print(f"Missing parallel file: {f_vi}")
                continue
            
            with open(f_en, 'r', encoding='utf-8') as fe:
                lines_en = fe.readlines()
            with open(f_vi, 'r', encoding='utf-8') as fv:
                lines_vi = fv.readlines()
            
            if len(lines_en) != len(lines_vi):
                print(f"[PARITY ERROR] Deleting mismatched pair: {en_dir}/{f} (EN:{len(lines_en)}, VI:{len(lines_vi)})")
                os.remove(f_en)
                os.remove(f_vi)
            else:
                total_final_docs += 1
                total_final_lines += len(lines_en)
    
    return total_final_docs, total_final_lines


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        input_folder = "input_datasets"

    if not os.path.isdir(input_folder):
        print(f"Directory '{input_folder}' not found. Please create it and add files, or specify a valid directory.")
        sys.exit(1)

    print("Parsing files in parallel and ensuring line alignment...")

    # Group files by their base prefix
    files = sorted(os.listdir(input_folder))
    file_groups = {}
    
    for f in files:
        if f.endswith(".en.xml") or f.endswith(".vi.xml"):
            lang = "en" if f.endswith(".en.xml") else "vi"
            ext = ".en.xml" if lang == "en" else ".vi.xml"
            prefix = f[:-len(ext)]
            file_groups.setdefault(prefix, {})[lang] = f
        elif f.endswith(".en") or f.endswith(".vi"):
            lang = "en" if f.endswith(".en") else "vi"
            ext = ".en" if lang == "en" else ".vi"
            prefix = f[:-len(ext)]
            file_groups.setdefault(prefix, {})[lang] = f

    for prefix, pair in sorted(file_groups.items()):
        if "en" not in pair or "vi" not in pair:
            print(f"Warning: Skipping {prefix} as it doesn't have both EN and VI files.")
            continue
        
        file_en = pair["en"]
        file_vi = pair["vi"]
        path_en = os.path.join(input_folder, file_en)
        path_vi = os.path.join(input_folder, file_vi)
        
        print(f"\n=== Processing Pair: {file_en} & {file_vi} ===")
        
        if file_en.endswith(".xml"):
            extract_xml_documents_parallel(path_en, path_vi)
        else:
            extract_documents_parallel(path_en, path_vi)

    # Final parity check
    final_docs, final_lines = check_line_counts()

    print("\n" + "="*40)
    print("      FINAL EXTRACTION SUMMARY")
    print("="*40)
    print(f"Total original documents:      {TOTAL_ORIG_DOCS}")
    print(f"Total original lines:          {TOTAL_ORIG_LINES}")
    print("-" * 40)
    print(f"Total documents after processing: {final_docs}")
    print(f"Total lines after processing:     {final_lines}")
    
    # Save results summary
    print(f"\nExtraction complete.")
    
    # ------------------------------------------------------------------
    # Save all unique relations and concepts to amrs_token.json
    # ------------------------------------------------------------------
    root_dir = os.path.dirname(os.path.abspath(__file__))
    combined_original = all_unique_relations | all_unique_concepts
    
    # Filter out anything already in BERT vocab
    combined_original = {x for x in combined_original if x not in bert_vocab}
    
    rel_file_path = os.path.join(root_dir, "amrs_token.json")
    
    print(f"\nSaving {len(combined_original)} items (relations + concepts) to {rel_file_path}...")
    with open(rel_file_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(combined_original)), f, indent=4)

    # ------------------------------------------------------------------
    # Save simple unique relations and concepts (without trailing numbers)
    # ------------------------------------------------------------------
    simple_items = set()
    
    # Simplify relations: e.g., :ARG1 -> :ARG, :snt1 -> :snt
    for rel in all_unique_relations:
        simple_rel = re.sub(r'\d+(?=-of$|$)', '', rel)
        if simple_rel not in bert_vocab:
            simple_items.add(simple_rel)
        
    # Simplify concepts: e.g., say-01 -> say
    for concept in all_unique_concepts:
        simple_concept = re.sub(r'-\d+$', '', concept)
        if simple_concept not in bert_vocab:
            simple_items.add(simple_concept)
        
    simple_rel_file_path = os.path.join(root_dir, "amrs_token_simple.json")
    print(f"Saving {len(simple_items)} simple items to {simple_rel_file_path}...")
    with open(simple_rel_file_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(simple_items)), f, indent=4)
    
    if all_ignored_concepts:
        print(f"\nFound {len(all_ignored_concepts)} ignored concepts (not frames or structural):")
        # Print a sample if too many
        sample_ignored = sorted(list(all_ignored_concepts))
        if len(sample_ignored) > 50:
            print(", ".join(sample_ignored[:50]) + " ...")
        else:
            print(", ".join(sample_ignored))

    print("\nAll done.")
    print("="*40)

    print("\nProcessing complete! All parallel files have been aligned.")