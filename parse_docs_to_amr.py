"""
parse_docs_to_amr.py

Reads all folders ending in ".en" inside `output_dataset_doc/`, parses every
document file sentence-by-sentence using the transition-amr-parser, and writes
the Penman AMR output to a mirrored folder structure inside `output_dataset_amr/`.

Output format (matches doc_sen.amr convention):
  - One Penman block per sentence
  - A single blank line between consecutive sentence blocks

Usage:
    python parse_docs_to_amr.py [input_base_dir] [output_base_dir]

Defaults:
    input_base_dir  = output_dataset_doc
    output_base_dir = output_dataset_amr
"""

import os
from typing import List, Set
import sys
import json
import re

from transition_amr_parser.parse import AMRParser


# Global set to collect all unique relations (labels) across all parsed documents
all_unique_relations: Set[str] = set()


# ---------------------------------------------------------------------------
# Load the parser once (expensive – do NOT reload inside loops)
# ---------------------------------------------------------------------------
print("Loading AMR parser model (AMR3-structbart-L)...")
parser = AMRParser.from_pretrained('AMR3-structbart-L')
print("Parser ready.\n")


def parse_sentences(sentences: List[str], batch_size: int = 32) -> List[str]:
    """
    Takes a list of raw sentence strings and returns a list of Penman AMR strings.
    Sentences are processed in chunks of `batch_size` to avoid CUDA OOM on large documents.
    Empty/whitespace-only strings are skipped and returned as empty strings.
    """
    # Separate non-empty sentences, tracking original positions
    indexed = [(i, s) for i, s in enumerate(sentences) if s.strip()]
    result = [""] * len(sentences)

    if not indexed:
        return result

    # Process in chunks    
    for chunk_start in range(0, len(indexed), batch_size):
        chunk = indexed[chunk_start: chunk_start + batch_size]
        indices, non_empty = zip(*chunk)

        tokenized = [parser.tokenize(s)[0] for s in non_empty]
        _, machines = parser.parse_sentences(tokenized)

        for idx, machine in zip(indices, machines):
            amr = machine.get_amr()
            result[idx] = amr.to_penman(jamr=True, isi=False)
            
            # Collect all relation labels from the AMR edges
            for _, rel, _ in amr.edges:
                all_unique_relations.add(rel)

    return result


def process_document(doc_path: str, out_path: str) -> None:
    """
    Reads a single document file, parses each non-empty line as a sentence,
    and writes the Penman AMR blocks separated by blank lines to out_path.
    """
    with open(doc_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    # Each non-empty line is treated as one sentence
    sentences = [line for line in lines if line.strip()]

    if not sentences:
        print(f"  [SKIP] {doc_path} – no sentences found.")
        return

    print(f"  Parsing {len(sentences)} sentences from {os.path.basename(doc_path)} "
          f"(~{-(-len(sentences)//32)} chunk(s)) ...")
    amr_blocks = parse_sentences(sentences)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for penman in amr_blocks:
            if penman.strip():
                f.write(penman.strip())
                f.write("\n\n")   # blank line between sentence blocks

    print(f"  -> Written: {out_path}")


def process_folder(en_folder: str, out_folder: str) -> None:
    """
    Processes all .txt files inside an .en doc folder.
    """
    def _doc_num(filename: str) -> int:
        # Extract the integer from "doc-N.txt" for numeric sorting
        try:
            return int(filename.split('-')[1].split('.')[0])
        except (IndexError, ValueError):
            return 0

    doc_files = sorted(
        (f for f in os.listdir(en_folder)
         if f.endswith(".txt") and os.path.isfile(os.path.join(en_folder, f))),
        key=_doc_num
    )

    if not doc_files:
        print(f"  [WARN] No .txt files found in {en_folder}")
        return

    print(f"\n=== Folder: {os.path.basename(en_folder)} – {len(doc_files)} document(s) ===")
    os.makedirs(out_folder, exist_ok=True)

    for doc_file in doc_files:
        doc_path = os.path.join(en_folder, doc_file)
        out_path = os.path.join(out_folder, doc_file)
        process_document(doc_path, out_path)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Resolve directories
    # ------------------------------------------------------------------
    input_base  = sys.argv[1] if len(sys.argv) > 1 else "output_dataset_doc"
    output_base = sys.argv[2] if len(sys.argv) > 2 else "output_dataset_amr"

    if not os.path.isdir(input_base):
        print(f"Input directory '{input_base}' not found.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Find all folders ending with ".en" inside input_base
    # ------------------------------------------------------------------
    en_folders = sorted(
        d for d in os.listdir(input_base)
        if d.endswith(".en") and os.path.isdir(os.path.join(input_base, d))
    )

    if not en_folders:
        print(f"No folders ending in '.en' found inside '{input_base}'.")
        sys.exit(0)

    print(f"Found {len(en_folders)} English folder(s): {en_folders}\n")

    # ------------------------------------------------------------------
    # Process each .en folder
    # ------------------------------------------------------------------
    for folder_name in en_folders:
        en_folder  = os.path.join(input_base, folder_name)
        out_folder = os.path.join(output_base, folder_name)
        process_folder(en_folder, out_folder)

    print("\nAll documents parsed. AMR files are in:", output_base)

    # ------------------------------------------------------------------
    # Save all unique relations to custom_relation_amrs.json in the root folder
    # ------------------------------------------------------------------
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path = os.path.join(root_dir, "custom_relation_amrs.json")
    
    print(f"Saving {len(all_unique_relations)} unique relations to {rel_file_path}...")
    with open(rel_file_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_unique_relations)), f, indent=4)

    # ------------------------------------------------------------------
    # Save simple unique relations (without trailing numbers)
    # ------------------------------------------------------------------
    simple_relations = set()
    for rel in all_unique_relations:
        simple_rel = re.sub(r'\d+(?=-of$|$)', '', rel)
        simple_relations.add(simple_rel)
        
    simple_rel_file_path = os.path.join(root_dir, "custom_relation_amrs_simple.json")
    print(f"Saving {len(simple_relations)} simple relations to {simple_rel_file_path}...")
    with open(simple_rel_file_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(simple_relations)), f, indent=4)
    
    print("Done.")
