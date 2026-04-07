import os
import torch
import sys
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

def is_title_line(sentence):
    """
    Detects Title Lines while ignoring Q&A dialogue tags.
    """
    if " : " not in sentence:
        return False
        
    prefix = sentence.split(" : ")[0].strip()
    words = prefix.split()
    if not words:
        return False
        
    # Reject if the prefix is too long to reasonably be a name/speaker list
    if len(words) > 10:
        return False
        
    # Reject conversational prefixes that mimic speaker tags
    first_word = words[0].lower()
    conversational_starters = {
        "and", "but", "so", "now", "the", "my", "i", "we", "he", "she", "they", 
        "what", "which", "why", "it", "this", "another", "or", "then", "first", 
        "second", "third", "number", "a", "an", "here", "there", "because", "when", 
        "how", "if", "well", "yes", "no", "that", "these", "those", "one", "two", 
        "three", "some", "someone", "everyone", "nobody", "anybody", "something",
        "let's", "let", "education", "question", "questions", "enigma"
    }
    if first_word in conversational_starters:
        return False

    # True title lines rarely end with periods or quotes.
    # We explicitly allow '?' and '!' since many TED talks have them in the title.
    if sentence.strip().endswith((".", '"', "&quot;")):
        return False
            
    return True

def find_boundaries_from_english(filepath, model, min_sentences=30, window_size=5, sim_threshold=0.15):
    """
    Calculates boundaries based solely on an English file.
    Since SentenceTransformer (all-MiniLM-L6-v2) is optimized for English, this provides the most reliable semantic boundary detection.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        # DO NOT exclude empty lines! Excluding them destroys the 1-to-1 parallel indexing
        sentences = [line.strip() for line in f.readlines()]

    # First, let's determine if this file uses the explicit "Speaker Name : Title" formatting
    title_lines_count = sum(1 for s in sentences if is_title_line(s))
    has_title_lines = title_lines_count > 10

    if has_title_lines:
        print(f"Detected {title_lines_count} Title Lines in {filepath}! Falling back to strict Title Line segmentation (Disabling AI Safety Net).")
    else:
        print(f"Embedding {len(sentences)} sentences from {filepath} for AI safety net...")
        embeddings = model.encode(sentences, convert_to_tensor=True)

    # Common TED sign-off keywords (English only)
    sign_off_keywords = [
        "thank you", 
        "thanks"
    ]

    boundaries = [0] # Initialize with the start of the file
    last_boundary = 0
    
    print("Scanning for boundaries based on English text...")
    for i in range(15, len(sentences) - window_size):
        
        # Trigger A1: The Title Line (Highly reliable for train.en)
        if (i - last_boundary >= 15) and is_title_line(sentences[i]):
            # Split BEFORE the title line so the new document gets the title
            boundaries.append(i)
            last_boundary = i
            continue

        # If it's a strongly structured file (like train.en), DO NOT allow "Thank you" or semantic shift 
        # to ruin the segmentation. We ONLY split on title lines!
        if has_title_lines:
            continue

        current_sentence_lower = sentences[i].lower()
        is_sign_off = any(keyword in current_sentence_lower for keyword in sign_off_keywords)
        
        # Trigger A2: The "Thank You" Cut (Highly reliable for tst2013 where titles are stripped)
        if is_sign_off and (i - last_boundary >= 15):
            boundaries.append(i + 1)
            last_boundary = i + 1
            continue
            
        if i - last_boundary < min_sentences:
            continue
            
        # Trigger B: Semantic Shift
        block_before = torch.mean(embeddings[i - window_size : i], dim=0)
        block_after = torch.mean(embeddings[i : i + window_size], dim=0)
        sim = util.cos_sim(block_before, block_after).item()

        if sim < sim_threshold:
            boundaries.append(i)
            last_boundary = i

    # Cap the final boundary at the end of the file
    if boundaries[-1] != len(sentences):
        boundaries.append(len(sentences))

    return boundaries, len(sentences)


def extract_and_save_documents(boundaries, target_filepath, expected_total_lines):
    """
    Given a set of pre-calculated boundaries, slice a target parallel file and save the segments.
    """
    with open(target_filepath, 'r', encoding='utf-8') as f:
        # DO NOT exclude empty lines! Excluding them destroys the 1-to-1 parallel indexing
        # if the English file has a blank line where the Vietnamese file has a typo/stray character.
        sentences = [line.strip() for line in f.readlines()]
        
    if len(sentences) != expected_total_lines:
        print(f"WARNING: Line count mismatch! {target_filepath} has {len(sentences)} lines, but English had {expected_total_lines}.")
        # To avoid out of bounds errors, we cap the boundaries safely
        boundaries = [min(b, len(sentences)) for b in boundaries]

    basename = os.path.basename(target_filepath)
    safe_basename = basename.replace(".", "_")
    output_dir = f"{safe_basename}_docs"

    os.makedirs(output_dir, exist_ok=True)

    # Extract and Save the Documents
    print(f"\n--- Extraction Results for folder: {output_dir} ---")
    total_docs = 0
    total_words = 0

    for idx in range(len(boundaries) - 1):
        start_idx = boundaries[idx]
        end_idx = boundaries[idx+1]
        
        if start_idx >= end_idx:
            continue
            
        doc_sentences = sentences[start_idx:end_idx]
        
        out_file = os.path.join(output_dir, f"doc_{idx+1:03d}.txt")
        with open(out_file, 'w', encoding='utf-8') as out:
            out.write("\n".join(doc_sentences))
            
        num_sentences = len(doc_sentences)
        num_words = sum(len(sentence.split()) for sentence in doc_sentences)
        
        total_docs += 1
        total_words += num_words
            
        print(f"  File doc_{idx+1:03d}.txt: {num_sentences} sentences, {num_words} words.")

    print(f"Total for '{output_dir}': {total_docs} document files, {total_words} words.\n")


if __name__ == "__main__":
    # Get input directory from command line if provided, else default to "input_datasets"
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        input_folder = "input_datasets"

    if not os.path.isdir(input_folder):
        print(f"Directory {input_folder} not found. Please create it and add files, or specify a valid directory.")
        sys.exit(1)

    print("Loading SentenceTransformer model (English optimized) for the whole dataset...")
    # Load once for efficiency
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Group parallel files by their base name (e.g., 'tst2013' -> ['tst2013.en', 'tst2013.vi'])
    file_groups = defaultdict(list)
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        if os.path.isfile(filepath):
            base = os.path.splitext(filename)[0]
            file_groups[base].append(filepath)

    for base, paths in file_groups.items():
        # Find the English file to use as our "anchor" for boundary calculation
        en_file = next((p for p in paths if p.endswith('.en')), None)
        
        if not en_file:
            print(f"\nSkipping group '{base}' because no .en file was found to calculate boundaries.")
            continue
            
        print(f"\n=== Processing parallel text group: {base} ===")
        # 1. Calculate boundaries exactly once using the English file
        boundaries, total_lines = find_boundaries_from_english(en_file, model)
        
        # 2. Slice all files in this group (including the .en file itself) using those EXACT timestamps/lines
        for target_filepath in paths:
            extract_and_save_documents(boundaries, target_filepath, total_lines)

    print("\nParallel Segmentation complete! All files have been synchronized and are ready for transition-amr-parser.")