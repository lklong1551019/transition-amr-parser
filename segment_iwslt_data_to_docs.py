import os
import sys
import re


def extract_documents(filepath):
    """
    Extracts documents from IWSLT .tags.* files.
    Each talk is delimited by <url> tags and contains <title> and <description> header tags,
    followed by plain-text transcript lines.
    """
    basename = os.path.basename(filepath)
    if basename == "train.tags.en-vi.en":
        output_dir_name = "train_en-vi.en"
    elif basename == "train.tags.en-vi.vi":
        output_dir_name = "train_en-vi.vi"
    else:
        output_dir_name = basename.replace(".tags.", "_")

    output_dir = os.path.join("output_dataset_doc", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Talks are separated by <url> tags in IWSLT
    talks_raw = content.split('<url>')

    total_docs = 0
    total_words = 0
    print(f"\n--- Extraction Results for folder: {output_dir} ---")

    doc_idx = 1
    for talk in talks_raw[1:]:  # skip preamble before first <url>
        if not talk.strip():
            continue

        title_match = re.search(r'<title>(.*?)</title>', talk, re.DOTALL)
        desc_match = re.search(r'<description>(.*?)</description>', talk, re.DOTALL)

        title_text = title_match.group(1).strip() if title_match else ""
        desc_text = desc_match.group(1).strip() if desc_match else ""

        # Extract the transcript text (everything after the last header tag)
        last_tag_idx = 0
        for tag in ['</url>', '</keywords>', '</speaker>', '</talkid>', '</title>', '</description>']:
            pos = talk.rfind(tag)
            if pos != -1:
                last_tag_idx = max(last_tag_idx, pos + len(tag))

        raw_transcript = talk[last_tag_idx:]

        # Clean out reviewer/translator tags and self-closing tags
        raw_transcript = re.sub(r'<reviewer.*?>.*?</reviewer>', '', raw_transcript, flags=re.DOTALL)
        raw_transcript = re.sub(r'<translator.*?>.*?</translator>', '', raw_transcript, flags=re.DOTALL)
        raw_transcript = re.sub(r'<[^>]+/>', '', raw_transcript, flags=re.DOTALL)

        transcript_text = raw_transcript.strip()

        parts = []
        if title_text:
            parts.append(title_text)
        if desc_text:
            parts.append(desc_text)
        if transcript_text:
            parts.append(transcript_text)

        doc_content = "\n".join(parts)

        if doc_content:
            out_file = os.path.join(output_dir, f"doc-{doc_idx}.txt")
            with open(out_file, 'w', encoding='utf-8') as out:
                out.write(doc_content)

            num_words = sum(len(line.split()) for line in doc_content.split('\n'))
            total_words += num_words
            total_docs += 1
            doc_idx += 1

    print(f"Total for '{output_dir}': {total_docs} document files, {total_words} words.")


def extract_xml_documents(filepath):
    """
    Extracts documents from IWSLT .xml files.
    Each document is enclosed in <doc ...> ... </doc> tags and contains:
      - <title>  for the talk title
      - <seg id="N"> for each transcript sentence

    Output folder name is derived from the filename, e.g.:
      IWSLT15.TED.tst2015.en-vi.en.xml  ->  tst2015.en-vi.en
    """
    basename = os.path.basename(filepath)

    # Strip leading "IWSLT<year>.TED." prefix and ".xml" extension.
    # e.g. "IWSLT15.TED.tst2015.en-vi.en.xml" -> "tst2015.en-vi.en"
    name_no_ext = os.path.splitext(basename)[0]       # IWSLT15.TED.tst2015.en-vi.en
    parts = name_no_ext.split(".", 2)                  # ["IWSLT15", "TED", "tst2015.en-vi.en"]
    if len(parts) == 3 and parts[1].upper() == "TED":
        output_dir_name = parts[2]                     # tst2015.en-vi.en
    else:
        output_dir_name = name_no_ext                  # fallback

    output_dir = os.path.join("output_dataset_doc", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all <doc ...> ... </doc> blocks
    doc_blocks = re.findall(r'<doc\b[^>]*>(.*?)</doc>', content, re.DOTALL)

    total_docs = 0
    total_words = 0
    print(f"\n--- Extraction Results for folder: {output_dir} ---")

    for doc_idx, block in enumerate(doc_blocks, start=1):
        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', block, re.DOTALL)
        title_text = title_match.group(1).strip() if title_match else ""

        # Extract all <seg> sentences in document order
        seg_texts = re.findall(r'<seg[^>]*>(.*?)</seg>', block, re.DOTALL)
        seg_texts = [s.strip() for s in seg_texts if s.strip()]

        parts = []
        if title_text:
            parts.append(title_text)
        parts.extend(seg_texts)

        doc_content = "\n".join(parts)

        if doc_content:
            out_file = os.path.join(output_dir, f"doc-{doc_idx}.txt")
            with open(out_file, 'w', encoding='utf-8') as out:
                out.write(doc_content)

            num_words = sum(len(line.split()) for line in doc_content.split('\n'))
            total_words += num_words
            total_docs += 1

    print(f"Total for '{output_dir}': {total_docs} document files, {total_words} words.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        input_folder = "input_datasets"

    if not os.path.isdir(input_folder):
        print(f"Directory '{input_folder}' not found. Please create it and add files, or specify a valid directory.")
        sys.exit(1)

    print("Parsing files based on XML tags...")

    for filename in sorted(os.listdir(input_folder)):
        filepath = os.path.join(input_folder, filename)
        if not os.path.isfile(filepath):
            continue

        print(f"\n=== Processing file: {filename} ===")

        if filename.endswith(".xml"):
            extract_xml_documents(filepath)
        else:
            extract_documents(filepath)

    print("\nExtraction complete! All files have been successfully parsed.")