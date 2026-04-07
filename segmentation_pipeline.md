# IWSLT Data to Documents Segmentation Pipeline

This document explains the sentence-to-document segmentation pipeline implemented in `segment_iwslt_data_to_docs.py`.

## Overview
The goal of this pipeline is to split large, contiguous datasets of parallel sentences (like IWSLT `.en` and `.vi` datasets) into smaller, coherent document files (`doc_001.txt`, `doc_002.txt`, etc.). It maintains the critical 1-to-1 parallel sentence alignment between different languages by always using the English file (`.en`) as the specific anchor point for computing document boundaries.

## Key Mechanisms

The script utilizes a few reliable triggers to determine where one document (like a TED talk) ends and a new one begins:

1. **Title Line Segmentation (Trigger A1)**: 
   - Strongly structured files (such as `train.en`) often begin documents with a title line (e.g., `Speaker Name : Title`).
   - If the script detects numerous title lines, it switches to a **strict** title line segmentation mode, which takes precedence. It splits documents specifically at these title transitions.

2. **Sign-off Cutoffs (Trigger A2)**:
   - For datasets where titles are stripped out (like `tst2013.en`), the script looks for common TED talk sign-off phrases in English (e.g., `"thank you"`, `"thanks"`).

3. **Semantic Shift Safety Net (Trigger B)**:
   - To handle sections lacking explicit titles or sign-offs, the script falls back to an AI safety net using the `all-MiniLM-L6-v2` SentenceTransformer. 
   - By calculating the cosine similarity of text windows before and after a specific line, the script can pinpoint abrupt topical shifts indicating a new document boundary.

## How it works (Parallel Alignment)
Instead of processing each file independently, the script:
1. Groups related parallel files by their **basename** (e.g., `tst2013.en` and `tst2013.vi` are grouped).
2. Computes the boundaries exclusively using the `.en` file.
3. Iterates over all files in the group (e.g., `.vi` files) and segments them using those EXACT computed boundaries, strictly preserving index lengths and empty lines.

## Directory Example Output

Here is a visual example of how the pipeline processes the data.

### 1: Input Setup
You run the script targeting an input directory containing your unsegmented language files:
```bash
python segment_iwslt_data_to_docs.py input_datasets
```

Your `input_datasets` folder structure:
```text
input_datasets/
├── train.en
├── train.vi
├── tst2013.en
└── tst2013.vi
```

### 2: Output
For every inputted text file, a matching safe-basename folder is created in your root directory containing the document segments.

```text
train_en_docs/
├── doc_001.txt
├── doc_002.txt
├── ...
└── doc_133.txt

train_vi_docs/
├── doc_001.txt
├── doc_002.txt
├── ...
└── doc_133.txt

tst2013_en_docs/
├── doc_001.txt
├── ...

tst2013_vi_docs/
├── doc_001.txt
├── ...
```

Notice how `train_en_docs/doc_001.txt` perfectly parallels `train_vi_docs/doc_001.txt` in sentence length, thanks to the exact boundary sharing pattern computed from the `.en` source.
