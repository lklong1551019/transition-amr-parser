import re
from datasets import load_dataset

def clean_wikitext_tokens(text):
    """
    Removes WikiText specific tokenization artifacts to restore natural text.
    Essential for clean AMR parsing and diffusion model training.

    Reason:
    WikiText-103 was preprocessed using an older, standardized script (often associated with the Moses tokenizer used in machine translation).
    Before modern subword tokenizers like BPE (Byte-Pair Encoding) became the standard, NLP researchers had to be very explicit about 
    how punctuation was handled.

    The creators wanted to separate general punctuation from words (so the model learns that a comma is an independent entity),
    but they faced a problem with hyphens:

    Sometimes a dash is just punctuation (like an em-dash separating clauses).

    Sometimes a hyphen fundamentally links two words together into a single semantic concept (a compound word).

    To preserve the semantic link of compound words while still treating the hyphen as its own token, they replaced compound hyphens with the special @-@ string.
    This told the language model, "These words are joined," and made it easy to mathematically reverse the process (detokenization) during evaluation.
    """
    # Fix compound hyphens: " co @-@ host " -> " co-host "
    text = re.sub(r'\s*@-@\s*', '-', text)
    # Fix commas in numbers: " 10 @,@ 000 " -> " 10,000 "
    text = re.sub(r'\s*@,@\s*', ',', text)
    # Fix decimals: " 3 @.@ 14 " -> " 3.14 "
    text = re.sub(r'\s*@\.@\s*', '.', text)
    
    return text

def process_and_export_wikitext(split_name="train", output_file="wiki_data_parsed"):
    """
    Loads raw WikiText, reconstructs documents, detokenizes, retains headers,
    and streams the output directly to a formatted text file.
    """
    print(f"Loading raw WikiText-103 ({split_name} split)...")
    raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split_name)

    target_output_file = output_file + "_" + split_name + ".txt"
    print(f"Parsing, cleaning, and writing to {target_output_file}...")
    
    # WikiText delineates main articles with single equals signs: "= Title ="
    article_header_pattern = re.compile(r'^= [^=]+ =$')
    
    current_doc = []
    doc_count = 0
    
    # Open the file in write mode with UTF-8 encoding
    # Python will automatically create the file if it does not already exist as we're using "write" mode.
    with open(target_output_file, 'w', encoding='utf-8') as f:
        for item in raw_dataset:
            line = item['text'].strip()
            
            # Skip empty lines to keep the document dense
            if not line:
                continue
                
            # 1. Apply Detokenization
            line = clean_wikitext_tokens(line)
            
            # 2. Check Document Boundaries (Main Titles)
            if article_header_pattern.match(line):
                # If we have an accumulated document, write it to the file
                if current_doc and len(current_doc) > 5:
                    doc_text = " ".join(l for l in current_doc if l)
                    num_words = len(doc_text.split())
                    num_sentences = len(re.findall(r'[.!?]+(?:\s|$)', doc_text))
                    if num_sentences == 0 and num_words > 0:
                        num_sentences = 1
                    
                    print(f"Document {doc_count + 1}: {num_sentences} sentences, {num_words} words")
                    
                    f.write("\n".join(current_doc))
                    # Separate documents with a clear, undeniable delimiter
                    f.write("\n\n" + "="*80 + "\n\n") 
                    doc_count += 1
                    
                # Reset the list and start the new document without a Title
                current_doc = []
                
            # 3. Handle Sub-sections
            elif line.startswith('='):
                # WikiText uses multiple '=' for deeper sub-sections
                # Add an empty line to separate sub-sections
                if current_doc and current_doc[-1] != "":
                    current_doc.append("")
                
            # 4. Handle Standard Prose
            else:
                current_doc.append(line)
                
        # Catch and write the final document in the loop
        if current_doc and len(current_doc) > 5:
            doc_text = " ".join(l for l in current_doc if l)
            num_words = len(doc_text.split())
            num_sentences = len(re.findall(r'[.!?]+(?:\s|$)', doc_text))
            if num_sentences == 0 and num_words > 0:
                num_sentences = 1
            
            print(f"Document {doc_count + 1}: {num_sentences} sentences, {num_words} words")
            
            f.write("\n".join(current_doc))
            f.write("\n\n" + "="*80 + "\n\n")
            doc_count += 1
            
    print(f"\nProcessing Complete!")
    print(f"Successfully saved {doc_count} heavily formatted documents to '{target_output_file}'")

if __name__ == "__main__":
    # You can change this to "validation" or "test" to process the other splits
    process_and_export_wikitext(split_name="train", output_file="wiki_data_parsed")



#### This ver will print the output with title and section to the file ####
# import re
# from datasets import load_dataset

# def clean_wikitext_tokens(text):
#     """
#     Removes WikiText specific tokenization artifacts to restore natural text.
#     Essential for clean AMR parsing and diffusion model training.

#     Reason:
#     WikiText-103 was preprocessed using an older, standardized script (often associated with the Moses tokenizer used in machine translation).
#     Before modern subword tokenizers like BPE (Byte-Pair Encoding) became the standard, NLP researchers had to be very explicit about 
#     how punctuation was handled.

#     The creators wanted to separate general punctuation from words (so the model learns that a comma is an independent entity),
#     but they faced a problem with hyphens:

#     Sometimes a dash is just punctuation (like an em-dash separating clauses).

#     Sometimes a hyphen fundamentally links two words together into a single semantic concept (a compound word).

#     To preserve the semantic link of compound words while still treating the hyphen as its own token, they replaced compound hyphens with the special @-@ string.
#     This told the language model, "These words are joined," and made it easy to mathematically reverse the process (detokenization) during evaluation.
#     """
#     # Fix compound hyphens: " co @-@ host " -> " co-host "
#     text = re.sub(r'\s*@-@\s*', '-', text)
#     # Fix commas in numbers: " 10 @,@ 000 " -> " 10,000 "
#     text = re.sub(r'\s*@,@\s*', ',', text)
#     # Fix decimals: " 3 @.@ 14 " -> " 3.14 "
#     text = re.sub(r'\s*@\.@\s*', '.', text)
    
#     return text

# def process_and_export_wikitext(split_name="train", output_file="wiki_data_parsed"):
#     """
#     Loads raw WikiText, reconstructs documents, detokenizes, retains headers,
#     and streams the output directly to a formatted text file.
#     """
#     print(f"Loading raw WikiText-103 ({split_name} split)...")
#     raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split_name)

#     target_output_file = output_file + "_" + split_name + ".txt"
#     print(f"Parsing, cleaning, and writing to {target_output_file}...")
    
#     # WikiText delineates main articles with single equals signs: "= Title ="
#     article_header_pattern = re.compile(r'^= [^=]+ =$')
    
#     current_doc = []
#     doc_count = 0
    
#     # Open the file in write mode with UTF-8 encoding
#     # Python will automatically create the file if it does not already exist as we're using "write" mode.
#     with open(target_output_file, 'w', encoding='utf-8') as f:
#         for item in raw_dataset:
#             line = item['text'].strip()
            
#             # Skip empty lines to keep the document dense
#             if not line:
#                 continue
                
#             # 1. Apply Detokenization
#             line = clean_wikitext_tokens(line)
            
#             # 2. Check Document Boundaries (Main Titles)
#             if article_header_pattern.match(line):
#                 # If we have an accumulated document, write it to the file
#                 if current_doc and len(current_doc) > 5:
#                     f.write("\n".join(current_doc))
#                     # Separate documents with a clear, undeniable delimiter
#                     f.write("\n\n" + "="*80 + "\n\n") 
#                     doc_count += 1
                    
#                 # Reset the list and start the new document with a labeled Title
#                 current_doc = []
#                 clean_title = line.replace('=', '').strip()
#                 current_doc.append(f"Title: {clean_title}")
                
#             # 3. Handle Sub-sections
#             elif line.startswith('='):
#                 # WikiText uses multiple '=' for deeper sub-sections (e.g., "= = History = =")
#                 clean_sub = line.replace('=', '').strip()
#                 # Adding a newline before sub-sections makes the text more readable
#                 current_doc.append(f"\nSub-section: {clean_sub}")
                
#             # 4. Handle Standard Prose
#             else:
#                 current_doc.append(line)
                
#         # Catch and write the final document in the loop
#         if current_doc and len(current_doc) > 5:
#             f.write("\n".join(current_doc))
#             f.write("\n\n" + "="*80 + "\n\n")
#             doc_count += 1
            
#     print(f"\nProcessing Complete!")
#     print(f"Successfully saved {doc_count} heavily formatted documents to '{target_output_file}'")

# if __name__ == "__main__":
#     # You can change this to "validation" or "test" to process the other splits
#     process_and_export_wikitext(split_name="train", output_file="wiki_data_parsed")