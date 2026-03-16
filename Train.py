from transition_amr_parser.parse import AMRParser

# Download and save a model named AMR3.0 to cache
parser = AMRParser.from_pretrained('AMR3-structbart-L')


def get_amr_for_list(sentences):
    """
    Takes a list of strings and returns a list of Penman AMR strings.
    """
    
    # 1. Tokenize all sentences first
    # The parser works best when tokenization is handled explicitly
    tokenized_list = [parser.tokenize(s)[0] for s in sentences]
    
    # 2. Batch parse the sentences
    # machines contains the internal state of the transition-based parser
    _, machines = parser.parse_sentences(tokenized_list)
    
    # 3. Convert each machine state to a Penman string
    # jamr=True adds the word alignments (~ index)
    # isi=False removes the metadata IDs you wanted to avoid
    amr_outputs = []
    for m in machines:
        penman_str = m.get_amr().to_penman(jamr=True, isi=False)
        amr_outputs.append(penman_str)
        
    return amr_outputs


my_sentences = [
        "Hailey is going to London tomorrow.",
        "She is planning to go to Italy after London.",
        "She is going to see the Big Ben.",
        "Her friend Phil is meeting her in London."
    ]

results = get_amr_for_list(my_sentences)

# Print results separated by new lines
for i, amr in enumerate(results):
    print(f"# Sentence {i+1}: {my_sentences[i]}")
    print(amr)
    print("-" * 30) # Visual separator





# tokens, positions = parser.tokenize('Hailey is going to London tomorrow.')

# # Use parse_sentence() for single sentences or parse_sentences() for a batch
# annotations, machines = parser.parse_sentence(tokens)

# # Print Penman notation
# print(annotations)

# amr = machines.get_amr()
# # Print Penman notation with JAMR and no ISI annotations
# print(amr.to_penman(jamr=True, isi=False))

# # Plot the graph (requires matplotlib)
# amr.plot()