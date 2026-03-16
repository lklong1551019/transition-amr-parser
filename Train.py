from transition_amr_parser.parse import AMRParser

# Download and save a model named AMR3.0 to cache
parser = AMRParser.from_pretrained('AMR3-structbart-L')
tokens, positions = parser.tokenize('Hailey is going to London tomorrow.')

# Use parse_sentence() for single sentences or parse_sentences() for a batch
annotations, machines = parser.parse_sentence(tokens)

# Print Penman notation
print(annotations)

amr = machines.get_amr()
# Print Penman notation with JAMR and no ISI annotations
print(amr.to_penman(jamr=True, isi=False))

# Plot the graph (requires matplotlib)
amr.plot()