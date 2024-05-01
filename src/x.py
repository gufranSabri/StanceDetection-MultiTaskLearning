import itertools

# Define the available BERT indices
bert_indices = ['0', '1', '2', '3']

# Generate all possible combinations of the BERT indices without repeats
all_bert_combinations = list(itertools.permutations(bert_indices))

# Print each combination
for combination in all_bert_combinations:
    bert_str = ''.join(combination)
    command = f"python src/main.py -model parallel -bert {bert_str} -pool 1"
    print(command)
