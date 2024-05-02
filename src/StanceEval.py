
import sys

def print_usage():
    print("""
---------------------------
Usage:
python StanceEval.py goldFile guessFile

goldFile: file containing gold standards;
guessFile: file containing your prediction.

These two files have the same format:
ID<Tab>Target<Tab>Tweet<Tab>Stance
Only stance labels may be different between them!
---------------------------
""")

if len(sys.argv) == 2 and sys.argv[1] == "-u":
    print_usage()
    sys.exit()

if len(sys.argv) != 3:
    sys.stderr.write("\nError: Number of parameters are incorrect!\n")
    print_usage()
    sys.exit(1)

fn_gold = sys.argv[1]
fn_guess = sys.argv[2]

try:
    with open(fn_gold, 'r') as f_gold, open(fn_guess, 'r') as f_guess:
        gold_lines = f_gold.readlines()
        guess_lines = f_guess.readlines()
except IOError as e:
    sys.stderr.write(f"Error: Cannot open file: {e.filename}\n")
    sys.exit(1)

gold_lines = [line.strip() for line in gold_lines]
guess_lines = [line.strip() for line in guess_lines]

if len(guess_lines) != len(gold_lines):
    sys.stderr.write("\nError: Make sure the number of lines in your prediction file is same as that in the gold-standard file!\n")
    sys.stderr.write(f"The gold-standard file contains {len(gold_lines)} lines, but the prediction file contains {len(guess_lines)} lines.\n")
    sys.exit(1)

targets = ["Women empowerment", "Covid Vaccine", "Digital Transformation"]
cats = ["FAVOR", "AGAINST", "NONE"] 

# Initialize dictionaries to store statistics for each target
num_of_true_pos_of_each_target = {target: {cat: 0 for cat in cats} for target in targets}
num_of_guess_of_each_target = {target: {cat: 0 for cat in cats} for target in targets}
num_of_gold_of_each_target = {target: {cat: 0 for cat in cats} for target in targets}

for gold_line, guess_line in zip(gold_lines, guess_lines):
    if gold_line == "ID\tTarget\tTweet\tStance":
        continue

    gold_arr = gold_line.split("\t")
    guess_arr = guess_line.split("\t")

    if len(gold_arr) != 4:
        sys.stderr.write(f"\nError: the following line in the gold-standard file does not have a correct format:\n\n{gold_line}\n\n")
        sys.stderr.write("Correct format: ID<Tab>Target<Tab>Tweet<Tab>Stance\n")
        sys.exit(1)

    if len(guess_arr) != 4:
        sys.stderr.write(f"\nError: the following line in your prediction file does not have a correct format:\n\n{guess_line}\n\n")
        sys.stderr.write("Correct format: ID<Tab>Target<Tab>Tweet<Tab>Stance\n")
        sys.exit(1)

    gold_target = gold_arr[1]
    gold_lbl = gold_arr[3]
    guess_target = guess_arr[1]
    guess_lbl = guess_arr[3]

    if gold_target not in targets:
        sys.stderr.write(f"\nError: the target \"{gold_target}\" in the gold-standard file is invalid:\n\n{gold_line}\n\n")
        sys.exit(1)

    if guess_target not in targets:
        sys.stderr.write(f"\nError: the target \"{guess_target}\" in the prediction file is invalid:\n\n{guess_line}\n\n")
        sys.exit(1)

    if gold_lbl not in cats:
        sys.stderr.write(f"\nError: the stance label \"{gold_lbl}\" in the gold-standard file is invalid:\n\n{gold_line}\n\n")
        sys.exit(1)

    if guess_lbl not in cats:
        sys.stderr.write(f"\nError: the stance label \"{guess_lbl}\" in the prediction file is invalid:\n\n{guess_line}\n\n")
        sys.exit(1)

    num_of_gold_of_each_target[gold_target][gold_lbl] += 1
    num_of_guess_of_each_target[guess_target][guess_lbl] += 1

    if guess_lbl == gold_lbl:
        num_of_true_pos_of_each_target[guess_target][guess_lbl] += 1

prec_by_target = {target: {cat: 0 for cat in cats} for target in targets}
recall_by_target = {target: {cat: 0 for cat in cats} for target in targets}
f_by_target = {target: {cat: 0 for cat in cats} for target in targets}
macro_f_by_target = {target: 0.0 for target in targets}

for target in targets:
    macro_f = 0.0
    n_cat = 0

    for cat in cats:
        n_tp = num_of_true_pos_of_each_target[target][cat]
        n_guess = num_of_guess_of_each_target[target][cat]
        n_gold = num_of_gold_of_each_target[target][cat]

        p = 0
        r = 0
        f = 0

        if n_guess != 0:
            p = n_tp / n_guess
        if n_gold != 0:
            r = n_tp / n_gold
        if p + r != 0:
            f = 2 * p * r / (p + r)

        prec_by_target[target][cat] = p
        recall_by_target[target][cat] = r
        f_by_target[target][cat] = f

        if cat in ["FAVOR", "AGAINST"]:
            n_cat += 1
            macro_f += f

    macro_f = macro_f / n_cat
    macro_f_by_target[target] = macro_f

    # Print results for each target
    print(f"\n\n============\nResults for Target: {target}\n============")
    for cat in cats:
        if cat in ["FAVOR", "AGAINST"]:
            print(f"{cat:<9} precision: {prec_by_target[target][cat]:.4f} recall: {recall_by_target[target][cat]:.4f} f-score: {f_by_target[target][cat]:.4f}")
    print("------------")
    print(f"Macro F: {macro_f:.4f}\n\n")

# Compute overall macro F1-score across all targets
overall_macro_f = sum(macro_f_by_target.values()) / len(targets)
print(f"\n\n============\nOverall Macro F1-score across all targets: {overall_macro_f:.4f}\n============\n")