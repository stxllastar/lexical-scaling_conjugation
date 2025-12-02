## Script for generating training data with eight suffix variants ##

import csv

# Classes of stems
classes = {
    "t-regular": ["pat", "met", "pot", "sut", "kat"],
    "t-irregular": ["hot", "mut", "mot", "nit", "tot", "not", "mat"],
    "s-regular": ["tus", "sis", "hus", "kis"],
    "s-irregular": ["mis", "pes", "ses", "nes", "nus"]
}

# Flatten stems and assign s numbers
stems_list = []
s_counter = 0
stem_to_s = {}
stem_to_class = {}
for cls, stems in classes.items():
    for stem in stems:
        stem_to_s[stem] = f"s{s_counter}"
        stems_list.append(stem)
        stem_to_class[stem] = cls
        s_counter += 1


# Endings with p indices
endings = ["a", "ta","e","te","tu","u","o","to"]
ending_to_p = {ending: f"p{i}" for i, ending in enumerate(endings)}


# Function to generate 3 SRs per stem+ending with coda rule
def generate_srs(stem, ending, stem_class):
    # Korean coda rule: s -> t for s-class stems with 'ta'
    if stem_class.startswith("s") and ending in ("ta","te","tu","to"):
        stem_mod = stem[:-1] + "t"
    else:
        stem_mod = stem
    # canonical
    canonical = stem_mod + ending
    # alternate candidates
    alt1 = stem[:-1] + "l"+ ending
    alt2 = stem[:-1] + ending
    return [canonical, alt1, alt2]




filename = "8suffix.tsv"

with open(filename, "w", newline="", encoding="utf-8") as tsvfile:
    writer = csv.writer(tsvfile, delimiter="\t")
    writer.writerow( [f"s{i}" for i in range(len(stems_list))] + ['p0', 'p1','p2','p3','p4','p5','p6','p7']) #writing the first row
    writer.writerow(["", "", "","*VcV", "Max", "Ident"]) #write constraints

    for stem in stems_list:
        stem_class = stem_to_class[stem]
        for ending in endings:
            input_id = f"{stem}-{ending}$$${stem_to_s[stem]}${ending_to_p[ending]}"
            candidates = generate_srs(stem, ending, stem_class)

            # Determine winner
            if stem_class == "t-irregular" and ending in ("a", "e", "u","o"):
                winner_idx = 1
            elif stem_class == "s-irregular" and ending in ("a", "e", "u","o"):
                winner_idx = 2
            else:
                winner_idx = 0

            for idx, cand in enumerate(candidates):
                perc = 1.0 if idx == winner_idx else 0.0
                vc, mx, ident = 0, 0, 0
                # faithful candidate violates *VcV under -a 
                if ending in ("a", "e", "u","o") and idx == 0:
                    vc = 1
                if idx == 1:
                    ident = 1
                if idx == 2:
                    mx = 1
                # Only put input on first row
                row_input = input_id if idx == 0 else ""
                writer.writerow([row_input, cand, perc, int(vc), int(mx), int(ident)])
                
