import os
import re
from collections import defaultdict
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from parse_gb import parse_cds_with_flanks


# ===== codon to amino acid table =====

CODON_TO_AA = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGA':'R','AGG':'R',
    'AGT':'S','AGC':'S',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

#Creates reverse mapping from amino acids to their possible codons.
#Used for finding synonymous codons
AA_TO_CODONS = {}
for c, aa in CODON_TO_AA.items():
    if aa != '*':
        AA_TO_CODONS.setdefault(aa, []).append(c)

# ===== Amino acid distance (Grantham) =====
#Chemical composition, Polarity, Molecular volume
AA_ORDER = ['S','R','L','P','T','A','V','G','I','F','Y','C','H','Q','N','K','D','E','M','W']
_GRAN_UPPER = {
 'S': {'R':110,'L':145,'P':74,'T':58,'A':99,'V':124,'G':56,'I':142,'F':155,'Y':144,'C':112,'H':89,'Q':68,'N':46,'K':121,'D':65,'E':80,'M':135,'W':177},
 'R': {'L':102,'P':103,'T':71,'A':112,'V':96,'G':125,'I':97,'F':97,'Y':77,'C':180,'H':29,'Q':43,'N':86,'K':26,'D':96,'E':54,'M':91,'W':101},
 'L': {'P':98,'T':92,'A':96,'V':32,'G':138,'I':5,'F':22,'Y':36,'C':198,'H':99,'Q':113,'N':153,'K':107,'D':172,'E':138,'M':15,'W':61},
 'P': {'T':38,'A':27,'V':68,'G':42,'I':95,'F':114,'Y':110,'C':169,'H':77,'Q':76,'N':91,'K':103,'D':108,'E':93,'M':87,'W':147},
 'T': {'A':58,'V':69,'G':59,'I':89,'F':103,'Y':92,'C':149,'H':47,'Q':42,'N':65,'K':78,'D':85,'E':65,'M':81,'W':128},
 'A': {'V':64,'G':60,'I':94,'F':113,'Y':112,'C':195,'H':86,'Q':91,'N':111,'K':106,'D':126,'E':107,'M':84,'W':148},
 'V': {'G':109,'I':29,'F':50,'Y':55,'C':192,'H':84,'Q':96,'N':133,'K':97,'D':152,'E':121,'M':21,'W':88},
 'G': {'I':135,'F':153,'Y':147,'C':159,'H':98,'Q':87,'N':80,'K':127,'D':94,'E':98,'M':127,'W':184},
 'I': {'F':21,'Y':33,'C':198,'H':94,'Q':109,'N':149,'K':102,'D':168,'E':134,'M':10,'W':61},
 'F': {'Y':22,'C':205,'H':100,'Q':116,'N':158,'K':102,'D':177,'E':140,'M':28,'W':40},
 'Y': {'C':194,'H':83,'Q':99,'N':143,'K':85,'D':160,'E':122,'M':36,'W':37},
 'C': {'H':174,'Q':154,'N':139,'K':202,'D':154,'E':170,'M':196,'W':215},
 'H': {'Q':24,'N':68,'K':32,'D':81,'E':40,'M':87,'W':115},
 'Q': {'N':46,'K':53,'D':61,'E':29,'M':101,'W':130},
 'N': {'K':94,'D':23,'E':42,'M':142,'W':174},
 'K': {'D':101,'E':56,'M':95,'W':110},
 'D': {'E':45,'M':160,'W':181},
 'E': {'M':126,'W':152},
 'M': {'W':67},
 'W': {}
}

#Convert to full symmetric matrix (literally a lookup table)
GRAN_DIST = {aa: {bb: 0 for bb in AA_ORDER} for aa in AA_ORDER}
for a, row in _GRAN_UPPER.items():
    for b, v in row.items():
        GRAN_DIST[a][b] = v #Set upper triangle value
        GRAN_DIST[b][a] = v #Mirror to lower triangle

# ===== exon concatenation =====
def build_cds_sequences(parsed_exons):
    """
    Input: list of dicts {"cds_index","exon_index","seq"} with ±1nt flanks included.
    Output: dict {cds_index: concatenated coding sequence (flanks removed)}
    """
    bins = defaultdict(list) #Groups exons by CDS index
    for e in parsed_exons:
        core = e["seq"][1:-1] #Drop flanks per exon
        bins[e["cds_index"]].append((e["exon_index"], core))

    cds_seq = {}
    for i, parts in bins.items():
        parts.sort(key=lambda x: x[0]) #Sort by exon index order
        cds_seq[i] = "".join(p for _, p in parts) #Concatenate exon sequences
    return cds_seq

# ===== load the 96-context weights (methylation_level==0 only)=====
# can change to other weights
def load_mut96_weights(path_to_table):
    table_text = open(path_to_table, encoding="utf-8").read()
    wt = {}
    #Loops over each line in the file
    for line in table_text.strip().splitlines():
        line = line.strip()
        if not line or line.lower().startswith("context"): #skip empty and header line
            continue

        #Splits the line on any whitespace
        cols = re.split(r"\s+", line)

        #pull the columns
        context, ref, alt, methyl, mu = cols[0].upper(), cols[1].upper(), cols[2].upper(), cols[3], cols[4]
        #Only keep rows with meth level of 0
        if str(methyl) != "0":
            continue

        mu_val = float(mu) #String to float
        wt[(context, ref, alt)] = mu_val
    return wt

# ===== weighted MSE per codon using Grantham distance =====
def ms_for_codon_GRANTHAM_weighted(full_seq, codon_start, weights):
    codon = full_seq[codon_start:codon_start+3].upper()
    if len(codon) < 3:
        return None
    aa0 = CODON_TO_AA.get(codon)
    if aa0 in (None, '*'):
        return None

    L = len(full_seq)
    pos_neighbors = []
    for p in range(3):
        i = codon_start + p
        left  = full_seq[i-1].upper() if i-1 >= 0 else None
        right = full_seq[i+1].upper() if i+1 < L else None
        pos_neighbors.append((left, codon[p], right))

    bases = ['A','C','G','T']
    num, den = 0.0, 0.0
    for p in range(3):
        left, ref, right = pos_neighbors[p]
        if left is None or right is None:
            continue
        context = (left + ref + right).upper()
        for alt in bases:
            if alt == ref:
                continue
            mu = weights.get((context, ref, alt))
            if not mu or mu <= 0.0:
                continue
            mutated = list(codon)
            mutated[p] = alt
            mut_codon = "".join(mutated)
            aa = CODON_TO_AA.get(mut_codon)
            if not aa or aa == '*' or aa == aa0:
                continue  # skip stop or synonymous
            dist = GRAN_DIST.get(aa0, {}).get(aa)
            if dist is None:
                continue
            num += mu * (dist * dist)
            den += mu
    if den == 0.0:
        return 0.0
    return num / den

# ===== overall weighted score per CDS =====
def cds_overall_ms_weighted_from_seq_map(seq_map, weights, agg="mean"):
    results = {}
    for i, seq in seq_map.items():
        L = len(seq) - (len(seq) % 3)
        scores = []
        for s in range(0, L, 3):
            ms = ms_for_codon_GRANTHAM_weighted(seq, s, weights)
            if ms is not None:
                scores.append(ms)
        if not scores:
            results[i] = {"score": 0.0, "agg": agg, "n_codons": 0, "sum": 0.0, "mean": 0.0}
            continue
        total = sum(scores)
        mean = total / len(scores)
        results[i] = {
            "score": mean if agg == "mean" else total,
            "agg": agg,
            "n_codons": len(scores),
            "sum": total,
            "mean": mean,
        }
    return results

def cds_overall_ms_weighted(parsed_exons, mut96_text_path, agg="mean"):
    weights = load_mut96_weights(mut96_text_path)
    cds_map = build_cds_sequences(parsed_exons)
    return cds_overall_ms_weighted_from_seq_map(cds_map, weights, agg=agg)

def cds_overall_ms_weighted_for_mutants(mutant_lists_by_cds, weights, agg="mean"):
    out = {}
    for i, seq_list in mutant_lists_by_cds.items():
        scores = []
        for seq in seq_list:
            res = cds_overall_ms_weighted_from_seq_map({i: seq}, weights, agg=agg)
            scores.append(res[i]["score"])
        out[i] = scores
    return out

# ===== helpers =====
def _aa_sequence(seq):
    aa = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        c = seq[i:i+3].upper()
        aa.append(CODON_TO_AA.get(c, 'X'))
    return "".join(aa)

# ===== fully-synonymous recoding (NumPy RNG) =====
def generate_fully_synonymous_variants(parsed_exons, n_per_cds=1000, rng_seed=None, keep_start=True):
    """
    For each CDS, generate n_per_cds variants where every codon (except start if keep_start=True)
    is replaced with a random synonymous codon different from the original, when possible.
    Leaves codons with no synonyms unchanged (e.g., ATG, TGG).
    Returns: dict[int -> list[str]] of recoded CDS sequences.
    """
    rng = np.random.default_rng(rng_seed)
    cds_map = build_cds_sequences(parsed_exons)
    out = {i: [] for i in cds_map}

    for i, seq in cds_map.items():
        seq = seq.upper()
        L = len(seq) - (len(seq) % 3)
        seq = seq[:L]
        if L < 3:
            continue

        n_codons = L // 3
        original_codons = [seq[j:j+3] for j in range(0, L, 3)]

        for _ in range(n_per_cds):
            new_codons = original_codons.copy()
            # iterate codons; skip 0 if keep_start=True
            start_idx = 1 if keep_start else 0
            for codon_idx in range(start_idx, n_codons):
                codon = original_codons[codon_idx]
                aa = CODON_TO_AA.get(codon)
                if not aa or aa == '*':
                    continue
                synonyms = AA_TO_CODONS.get(aa, [])
                choices = [c for c in synonyms if c != codon]
                if not choices:
                    continue
                new_codons[codon_idx] = rng.choice(choices)
            new_seq = "".join(new_codons)
            out[i].append(new_seq)
    return out

# ===== plotting & multi-file driver (SIMPLIFIED - matching Script 1 style) =====
def plot_synonymous_histograms_per_cds(
    syn_scores_by_cds, ref_scores_by_cds, out_dir, file_tag, bins=30
):
    """
    Saves one histogram per CDS for a single source file.
    """
    os.makedirs(out_dir, exist_ok=True)

    for cds_id, syn_scores in syn_scores_by_cds.items():
        if not syn_scores:
            continue
        arr = np.asarray(syn_scores, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        if hi == lo:
            hi = lo + 1e-6

        plt.figure()
        plt.hist(arr, bins=bins, range=(lo, hi), alpha=0.65, label=f"Synonymous (n={len(arr)})")

        ref_score = ref_scores_by_cds.get(cds_id, {}).get("score", None)
        if ref_score is not None:
            plt.axvline(ref_score, linestyle="--", linewidth=1.5, label=f"Reference = {ref_score:.6f}")

        plt.xlabel("Grantham-weighted MSE score")
        plt.ylabel("Number of variants")
        plt.title(f"{file_tag} — CDS {cds_id}: Synonymous score distribution")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{file_tag}__cds_{cds_id}__syn_hist.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

def write_summary_csv(ref_scores_by_cds, syn_scores_by_cds, out_dir, file_tag):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{file_tag}__summary.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("file_tag,cds_id,n_syn,ref_score,min,median,max,mean,std\n")
        for cds_id, syn in syn_scores_by_cds.items():
            if syn:
                arr = np.asarray(syn, dtype=float)
                row = [
                    file_tag,
                    cds_id,
                    len(arr),
                    ref_scores_by_cds.get(cds_id, {}).get("score", float("nan")),
                    float(np.min(arr)),
                    float(np.median(arr)),
                    float(np.max(arr)),
                    float(np.mean(arr)),
                    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                ]
            else:
                row = [file_tag, cds_id, 0,
                       ref_scores_by_cds.get(cds_id, {}).get("score", float("nan")),
                       "", "", "", "", ""]
            f.write(",".join(map(str, row)) + "\n")

def process_genbank_file(
    gb_path,
    weights_dict,
    n_syn_variants=1000,
    rng_seed=123,
    keep_start=True,
    agg="mean",
):
    """
    Parses a single GenBank file and returns:
      - ref_scores_by_cds: dict[int -> {..., 'score': float, ...}]
      - syn_scores_by_cds: dict[int -> List[float]] scores for fully synonymous variants
    """
    parsed = parse_cds_with_flanks(gb_path)
    ref_scores_by_cds = cds_overall_ms_weighted_from_seq_map(
        build_cds_sequences(parsed), weights_dict, agg=agg
    )
    syn_variants = generate_fully_synonymous_variants(
        parsed, n_per_cds=n_syn_variants, rng_seed=rng_seed, keep_start=keep_start
    )
    syn_scores_by_cds = cds_overall_ms_weighted_for_mutants(
        syn_variants, weights_dict, agg=agg
    )
    return ref_scores_by_cds, syn_scores_by_cds

if __name__ == "__main__":
    GENBANK_FILES = [
        "sequence-hbb.gb",
        "sequence-hba1.gb",
        "sequence-mb.gb",
        "sequence-polr2a.gb",
        "sequence-cycs.gb",
        "sequence-gapdh.gb",
        "sequence-actb.gb",
        "sequence-hspa1l.gb",
        "sequence-ldha.gb",
        "sequence-atp5f1a.gb"
        # add more files here
    ]
    WEIGHTS_PATH = "mutation_rate_methylation_bins.txt"
    OUT_DIR = "multi_seq_grantham_hists_2"
    N_SYNONYMOUS = 1000
    RNG_SEED = 777
    KEEP_START = True
    AGGREGATION = "mean"

    t0 = perf_counter()
    weights = load_mut96_weights(WEIGHTS_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    for gb in GENBANK_FILES:
        file_tag = os.path.splitext(os.path.basename(gb))[0]
        print(f"\n=== Processing {gb} ===")
        ref_scores, syn_scores_by_cds = process_genbank_file(
            gb,
            weights_dict=weights,
            n_syn_variants=N_SYNONYMOUS,
            rng_seed=RNG_SEED,
            keep_start=KEEP_START,
            agg=AGGREGATION,
        )

        # Console summary
        print("Reference scores per CDS:", {k: v["score"] for k, v in ref_scores.items()})
        for cds_id, syn_scores in syn_scores_by_cds.items():
            if syn_scores:
                arr = np.asarray(syn_scores, dtype=float)
                print(
                    f"[{file_tag} | CDS {cds_id}] Syn variants (n={len(arr)}): "
                    f"min={arr.min():.6f}  median={np.median(arr):.6f}  max={arr.max():.6f}"
                )
            else:
                print(f"[{file_tag} | CDS {cds_id}] No synonymous variants generated")

        plot_synonymous_histograms_per_cds(
            syn_scores_by_cds, ref_scores, out_dir=OUT_DIR, file_tag=file_tag, bins=30
        )
        write_summary_csv(ref_scores, syn_scores_by_cds, out_dir=OUT_DIR, file_tag=file_tag)

    elapsed = perf_counter() - t0
    print(f"\nAll done. Elapsed: {elapsed:.3f}s")
    print(f"Outputs saved under: {OUT_DIR}/")

