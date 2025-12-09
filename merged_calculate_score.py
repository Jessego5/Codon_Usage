import os
import re
from collections import defaultdict
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

from parse_gb import parse_cds_with_flanks


# Codon and amino-acid mappings

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

# reverse map: amino acid -> synonymous codons
AA_TO_CODONS = {}
for c, aa in CODON_TO_AA.items():
    if aa != '*':
        AA_TO_CODONS.setdefault(aa, []).append(c)


# PR matrix

PR = {
    'A':7.0,'R':9.1,'N':10.0,'D':13.0,'C':4.8,'Q':8.6,'E':12.5,'G':7.9,'H':8.4,
    'I':4.9,'L':4.9,'K':10.1,'M':5.3,'F':5.0,'P':6.6,'S':7.5,'T':6.6,
    'W':5.2,'Y':5.4,'V':5.6,
}



# Grantham matrix

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

GRAN_DIST = {a: {b: 0 for b in AA_ORDER} for a in AA_ORDER}
for a, row in _GRAN_UPPER.items():
    for b, v in row.items():
        GRAN_DIST[a][b] = v
        GRAN_DIST[b][a] = v



# PMBEC matrix

PMBEC_ORDER = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

_PMBEC_ROWS = {
    'A':[0.3229,0.0114,-0.0156,-0.0026,-0.0509,0.0383,-0.0833,-0.0020,-0.0104,-0.0424,-0.0651,-0.0854,0.0757,-0.0483,-0.0136,0.0385,0.0520,0.0811,-0.1256,-0.0748],
    'C':[0.0114,0.1007,0.0103,0.0148,0.0346,0.0093,-0.0075,0.0048,-0.0459,-0.0183,-0.0156,0.0021,-0.0086,-0.0310,-0.0483,-0.0218,-0.0227,-0.0154,0.0412,0.0060],
    'D':[-0.0156,0.0103,0.1572,0.0724,-0.0190,-0.0087,-0.0180,-0.0283,-0.0634,-0.0280,-0.0241,0.0194,0.0428,0.0004,-0.0591,-0.0105,0.0066,-0.0194,0.0042,-0.0192],
    'E':[-0.0026,0.0148,0.0724,0.1318,-0.0052,-0.0055,-0.0336,-0.0135,-0.0695,0.0035,-0.0172,0.0053,0.0226,0.0281,-0.0697,-0.0172,-0.0055,-0.0081,-0.0079,-0.0231],
    'F':[-0.0509,0.0346,-0.0190,-0.0052,0.2592,-0.0045,-0.0064,0.0629,-0.0492,0.0489,0.0315,-0.0224,-0.0920,-0.0930,-0.0626,-0.0868,-0.0777,-0.0158,0.0626,0.0858],
    'G':[0.0383,0.0093,-0.0087,-0.0055,-0.0045,0.1225,-0.0256,-0.0207,-0.0137,-0.0326,-0.0264,-0.0040,0.0094,0.0043,-0.0232,0.0304,0.0086,-0.0127,-0.0247,-0.0205],
    'H':[-0.0833,-0.0075,-0.0180,-0.0336,-0.0064,-0.0256,0.2077,-0.0889,0.0761,-0.0352,-0.0008,0.0354,-0.0682,-0.0089,0.1091,0.0094,-0.0234,-0.0870,0.0124,0.0366],
    'I':[-0.0020,0.0048,-0.0283,-0.0135,0.0629,-0.0207,-0.0889,0.2777,-0.0382,0.0886,0.0551,-0.0594,-0.0392,-0.0627,-0.1102,-0.0618,-0.0339,0.0913,0.0014,-0.0231],
    'K':[-0.0104,-0.0459,-0.0634,-0.0695,-0.0492,-0.0137,0.0761,-0.0382,0.2734,-0.0177,-0.0082,-0.0340,-0.0484,-0.0332,0.2152,0.0048,-0.0331,-0.0400,-0.0599,-0.0046],
    'L':[-0.0424,-0.0183,-0.0280,0.0035,0.0489,-0.0326,-0.0352,0.0886,-0.0177,0.1627,0.0751,-0.0112,-0.0510,-0.0013,-0.0749,-0.0585,-0.0313,0.0450,-0.0075,-0.0137],
    'M':[-0.0651,-0.0156,-0.0241,-0.0172,0.0315,-0.0264,-0.0008,0.0551,-0.0082,0.0751,0.1570,0.0075,-0.0916,0.0019,-0.0257,-0.0295,-0.0455,-0.0120,0.0210,0.0126],
    'N':[-0.0854,0.0021,0.0194,0.0053,-0.0224,-0.0040,0.0354,-0.0594,-0.0340,-0.0112,0.0075,0.1515,-0.0106,0.0414,-0.0338,0.0171,-0.0030,-0.0437,0.0296,-0.0021],
    'P':[0.0757,-0.0086,0.0428,0.0226,-0.0920,0.0094,-0.0682,-0.0392,-0.0484,-0.0510,-0.0916,-0.0106,0.3546,0.0481,-0.0377,-0.0069,0.0199,0.0225,-0.0525,-0.0890],
    'Q':[-0.0483,-0.0310,0.0004,0.0281,-0.0930,0.0043,-0.0089,-0.0627,-0.0332,-0.0013,0.0019,0.0414,0.0481,0.1772,0.0072,0.0357,0.0275,-0.0328,-0.0119,-0.0487],
    'R':[-0.0136,-0.0483,-0.0591,-0.0697,-0.0626,-0.0232,0.1091,-0.1102,0.2152,-0.0749,-0.0257,-0.0338,-0.0377,0.0072,0.3890,0.0204,-0.0409,-0.0935,-0.0558,0.0083],
    'S':[0.0385,-0.0218,-0.0105,-0.0172,-0.0868,0.0304,0.0094,-0.0618,0.0048,-0.0585,-0.0295,0.0171,-0.0069,0.0357,0.0204,0.1616,0.0839,-0.0082,-0.0444,-0.0561],
    'T':[0.0520,-0.0227,0.0066,-0.0055,-0.0777,0.0086,-0.0234,-0.0339,-0.0331,-0.0313,-0.0455,-0.0030,0.0199,0.0275,-0.0409,0.0839,0.1425,0.0493,-0.0265,-0.0469],
    'V':[0.0811,-0.0154,-0.0194,-0.0081,-0.0158,-0.0127,-0.0870,0.0913,-0.0400,0.0450,-0.0120,-0.0437,0.0225,-0.0328,-0.0935,-0.0082,0.0493,0.1728,-0.0289,-0.0445],
    'W':[-0.1256,0.0412,0.0042,-0.0079,0.0626,-0.0247,0.0124,0.0014,-0.0599,-0.0075,0.0210,0.0296,-0.0525,-0.0119,-0.0558,-0.0444,-0.0265,-0.0289,0.1940,0.0792],
    'Y':[-0.0748,0.0060,-0.0192,-0.0231,0.0858,-0.0205,0.0366,-0.0231,-0.0046,-0.0137,0.0126,-0.0021,-0.0890,-0.0487,0.0083,-0.0561,-0.0469,-0.0445,0.0792,0.2378],
}

PMBEC = {aa: dict(zip(PMBEC_ORDER, row)) for aa, row in _PMBEC_ROWS.items()}


# BLOSUM62 matrix

BLOSUM62_ORDER = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

_BLOSUM62_ROWS = {
    'A':[4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
    'R':[-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
    'N':[-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
    'D':[-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
    'C':[0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    'Q':[-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
    'E':[-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
    'G':[0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
    'H':[-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
    'I':[-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
    'L':[-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
    'K':[-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
    'M':[-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
    'F':[-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
    'P':[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
    'S':[1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
    'T':[0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
    'W':[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
    'Y':[-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
    'V':[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
}

BLOSUM62 = {aa: dict(zip(BLOSUM62_ORDER, row)) for aa, row in _BLOSUM62_ROWS.items()}


# Effect functions 

def effect_PR(aa0, aa):
    delta = PR[aa] - PR[aa0]
    return delta * delta

def effect_GRANTHAM(aa0, aa):
    d = GRAN_DIST[aa0][aa]
    return d * d

def effect_PMBEC(aa0, aa):
    base = PMBEC[aa0][aa0]  # self-substitution score
    sub  = PMBEC[aa0][aa] # mutant substitution score
    delta = base - sub # loss of compatibility
    return delta * delta

def effect_BLOSUM62(aa0, aa):
    base = BLOSUM62[aa0][aa0]   # self-substitution score
    sub  = BLOSUM62[aa0][aa]    # mutant substitution score
    delta = base - sub          # loss of compatibility
    return delta * delta

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
    for line in table_text.strip().splitlines():
        line = line.strip()
        if not line or line.lower().startswith("context"): #skip empty and header line
            continue
        cols = re.split(r"\s+", line)
        context, ref, alt, methyl, mu = cols[0].upper(), cols[1].upper(), cols[2].upper(), cols[3], cols[4]
        if str(methyl) != "0":
            continue
        mu_val = float(mu) #String to float
        wt[(context, ref, alt)] = mu_val
    return wt


# ===== weighted MSE per codon (generic effect) =====

def ms_for_codon_weighted(full_seq, codon_start, weights, effect_fn):
    codon = full_seq[codon_start:codon_start+3].upper()  # grab 3-nt window for this codon
    if len(codon) < 3:  # incomplete codon near sequence end
        return None

    aa0 = CODON_TO_AA.get(codon)  # translate reference codon
    if aa0 in (None, '*'):  # skip unknown or stop codon
        return None

    L = len(full_seq)
    bases = ['A','C','G','T']  # dna alphabet; no ambiguous codes here
    num, den = 0.0, 0.0        # accumulate weighted squared distances and total weight

    for p in range(3):
        i = codon_start + p
        if i == 0 or i == L - 1:
            continue

        left  = full_seq[i-1].upper()
        ref   = full_seq[i].upper()
        right = full_seq[i+1].upper()
        context = (left + ref + right).upper()

        for alt in bases:
            if alt == ref:  # no-change mutation
                continue

            mu = weights.get((context, ref, alt))  # context-specific mutation rate
            if not mu or mu <= 0.0:
                continue

            mutated = list(codon)    # create mutated codon at position p
            mutated[p] = alt
            mut_codon = "".join(mutated)

            aa = CODON_TO_AA.get(mut_codon)
            if not aa or aa == '*' or aa == aa0:
                continue  # skip unknown, stop, or synonymous changes

            eff = effect_fn(aa0, aa)
            if eff is None:
                continue

            num += mu * eff  # weighted squared effect size
            den += mu        # total mutation weight

    if den == 0.0:  # no valid nonsynonymous mutations observed in this context
        return 0.0
    return num / den


# ===== overall weighted score per CDS =====

def cds_overall_ms_weighted_from_seq_map(seq_map, weights, effect_fn, agg="mean"):
    results = {}
    for i, seq in seq_map.items():
        L = len(seq) - (len(seq) % 3)  # trim to full codons (multiple of 3)
        scores = []
        for s in range(0, L, 3):  # iterate codon starts
            ms = ms_for_codon_weighted(seq, s, weights, effect_fn)
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


def cds_overall_ms_weighted_for_mutants(mutant_lists_by_cds, weights, effect_fn, agg="mean"):
    out = {}
    for i, seq_list in mutant_lists_by_cds.items():
        scores = []
        for seq in seq_list:
            res = cds_overall_ms_weighted_from_seq_map({i: seq}, weights, effect_fn, agg=agg)
            scores.append(res[i]["score"])
        out[i] = scores
    return out


# ===== fully-synonymous recoding (numpy rng) =====

def generate_fully_synonymous_variants(parsed_exons, n_per_cds=1000, rng_seed=None, keep_start=True):
    rng = np.random.default_rng(rng_seed)  # reproducible rng
    cds_map = build_cds_sequences(parsed_exons)
    out = {i: [] for i in cds_map}  # preallocate output lists

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


# ===== plotting =====

def plot_synonymous_histograms_per_cds(
    syn_scores_by_cds, ref_scores_by_cds, out_dir, file_tag, bins=30
):
    os.makedirs(out_dir, exist_ok=True)
    for cds_id, syn_scores in syn_scores_by_cds.items():
        if not syn_scores:
            continue
        arr = np.asarray(syn_scores, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        if hi == lo:
            hi = lo + 1e-6

        plt.figure()
        plt.hist(arr, bins=bins, range=(lo, hi), alpha=0.65, label=f"synonymous (n={len(arr)})")

        ref_score = ref_scores_by_cds.get(cds_id, {}).get("score", None)
        if ref_score is not None:
            plt.axvline(ref_score, linestyle="--", linewidth=1.5, label=f"reference = {ref_score:.6f}")

        plt.xlabel("weighted mse score")
        plt.ylabel("number of variants")
        plt.title(f"{file_tag} — cds {cds_id}: synonymous score distribution")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{file_tag}__cds_{cds_id}__syn_hist.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

if __name__ == "__main__":

    EFFECT_FN = effect_GRANTHAM

    GENBANK_FILES = [
        "sequence-hbb.gb","sequence-hba1.gb","sequence-mb.gb",
        "sequence-polr2a.gb","sequence-cycs.gb","sequence-gapdh.gb",
        "sequence-actb.gb","sequence-hspa1l.gb","sequence-ldha.gb",
        "sequence-atp5f1a.gb"
    ]

    WEIGHTS_PATH = "mutation_rate_methylation_bins.txt"
    OUT_DIR = f"multi_seq_{EFFECT_FN.__name__}_hists"
    N_SYNONYMOUS = 1000
    RNG_SEED = 888
    KEEP_START = True
    AGGREGATION = "mean"

    t0 = perf_counter()
    weights = load_mut96_weights(WEIGHTS_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    for gb in GENBANK_FILES:
        file_tag = os.path.splitext(os.path.basename(gb))[0]
        print(f"\n=== processing {gb} ===")

        parsed = parse_cds_with_flanks(gb)
        cds_map = build_cds_sequences(parsed)

        ref_scores = cds_overall_ms_weighted_from_seq_map(
            cds_map, weights, EFFECT_FN, agg=AGGREGATION
        )

        syn_variants = generate_fully_synonymous_variants(
            parsed, n_per_cds=N_SYNONYMOUS, rng_seed=RNG_SEED, keep_start=KEEP_START
        )

        syn_scores = cds_overall_ms_weighted_for_mutants(
            syn_variants, weights, EFFECT_FN, agg=AGGREGATION
        )

        print("reference scores per cds:", {k: v["score"] for k, v in ref_scores.items()})

        plot_synonymous_histograms_per_cds(
            syn_scores, ref_scores, out_dir=OUT_DIR, file_tag=file_tag, bins=30
        )

    elapsed = perf_counter() - t0
    print(f"\nall done. elapsed: {elapsed:.3f}s")
    print(f"outputs saved under: {OUT_DIR}/")
