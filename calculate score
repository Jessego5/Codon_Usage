import os
import re
from collections import defaultdict, Counter
from time import perf_counter
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt


from parse_gb import parse_cds_with_flanks


# =========================
# Codon <-> AA mappings
# =========================
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

AA_TO_CODONS = {}
for codon, aa in CODON_TO_AA.items():
    if aa != '*':
        AA_TO_CODONS.setdefault(aa, []).append(codon)


# =========================
# Metric definitions
# all return (delta)^2
# =========================

PR = {
    'A':7.0,'R':9.1,'N':10.0,'D':13.0,'C':4.8,'Q':8.6,'E':12.5,'G':7.9,'H':8.4,
    'I':4.9,'L':4.9,'K':10.1,'M':5.3,'F':5.0,'P':6.6,'S':7.5,'T':6.6,
    'W':5.2,'Y':5.4,'V':5.6,
}

def effect_PR(aa0, aa):
    d = PR[aa] - PR[aa0]
    return d * d


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

def effect_GRANTHAM(aa0, aa):
    d = GRAN_DIST[aa0][aa]
    return d * d


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

def effect_PMBEC(aa0, aa):
    base = PMBEC[aa0][aa0]
    sub  = PMBEC[aa0][aa]
    d = base - sub
    return d * d


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

def effect_BLOSUM62(aa0, aa):
    base = BLOSUM62[aa0][aa0]
    sub  = BLOSUM62[aa0][aa]
    d = base - sub
    return d * d


METRIC_TO_EFFECT = {
    "PR": effect_PR,
    "GRANTHAM": effect_GRANTHAM,
    "BLOSUM62": effect_BLOSUM62,
    "PMBEC": effect_PMBEC,
}


COMP = {'A':'T','T':'A','C':'G','G':'C'}

def revcomp_trinuc(ctx3):
    return "".join(COMP[b] for b in ctx3[::-1])

def infer_canonical_centers(weights_dict):
    centers = Counter()
    for (ctx, ref, alt), mu in weights_dict.items():
        centers[ref] += 1
    keys = set(centers.keys())
    if keys and keys.issubset({"A","C"}):
        return "AC"
    if keys and keys.issubset({"C","T"}):
        return "CT"
    return "NONE"

def get_mu_canonical(context, ref, alt, weights_dict, canonical_mode):
    context = context.upper()
    ref = ref.upper()
    alt = alt.upper()

    if canonical_mode == "NONE":
        return weights_dict.get((context, ref, alt))

    if canonical_mode == "AC":
        if ref in ("A","C"):
            return weights_dict.get((context, ref, alt))
        ctx2 = revcomp_trinuc(context)
        return weights_dict.get((ctx2, COMP[ref], COMP[alt]))

    if canonical_mode == "CT":
        if ref in ("C","T"):
            return weights_dict.get((context, ref, alt))
        ctx2 = revcomp_trinuc(context)
        return weights_dict.get((ctx2, COMP[ref], COMP[alt]))

    return weights_dict.get((context, ref, alt))


def build_cds_sequences(parsed_exons):
    bins = defaultdict(list)
    for e in parsed_exons:
        core = e["seq"][1:-1]
        bins[e["cds_index"]].append((e["exon_index"], core))

    cds_seq = {}
    for cds_id, parts in bins.items():
        parts.sort(key=lambda x: x[0])
        cds_seq[cds_id] = "".join(p for _, p in parts)
    return cds_seq


def load_mut96_weights(path_to_table):
    table_text = open(path_to_table, encoding="utf-8").read()
    wt = {}
    for line in table_text.strip().splitlines():
        line = line.strip()
        if not line or line.lower().startswith("context"):
            continue
        cols = re.split(r"\s+", line)
        if len(cols) < 5:
            continue
        context, ref, alt, methyl, mu = cols[0].upper(), cols[1].upper(), cols[2].upper(), cols[3], cols[4]
        if str(methyl) != "0":
            continue
        try:
            mu_val = float(mu)
        except Exception:
            continue
        wt[(context, ref, alt)] = mu_val
    return wt


def has_only_acgt(seq):
    return all(b in "ACGT" for b in seq.upper())

def count_internal_stops(seq):
    seq = seq.upper()
    L = len(seq) - (len(seq) % 3)
    seq = seq[:L]
    n_codons = L // 3
    if n_codons <= 1:
        return 0
    stops = 0
    for s in range(0, L - 3, 3):
        c = seq[s:s+3]
        if CODON_TO_AA.get(c) == '*':
            stops += 1
    return stops

def codon_den_is_positive(seq, codon_start, weights, canonical_mode):
    seq = seq.upper()
    codon = seq[codon_start:codon_start+3]
    if len(codon) < 3 or any(b not in "ACGT" for b in codon):
        return False

    aa0 = CODON_TO_AA.get(codon)
    if not aa0 or aa0 == '*':
        return False

    L = len(seq)
    bases = "ACGT"

    for p in range(3):
        i = codon_start + p
        if i == 0 or i == L - 1:
            continue

        left, ref, right = seq[i-1], seq[i], seq[i+1]
        if left not in bases or ref not in bases or right not in bases:
            continue
        context = left + ref + right

        for alt in bases:
            if alt == ref:
                continue
            mu = get_mu_canonical(context, ref, alt, weights, canonical_mode)
            if not mu or mu <= 0:
                continue

            mutated = list(codon)
            mutated[p] = alt
            aa = CODON_TO_AA.get("".join(mutated))
            if not aa or aa == '*' or aa == aa0:
                continue
            return True

    return False

def qc_check_cds(seq, weights, canonical_mode,
                 min_codons=30,
                 require_multiple_of_3=True,
                 max_den0_frac=0.02,
                 disallow_internal_stops=True):
    info = {}
    seq = seq.upper()
    info["raw_len"] = len(seq)

    if not has_only_acgt(seq):
        return False, {**info, "reason": "non_ACGT_bases"}

    if require_multiple_of_3 and (len(seq) % 3 != 0):
        return False, {**info, "reason": "len_not_multiple_of_3"}

    L = len(seq) - (len(seq) % 3)
    seq2 = seq[:L]
    info["used_len"] = L
    n_codons = L // 3
    info["n_codons"] = n_codons

    if n_codons < min_codons:
        return False, {**info, "reason": f"too_short(<{min_codons}_codons)"}

    internal_stops = count_internal_stops(seq2)
    info["internal_stops"] = internal_stops
    if disallow_internal_stops and internal_stops > 0:
        return False, {**info, "reason": "internal_stop_codons"}

    den0 = 0
    for s in range(0, L, 3):
        if not codon_den_is_positive(seq2, s, weights, canonical_mode):
            den0 += 1
    info["den0_codons"] = den0
    info["den0_frac"] = den0 / max(n_codons, 1)

    if info["den0_frac"] > max_den0_frac:
        return False, {**info, "reason": f"den0_frac>{max_den0_frac}"}

    return True, {**info, "reason": "PASS"}


def ms_for_codon_weighted(full_seq, codon_start, weights, canonical_mode, effect_fn):
    codon = full_seq[codon_start:codon_start+3].upper()
    if len(codon) < 3 or any(b not in "ACGT" for b in codon):
        return None

    aa0 = CODON_TO_AA.get(codon)
    if aa0 in (None, '*'):
        return None

    L = len(full_seq)
    bases = "ACGT"
    num, den = 0.0, 0.0

    for p in range(3):
        i = codon_start + p
        if i == 0 or i == L - 1:
            continue

        left  = full_seq[i-1].upper()
        ref   = full_seq[i].upper()
        right = full_seq[i+1].upper()

        if left not in bases or ref not in bases or right not in bases:
            continue

        context = left + ref + right

        for alt in bases:
            if alt == ref:
                continue

            mu = get_mu_canonical(context, ref, alt, weights, canonical_mode)
            if not mu or mu <= 0.0:
                continue

            mutated = list(codon)
            mutated[p] = alt
            mut_codon = "".join(mutated)

            aa = CODON_TO_AA.get(mut_codon)
            if not aa or aa == '*' or aa == aa0:
                continue

            try:
                eff = effect_fn(aa0, aa)
            except Exception:
                continue

            num += mu * eff
            den += mu

    if den == 0.0:
        return 0.0
    return num / den


def cds_overall_ms_weighted_from_seq_map(seq_map, weights, canonical_mode, effect_fn, agg="mean"):
    results = {}
    for cds_id, seq in seq_map.items():
        seq = seq.upper()
        L = len(seq) - (len(seq) % 3)
        seq = seq[:L]

        scores = []
        for s in range(0, L, 3):
            ms = ms_for_codon_weighted(seq, s, weights, canonical_mode, effect_fn)
            if ms is not None:
                scores.append(ms)

        if not scores:
            results[cds_id] = {"score": 0.0, "agg": agg, "n_codons": 0, "sum": 0.0, "mean": 0.0}
            continue

        total = float(sum(scores))
        mean = float(total / len(scores))
        results[cds_id] = {
            "score": mean if agg == "mean" else total,
            "agg": agg,
            "n_codons": len(scores),
            "sum": total,
            "mean": mean,
        }
    return results


def cds_overall_ms_weighted_for_mutants(mutant_lists_by_cds, weights, canonical_mode, effect_fn, agg="mean"):
    out = {}
    for cds_id, seq_list in mutant_lists_by_cds.items():
        scores = []
        for seq in seq_list:
            res = cds_overall_ms_weighted_from_seq_map({cds_id: seq}, weights, canonical_mode, effect_fn, agg=agg)
            scores.append(res[cds_id]["score"])
        out[cds_id] = scores
    return out


def _codon_usage_by_aa(cds_seq, keep_start=True):
    cds_seq = cds_seq.upper()
    L = len(cds_seq) - (len(cds_seq) % 3)
    cds_seq = cds_seq[:L]

    usage = defaultdict(Counter)
    start_idx = 1 if keep_start else 0

    for idx, s in enumerate(range(0, L, 3)):
        codon = cds_seq[s:s+3]
        aa = CODON_TO_AA.get(codon)
        if not aa or aa == '*':
            continue
        if idx < start_idx:
            continue
        usage[aa][codon] += 1

    return usage


def generate_synonymous_variants_from_cds_seq(
    cds_seq,
    n_variants,
    rng,
    keep_start=True,
    syn_model="empirical",
    force_change=True,
):
    cds_seq = cds_seq.upper()
    L = len(cds_seq) - (len(cds_seq) % 3)
    cds_seq = cds_seq[:L]
    if L < 3:
        return []

    codons = [cds_seq[i:i+3] for i in range(0, L, 3)]
    n_codons = len(codons)

    usage = _codon_usage_by_aa(cds_seq, keep_start=keep_start) if syn_model == "empirical" else None

    variants = []
    start_pos = 1 if keep_start else 0

    for _ in range(n_variants):
        new_codons = codons.copy()

        for pos in range(start_pos, n_codons):
            ref_codon = codons[pos]
            aa0 = CODON_TO_AA.get(ref_codon)
            if not aa0 or aa0 == '*':
                continue

            synonyms = AA_TO_CODONS.get(aa0, [])
            if len(synonyms) <= 1:
                continue

            if syn_model == "uniform":
                choices = [c for c in synonyms if (c != ref_codon) or (not force_change)]
                if not choices:
                    continue
                new_codons[pos] = rng.choice(choices)

            else:
                counts = usage.get(aa0, Counter())
                probs = np.array([counts.get(c, 0) for c in synonyms], dtype=float)

                if probs.sum() <= 0:
                    choices = [c for c in synonyms if (c != ref_codon) or (not force_change)]
                    if not choices:
                        continue
                    new_codons[pos] = rng.choice(choices)
                else:
                    probs = probs / probs.sum()
                    if force_change:
                        mask = np.array([1.0 if c != ref_codon else 0.0 for c in synonyms], dtype=float)
                        probs2 = probs * mask
                        if probs2.sum() <= 0:
                            choices = [c for c in synonyms if c != ref_codon]
                            if not choices:
                                continue
                            new_codons[pos] = rng.choice(choices)
                        else:
                            probs2 = probs2 / probs2.sum()
                            new_codons[pos] = rng.choice(synonyms, p=probs2)
                    else:
                        new_codons[pos] = rng.choice(synonyms, p=probs)

        variants.append("".join(new_codons))

    return variants


def generate_fully_synonymous_variants(
    cds_map,
    n_per_cds,
    rng_seed,
    keep_start,
    syn_model,
    force_change,
):
    rng = np.random.default_rng(rng_seed)
    out = {}
    for cds_id, seq in cds_map.items():
        out[cds_id] = generate_synonymous_variants_from_cds_seq(
            seq,
            n_variants=n_per_cds,
            rng=rng,
            keep_start=keep_start,
            syn_model=syn_model,
            force_change=force_change,
        )
    return out


def plot_synonymous_histograms_per_cds(syn_scores_by_cds, ref_scores_by_cds, out_dir, file_tag, metric, bins=30):
    os.makedirs(out_dir, exist_ok=True)
    for cds_id, syn_scores in syn_scores_by_cds.items():
        if not syn_scores:
            continue
        arr = np.asarray(syn_scores, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        if hi == lo:
            hi = lo + 1e-6

        plt.figure()
        plt.hist(arr, bins=bins, range=(lo, hi), alpha=0.65,
                 label=f"synonymous (n={len(arr)})")

        ref_score = ref_scores_by_cds.get(cds_id, {}).get("score", None)
        if ref_score is not None:
            plt.axvline(ref_score, linestyle="--", linewidth=1.5,
                        label=f"reference = {ref_score:.6f}")

        plt.xlabel("weighted mse score")
        plt.ylabel("number of variants")
        plt.title(f"{file_tag} — {metric} — cds {cds_id}: synonymous score distribution")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{file_tag}__cds_{cds_id}__syn_hist.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in header) + "\n")


def side_label(ref, median, tol=1e-12):
    if ref is None or median is None:
        return ""
    if not np.isfinite(ref) or not np.isfinite(median):
        return ""
    if abs(ref - median) <= tol:
        return "EQUAL"
    return "LEFT" if ref < median else "RIGHT"


def run_one_metric(metric, args, script_dir, weights, canonical_mode, out_dir_metric):
    effect_fn = METRIC_TO_EFFECT[metric]
    combined_rows = []
    qc_rows = []

    os.makedirs(out_dir_metric, exist_ok=True)

    for gb in args.genes:
        gb_path = gb if os.path.isabs(gb) else os.path.join(script_dir, gb)
        file_tag = os.path.splitext(os.path.basename(gb_path))[0]

        parsed = parse_cds_with_flanks(gb_path)
        cds_map_all = build_cds_sequences(parsed)

        cds_map = {}
        for cds_id, seq in cds_map_all.items():
            if args.qc:
                passed, info = qc_check_cds(
                    seq,
                    weights=weights,
                    canonical_mode=canonical_mode,
                    min_codons=args.min_codons,
                    require_multiple_of_3=bool(args.require_multiple_of_3),
                    max_den0_frac=args.max_den0_frac,
                    disallow_internal_stops=bool(args.disallow_internal_stops),
                )
                qc_rows.append({
                    "metric": metric,
                    "syn_model": args.syn_model,
                    "canonical_mode": canonical_mode,
                    "file_tag": file_tag,
                    "cds_id": cds_id,
                    "passed": int(passed),
                    **info,
                })
                if not passed:
                    continue
            cds_map[cds_id] = seq

        if not cds_map:
            continue

        ref_scores = cds_overall_ms_weighted_from_seq_map(
            cds_map, weights, canonical_mode, effect_fn, agg=args.agg
        )

        syn_variants = generate_fully_synonymous_variants(
            cds_map=cds_map,
            n_per_cds=args.n_syn,
            rng_seed=args.seed,
            keep_start=bool(args.keep_start),
            syn_model=args.syn_model,
            force_change=bool(args.force_change),
        )

        syn_scores = cds_overall_ms_weighted_for_mutants(
            syn_variants, weights, canonical_mode, effect_fn, agg=args.agg
        )

        if args.make_hists:
            plot_synonymous_histograms_per_cds(
                syn_scores, ref_scores,
                out_dir=out_dir_metric,
                file_tag=file_tag,
                metric=metric,
                bins=args.bins
            )

        if args.write_per_gene_summary:
            sum_rows = []
            for cds_id, syn_list in syn_scores.items():
                arr = np.asarray(syn_list, dtype=float) if syn_list else None
                ref = float(ref_scores.get(cds_id, {}).get("score", np.nan))
                if syn_list:
                    sum_rows.append({
                        "file_tag": file_tag,
                        "cds_id": cds_id,
                        "n_syn": len(arr),
                        "ref_score": ref,
                        "min": float(arr.min()),
                        "median": float(np.median(arr)),
                        "max": float(arr.max()),
                        "mean": float(arr.mean()),
                        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    })
                else:
                    sum_rows.append({
                        "file_tag": file_tag,
                        "cds_id": cds_id,
                        "n_syn": 0,
                        "ref_score": ref,
                        "min": "",
                        "median": "",
                        "max": "",
                        "mean": "",
                        "std": "",
                    })
            path = os.path.join(out_dir_metric, f"{file_tag}__summary.csv")
            write_csv(
                path,
                header=["file_tag","cds_id","n_syn","ref_score","min","median","max","mean","std"],
                rows=sum_rows
            )

        for cds_id, syn_list in syn_scores.items():
            ref = float(ref_scores.get(cds_id, {}).get("score", np.nan))
            if syn_list:
                arr = np.asarray(syn_list, dtype=float)
                med = float(np.median(arr))
                side = side_label(ref, med)
                combined_rows.append({
                    "metric": metric,
                    "syn_model": args.syn_model,
                    "canonical_mode": canonical_mode,
                    "file_tag": file_tag,
                    "cds_id": cds_id,
                    "n_syn": len(arr),
                    "ref_score": ref,
                    "syn_min": float(arr.min()),
                    "syn_median": med,
                    "syn_max": float(arr.max()),
                    "syn_mean": float(arr.mean()),
                    "syn_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "ref_side_vs_syn_median": side,
                })
            else:
                combined_rows.append({
                    "metric": metric,
                    "syn_model": args.syn_model,
                    "canonical_mode": canonical_mode,
                    "file_tag": file_tag,
                    "cds_id": cds_id,
                    "n_syn": 0,
                    "ref_score": ref,
                    "syn_min": "",
                    "syn_median": "",
                    "syn_max": "",
                    "syn_mean": "",
                    "syn_std": "",
                    "ref_side_vs_syn_median": "",
                })

    return combined_rows, qc_rows


def compute_side_counts_per_metric(combined_rows):
    counts = defaultdict(lambda: Counter())
    for r in combined_rows:
        m = r.get("metric", "")
        side = r.get("ref_side_vs_syn_median", "")
        if side:
            counts[m][side] += 1

    out = []
    for m in sorted(counts.keys()):
        c = counts[m]
        out.append({
            "metric": m,
            "LEFT": c.get("LEFT", 0),
            "RIGHT": c.get("RIGHT", 0),
            "EQUAL": c.get("EQUAL", 0),
            "total_nonempty": c.get("LEFT", 0) + c.get("RIGHT", 0) + c.get("EQUAL", 0),
        })
    return out


def compute_concordance_same_side(combined_rows, metrics, exclude_equal=True):
    key_to_side = defaultdict(dict)
    for r in combined_rows:
        key = (r.get("file_tag",""), str(r.get("cds_id","")))
        metric = r.get("metric","")
        side = r.get("ref_side_vs_syn_median","")
        if metric and side:
            key_to_side[key][metric] = side

    out = []
    for a, b in itertools.combinations(metrics, 2):
        same = 0
        total = 0
        for key, d in key_to_side.items():
            sa = d.get(a, "")
            sb = d.get(b, "")
            if not sa or not sb:
                continue
            if exclude_equal:
                if sa not in ("LEFT","RIGHT") or sb not in ("LEFT","RIGHT"):
                    continue
            total += 1
            if sa == sb:
                same += 1
        prob = (same / total) if total > 0 else float("nan")
        out.append({
            "metric_a": a,
            "metric_b": b,
            "exclude_equal": int(exclude_equal),
            "n_compared": total,
            "n_same_side": same,
            "p_same_side": prob,
        })
    return out


def plot_bar_ref_side_by_metric(side_rows, out_path):
    metrics = [r["metric"] for r in side_rows]
    lefts = [r["LEFT"] for r in side_rows]
    rights = [r["RIGHT"] for r in side_rows]
    equals = [r["EQUAL"] for r in side_rows]

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, lefts, width, label="LEFT")
    plt.bar(x, rights, width, label="RIGHT")
    plt.bar(x + width, equals, width, label="EQUAL")
    plt.xticks(x, metrics)
    plt.ylabel("count")
    plt.title("Reference side vs synonymous median (by metric)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heatmap_concordance(concord_rows, metrics, out_path):
    idx = {m:i for i,m in enumerate(metrics)}
    mat = np.full((len(metrics), len(metrics)), np.nan, dtype=float)
    for i in range(len(metrics)):
        mat[i, i] = 1.0

    for r in concord_rows:
        a = r["metric_a"]
        b = r["metric_b"]
        p = r["p_same_side"]
        ia, ib = idx[a], idx[b]
        mat[ia, ib] = p
        mat[ib, ia] = p

    plt.figure(figsize=(7, 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="P(same side)")
    plt.xticks(np.arange(len(metrics)), metrics, rotation=45, ha="right")
    plt.yticks(np.arange(len(metrics)), metrics)
    plt.title("Concordance heatmap: P(same side) across metrics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--metrics", nargs="*", default=["PR","GRANTHAM","BLOSUM62","PMBEC"])
    p.add_argument("--syn_model", default="empirical", choices=["empirical","uniform"])
    p.add_argument("--weights", default="mutation_rate_methylation_bins.txt")
    p.add_argument("--n_syn", type=int, default=1000)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--keep_start", type=int, default=1)
    p.add_argument("--force_change", type=int, default=1)
    p.add_argument("--agg", default="mean", choices=["mean","sum"])
    p.add_argument("--bins", type=int, default=30)

    p.add_argument("--out_root", default="")
    p.add_argument("--make_hists", type=int, default=1)
    p.add_argument("--write_per_gene_summary", type=int, default=1)

    p.add_argument("--qc", type=int, default=1)
    p.add_argument("--min_codons", type=int, default=30)
    p.add_argument("--require_multiple_of_3", type=int, default=1)
    p.add_argument("--max_den0_frac", type=float, default=0.02)
    p.add_argument("--disallow_internal_stops", type=int, default=1)

    p.add_argument("--genes", nargs="*", default=[
        "sequence-hbb.gb","sequence-hba1.gb","sequence-mb.gb",
        "sequence-polr2a.gb","sequence-cycs.gb","sequence-gapdh.gb",
        "sequence-actb.gb","sequence-hspa1l.gb","sequence-ldha.gb",
        "sequence-atp5f1a.gb"
    ])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.make_hists = bool(args.make_hists)
    args.write_per_gene_summary = bool(args.write_per_gene_summary)
    args.qc = bool(args.qc)

    metrics = []
    for m in args.metrics:
        if m not in METRIC_TO_EFFECT:
            raise ValueError(f"Unknown metric: {m}. Choose from {list(METRIC_TO_EFFECT.keys())}")
        metrics.append(m)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = args.weights if os.path.isabs(args.weights) else os.path.join(script_dir, args.weights)

    weights = load_mut96_weights(weights_path)
    canonical_mode = infer_canonical_centers(weights)

    if args.out_root:
        out_root = args.out_root
        if not os.path.isabs(out_root):
            out_root = os.path.join(script_dir, out_root)
    else:
        qc_tag = "_qc" if args.qc else ""
        out_root = os.path.join(script_dir, f"multi_seq_allMetrics_{args.syn_model}{qc_tag}_hists")

    os.makedirs(out_root, exist_ok=True)

    print(f"[INFO] Metrics = {metrics}")
    print(f"[INFO] Syn model = {args.syn_model} | canonical centers = {canonical_mode}")
    print(f"[INFO] OUT_ROOT = {out_root}")

    t0 = perf_counter()

    combined_all = []
    qc_all = []

    for metric in metrics:
        out_dir_metric = os.path.join(out_root, f"{metric}")
        print(f"\n[RUN] metric={metric} -> {out_dir_metric}")
        comb, qc_rows = run_one_metric(
            metric=metric,
            args=args,
            script_dir=script_dir,
            weights=weights,
            canonical_mode=canonical_mode,
            out_dir_metric=out_dir_metric
        )
        combined_all.extend(comb)
        qc_all.extend(qc_rows)

    summary_path = os.path.join(out_root, "summary_all_metrics.csv")
    write_csv(
        summary_path,
        header=[
            "metric","syn_model","canonical_mode",
            "file_tag","cds_id","n_syn","ref_score",
            "syn_min","syn_median","syn_max","syn_mean","syn_std",
            "ref_side_vs_syn_median"
        ],
        rows=combined_all
    )
    print(f"[SAVE] {summary_path}")

    if args.qc:
        qc_path = os.path.join(out_root, "qc_summary.csv")
        write_csv(
            qc_path,
            header=[
                "metric","syn_model","canonical_mode",
                "file_tag","cds_id","passed","reason",
                "raw_len","used_len","n_codons","internal_stops","den0_codons","den0_frac"
            ],
            rows=qc_all
        )
        print(f"[SAVE] {qc_path}")

    side_rows = compute_side_counts_per_metric(combined_all)
    side_path = os.path.join(out_root, "side_counts_per_metric.csv")
    write_csv(side_path, header=["metric","LEFT","RIGHT","EQUAL","total_nonempty"], rows=side_rows)
    print(f"[SAVE] {side_path}")

    concord_rows = compute_concordance_same_side(combined_all, metrics=metrics, exclude_equal=True)
    concord_path = os.path.join(out_root, "concordance_same_side.csv")
    write_csv(concord_path, header=["metric_a","metric_b","exclude_equal","n_compared","n_same_side","p_same_side"], rows=concord_rows)
    print(f"[SAVE] {concord_path}")

    bar_path = os.path.join(out_root, "bar_ref_side_by_metric.png")
    plot_bar_ref_side_by_metric(side_rows, bar_path)
    print(f"[SAVE] {bar_path}")

    heat_path = os.path.join(out_root, "heatmap_concordance_same_side.png")
    plot_heatmap_concordance(concord_rows, metrics, heat_path)
    print(f"[SAVE] {heat_path}")

    elapsed = perf_counter() - t0
    print(f"\nAll done. Elapsed: {elapsed:.3f}s")
    print(f"Outputs saved under: {out_root}")
