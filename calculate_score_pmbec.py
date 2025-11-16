import os
import re
from collections import defaultdict
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from parse_gb import parse_cds_with_flanks

# ===== codon <-> amino acid 表 =====

CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGA': 'R', 'AGG': 'R',
    'AGT': 'S', 'AGC': 'S',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# 反向映射：氨基酸 -> 所有同义 codon
AA_TO_CODONS = {}
for c, aa in CODON_TO_AA.items():
    if aa != '*':
        AA_TO_CODONS.setdefault(aa, []).append(c)

# 20 个标准氨基酸（矩阵只管这 20 个）
AA_CANONICAL = ['A', 'R', 'N', 'D', 'C',
                'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P',
                'S', 'T', 'W', 'Y', 'V']


# ===== 读入氨基酸相似性矩阵 (PMBEC / BLOSUM 格式) =====
def load_aa_similarity_matrix(path, aa_letters=AA_CANONICAL):
    """
    读类似 BLOSUM62 的文本：
        A  R  N  D  C ...
     A  4 -1 -2 -2  0 ...
     R -1  5  0 -2 -3 ...
     ...

    - 忽略以 # 开头的注释行和空行
    - 只保留 20 个标准氨基酸
    返回:
        sim: dict[aa1][aa2] -> 相似度(浮点数，越大越相似)
        max_sim: 全矩阵中的最大相似度，用来转成“距离”
    """
    with open(path, encoding="utf-8") as f:
        raw_lines = f.readlines()

    # 去掉空行和注释
    lines = []
    for ln in raw_lines:
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith(";"):
            continue
        lines.append(ln)

    if not lines:
        raise ValueError(f"Matrix file '{path}' is empty or only comments.")

    # 第一行是列头：A R N D ...
    header_tokens = lines[0].split()
    # 只留我们关心的氨基酸
    header_cols = [tok for tok in header_tokens if tok in aa_letters]

    sim = {aa: {} for aa in aa_letters}
    max_sim = None

    # 后面每一行： 行氨基酸 + 一串数字
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        row_aa = parts[0]
        if row_aa not in aa_letters:
            continue

        # 数字数量应该和 header_tokens 一样
        values = parts[1:]
        if len(values) < len(header_tokens):
            # 略粗暴：不足就跳过该行
            continue

        # 把每个列氨基酸对应的值读进去
        for col_name, val_str in zip(header_tokens, values):
            if col_name not in aa_letters:
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue
            sim[row_aa][col_name] = val
            if max_sim is None or val > max_sim:
                max_sim = val

    if max_sim is None:
        raise ValueError(f"No numeric entries parsed from '{path}'.")

    # 确保对称（有些矩阵文件可能只保证一侧）
    for a in aa_letters:
        for b in aa_letters:
            if a == b:
                # 对角线没有就设成最大值
                sim[a].setdefault(b, max_sim)
            else:
                if b in sim[a]:
                    sim[b][a] = sim[a][b]
                elif a in sim[b]:
                    sim[a][b] = sim[b][a]

    return sim, max_sim


# ===== exon 拼接 =====
def build_cds_sequences(parsed_exons):
    """
    Input: list of dicts {"cds_index","exon_index","seq"} 含 ±1nt flanks
    Output: dict {cds_index: 拼好的 CDS 序列（已经去掉每个 exon 的 flank）}
    """
    bins = defaultdict(list)
    for e in parsed_exons:
        core = e["seq"][1:-1]  # 去掉两端 1 nt flank
        bins[e["cds_index"]].append((e["exon_index"], core))

    cds_seq = {}
    for i, parts in bins.items():
        parts.sort(key=lambda x: x[0])
        cds_seq[i] = "".join(p for _, p in parts)
    return cds_seq


# ===== 读取 96-context 突变率表 =====
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
        context, ref, alt, methyl, mu = (
            cols[0].upper(), cols[1].upper(), cols[2].upper(), cols[3], cols[4]
        )
        if str(methyl) != "0":
            continue
        try:
            mu_val = float(mu)
        except Exception:
            continue
        wt[(context, ref, alt)] = mu_val
    return wt


# ===== 基于相似矩阵的 weighted MSE per codon =====
def ms_for_codon_SUBMAT_weighted(full_seq, codon_start, weights, aa_sim, max_sim):
    """
    full_seq: 整条 CDS 序列（无 flanks）
    codon_start: 这个 codon 在 full_seq 中的起始 index
    weights: 96-context 突变率表
    aa_sim: 氨基酸相似度矩阵（字典）
    max_sim: 矩阵全局最大相似度，用来把相似度变成“距离”

    对每个可能的非同义点突变：
        dist(aa0 -> aa) = max_sim - sim[aa0][aa]
    然后返回加权均值 E[ dist^2 ]。
    """
    codon = full_seq[codon_start:codon_start + 3].upper()
    if len(codon) < 3:
        return None
    aa0 = CODON_TO_AA.get(codon)
    if aa0 in (None, '*'):
        return None

    L = len(full_seq)
    # 对这个 codon 的三个碱基，记录左右邻居
    pos_neighbors = []
    for p in range(3):
        i = codon_start + p
        left = full_seq[i - 1].upper() if i - 1 >= 0 else None
        right = full_seq[i + 1].upper() if i + 1 < L else None
        pos_neighbors.append((left, codon[p], right))

    bases = ['A', 'C', 'G', 'T']
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

            # 构造这个点突变后的新 codon
            mutated = list(codon)
            mutated[p] = alt
            mut_codon = "".join(mutated)
            aa = CODON_TO_AA.get(mut_codon)
            # 跳过 stop 或同义突变
            if not aa or aa == '*' or aa == aa0:
                continue

            sim_val = aa_sim.get(aa0, {}).get(aa, None)
            if sim_val is None:
                continue

            # 相似度 -> 距离，越不像距离越大
            dist = max_sim - sim_val
            num += mu * (dist * dist)
            den += mu

    if den == 0.0:
        return 0.0
    return num / den


# ===== overall weighted score per CDS =====
def cds_overall_ms_weighted_from_seq_map(seq_map, weights, aa_sim, max_sim, agg="mean"):
    results = {}
    for i, seq in seq_map.items():
        L = len(seq) - (len(seq) % 3)
        scores = []
        for s in range(0, L, 3):
            ms = ms_for_codon_SUBMAT_weighted(seq, s, weights, aa_sim, max_sim)
            if ms is not None:
                scores.append(ms)
        if not scores:
            results[i] = {
                "score": 0.0, "agg": agg, "n_codons": 0,
                "sum": 0.0, "mean": 0.0
            }
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


def cds_overall_ms_weighted(parsed_exons, mut96_text_path, aa_sim, max_sim, agg="mean"):
    weights = load_mut96_weights(mut96_text_path)
    cds_map = build_cds_sequences(parsed_exons)
    return cds_overall_ms_weighted_from_seq_map(cds_map, weights, aa_sim, max_sim, agg=agg)


def cds_overall_ms_weighted_for_mutants(mutant_lists_by_cds, weights, aa_sim, max_sim, agg="mean"):
    out = {}
    for i, seq_list in mutant_lists_by_cds.items():
        scores = []
        for seq in seq_list:
            res = cds_overall_ms_weighted_from_seq_map(
                {i: seq}, weights, aa_sim, max_sim, agg=agg
            )
            scores.append(res[i]["score"])
        out[i] = scores
    return out


# ===== helpers =====
def _aa_sequence(seq):
    aa = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        c = seq[i:i + 3].upper()
        aa.append(CODON_TO_AA.get(c, 'X'))
    return "".join(aa)


# ===== fully-synonymous recoding (跟之前脚本一致) =====
def generate_fully_synonymous_variants(parsed_exons,
                                       n_per_cds=1000,
                                       rng_seed=None,
                                       keep_start=True):
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
        original_codons = [seq[j:j + 3] for j in range(0, L, 3)]

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


# ===== 画直方图 + summary（跟前面脚本同结构） =====
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
        plt.hist(
            arr,
            bins=bins,
            range=(lo, hi),
            alpha=0.65,
            label=f"Synonymous (n={len(arr)})"
        )

        ref_score = ref_scores_by_cds.get(cds_id, {}).get("score", None)
        if ref_score is not None:
            plt.axvline(
                ref_score,
                linestyle="--",
                linewidth=1.5,
                label=f"Reference = {ref_score:.6f}",
            )

        plt.xlabel("Substitution-matrix–weighted MSE score")
        plt.ylabel("Number of variants")
        plt.title(f"{file_tag} — CDS {cds_id}: Synonymous score distribution")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(
            out_dir, f"{file_tag}__cds_{cds_id}__syn_hist_submat.png"
        )
        plt.savefig(out_path, dpi=150)
        plt.close()


def write_summary_csv(ref_scores_by_cds, syn_scores_by_cds, out_dir, file_tag):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{file_tag}__summary_submat.csv")
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
                row = [
                    file_tag, cds_id, 0,
                    ref_scores_by_cds.get(cds_id, {}).get("score", float("nan")),
                    "", "", "", "", ""
                ]
            f.write(",".join(map(str, row)) + "\n")


def process_genbank_file(
        gb_path,
        weights_dict,
        aa_sim,
        max_sim,
        n_syn_variants=1000,
        rng_seed=123,
        keep_start=True,
        agg="mean",
):
    parsed = parse_cds_with_flanks(gb_path)
    ref_scores_by_cds = cds_overall_ms_weighted_from_seq_map(
        build_cds_sequences(parsed), weights_dict, aa_sim, max_sim, agg=agg
    )
    syn_variants = generate_fully_synonymous_variants(
        parsed, n_per_cds=n_syn_variants, rng_seed=rng_seed, keep_start=keep_start
    )
    syn_scores_by_cds = cds_overall_ms_weighted_for_mutants(
        syn_variants, weights_dict, aa_sim, max_sim, agg=agg
    )
    return ref_scores_by_cds, syn_scores_by_cds


# ===== main =====
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
        "sequence-atp5f1a.gb",
    ]

    WEIGHTS_PATH = "mutation_rate_methylation_bins.txt"

    # 这里填你自己的矩阵文件（可以先用 BLOSUM62，之后换成 PMBEC）
    AA_MATRIX_PATH = "pmbec_or_blosum62_matrix.txt"

    OUT_DIR = "multi_seq_submat_hists"
    N_SYNONYMOUS = 1000
    RNG_SEED = 999
    KEEP_START = True
    AGGREGATION = "mean"

    t0 = perf_counter()

    # 读突变率表 & 氨基酸相似矩阵
    weights = load_mut96_weights(WEIGHTS_PATH)
    aa_sim, max_sim = load_aa_similarity_matrix(AA_MATRIX_PATH)

    os.makedirs(OUT_DIR, exist_ok=True)

    for gb in GENBANK_FILES:
        file_tag = os.path.splitext(os.path.basename(gb))[0]
        print(f"\n=== Processing {gb} ===")
        ref_scores, syn_scores_by_cds = process_genbank_file(
            gb,
            weights_dict=weights,
            aa_sim=aa_sim,
            max_sim=max_sim,
            n_syn_variants=N_SYNONYMOUS,
            rng_seed=RNG_SEED,
            keep_start=KEEP_START,
            agg=AGGREGATION,
        )

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
