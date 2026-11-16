"""Twenty-gene synonymous-recoding pilot using reusable FoldX BuildModel scores.

For each gene, sample the same nucleotide positions and alternative-base ranks
for the native sequence and each independently recoded synonymous sequence.
Synonymous outcomes have ddG=0; nonsense outcomes are tracked, not assigned a
FoldX ddG. This is a uniform mutation-sampling pilot, not the prior 96-context
weighted score.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable

ROOT = Path(__file__).resolve().parent
CODON_TO_AA = dict(CodonTable.unambiguous_dna_by_id[1].forward_table)
CODON_TO_AA.update({codon: '*' for codon in CodonTable.unambiguous_dna_by_id[1].stop_codons})
AA_TO_CODONS = defaultdict(list)
for codon, aa in CODON_TO_AA.items():
    if aa != '*':
        AA_TO_CODONS[aa].append(codon)

BASES = 'ACGT'


def load_inputs(input_dir=ROOT):
    cds = {r.id: str(r.seq).upper() for r in SeqIO.parse(input_dir / 'new20_cds.fasta', 'fasta')}
    proteins = {r.id: str(r.seq).upper() for r in SeqIO.parse(input_dir / 'new20_protein.fasta', 'fasta')}
    if not cds or cds.keys() != proteins.keys():
        raise ValueError('CDS/protein FASTA IDs are missing or inconsistent')
    scores = defaultdict(dict)
    for row in csv.DictReader((input_dir / 'foldx_20_ddg.csv').open()):
        mutation = row['mutation']
        key = (int(mutation[1:-1]) - 1, mutation[-1])
        gene = row['gene']
        if key in scores[gene]:
            raise ValueError(f'duplicate FoldX score: {gene} {mutation}')
        scores[gene][key] = float(row['ddg'])
    if scores.keys() != cds.keys():
        raise ValueError('FoldX score gene IDs differ from FASTA IDs')
    return cds, proteins, scores


def mutation_tables(seq, protein, gene_scores):
    """For every position/synonymous codon, map 3x3 base changes to ddG."""
    native_codons = [seq[i:i+3] for i in range(0, len(protein)*3, 3)]
    synonymous = [AA_TO_CODONS[aa] if i else [native_codons[0]]
                  for i, aa in enumerate(protein)]
    ddg = np.full((len(protein), 6, 3, 3), np.nan, dtype=np.float32)
    kind = np.full((len(protein), 6, 3, 3), -1, dtype=np.int8)
    # kind: 0 synonymous; 1 missense; 2 nonsense.
    for pos, (aa, choices) in enumerate(zip(protein, synonymous)):
        for codon_index, codon in enumerate(choices):
            for slot in range(3):
                alternatives = [b for b in BASES if b != codon[slot]]
                for rank, base in enumerate(alternatives):
                    changed = codon[:slot] + base + codon[slot+1:]
                    to_aa = CODON_TO_AA[changed]
                    if to_aa == aa:
                        kind[pos, codon_index, slot, rank] = 0
                        ddg[pos, codon_index, slot, rank] = 0
                    elif to_aa == '*':
                        kind[pos, codon_index, slot, rank] = 2
                    else:
                        kind[pos, codon_index, slot, rank] = 1
                        if (pos, to_aa) not in gene_scores:
                            raise ValueError(f'missing FoldX score: {pos+1} {aa}>{to_aa}')
                        ddg[pos, codon_index, slot, rank] = gene_scores[pos, to_aa]
    native_choice = np.array([options.index(codon) for options, codon in zip(synonymous, native_codons)],
                             dtype=np.int8)
    return synonymous, native_choice, ddg, kind


def run(n_recodings=100, n_mutations=5000, seed=888, input_dir=ROOT, output=None):
    cds, proteins, scores = load_inputs(input_dir)
    output = output or (input_dir / f'recoding_pilot_{n_recodings}x{n_mutations}')
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for gene_index, (gene, seq) in enumerate(cds.items()):
        protein = proteins[gene]
        if str(Seq(seq).translate()).rstrip('*') != protein:
            raise ValueError(f'CDS/protein mismatch: {gene}')
        synonyms, native, ddg_table, kind_table = mutation_tables(seq, protein, scores[gene])
        rng = np.random.default_rng(seed + gene_index)
        positions = rng.integers(0, 3*len(protein), size=n_mutations)
        sites, slots = positions // 3, positions % 3
        ranks = rng.integers(0, 3, size=n_mutations)
        choices = np.empty((n_recodings+1, len(protein)), dtype=np.int8)
        choices[0] = native
        choices[:, 0] = native[0]  # fixed start codon
        for site, options in enumerate(synonyms[1:], 1):
            choices[1:, site] = rng.integers(0, len(options), size=n_recodings)
        # Validate translation and allow overall codon usage to vary.
        for row in choices:
            assert all(CODON_TO_AA[synonyms[i][int(row[i])]] == aa
                       for i, aa in enumerate(protein))
        native_counts = Counter(synonyms[i][int(native[i])] for i in range(len(protein)))
        usage_changed = sum(Counter(synonyms[i][int(row[i])] for i in range(len(protein)))
                            != native_counts for row in choices[1:])
        observed_ddg = ddg_table[sites[None, :], choices[:, sites], slots[None, :], ranks[None, :]]
        observed_kind = kind_table[sites[None, :], choices[:, sites], slots[None, :], ranks[None, :]]
        if np.any(observed_kind < 0) or np.any(np.isnan(observed_ddg[observed_kind != 2])):
            raise ValueError(f'incomplete mutation table: {gene}')
        cost = np.minimum(np.maximum(observed_ddg, 0), 10) ** 2
        means = np.nanmean(cost, axis=1)
        stop_rates = np.mean(observed_kind == 2, axis=1)
        syn_rates = np.mean(observed_kind == 0, axis=1)
        percentile = 100*float(np.mean(means[1:] < means[0]))
        np.savez_compressed(output / f'{gene}.npz', ddg=observed_ddg, kind=observed_kind,
                            mean_cost=means, stop_fraction=stop_rates,
                            synonymous_fraction=syn_rates, positions=positions,
                            alternative_rank=ranks, codon_choices=choices)
        summary.append(dict(gene=gene, residues=len(protein), n_recodings=n_recodings,
                            mutations_per_sequence=n_mutations,
                            native_mean_cost=float(means[0]),
                            recoding_mean_cost=float(np.mean(means[1:])),
                            native_percentile_below=float(percentile),
                            native_synonymous_fraction=float(syn_rates[0]),
                            recoding_synonymous_fraction=float(np.mean(syn_rates[1:])),
                            native_stop_fraction=float(stop_rates[0]),
                            recoding_stop_fraction=float(np.mean(stop_rates[1:])),
                            recodings_with_changed_codon_usage=usage_changed))
        print(gene, f'percentile={percentile:.1f}', f'changed_usage={usage_changed}/{n_recodings}', flush=True)
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (output / 'METHOD.txt').write_text(
        'Synonymous codons independently sampled uniformly at each residue from all allowed codons, '
        'including the native codon. Start codon fixed; terminal stop excluded from sampling. '
        'Each gene uses the same sampled nucleotide positions and alternative-base ranks for native '
        'and all recodings. Alternative rank is among the three bases other than the current base; '
        'the actual replacement base can therefore differ. Mutations are uniformly sampled, '
        'not weighted by the earlier 96-context rates. Synonymous ddG=0. Nonsense events have no '
        'FoldX ddG, are excluded from mean cost, and their fraction is reported separately. '
        'Cost=max(min(ddG,10),0)^2. Individual raw ddG distributions and event types are in NPZ files.\n'
    )
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recodings', type=int, default=100)
    parser.add_argument('--mutations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--input-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.recodings, args.mutations, args.seed, args.input_dir, args.output_dir)
