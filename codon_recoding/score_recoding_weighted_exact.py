"""Exact context-weighted score over all single-base substitutions, per recoding."""
from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
from run_recoding_pilot import load_inputs, mutation_tables

CODE = {base: i for i, base in enumerate('ACGT')}
COMPLEMENT = {0: 3, 1: 2, 2: 1, 3: 0}


def rate_lookup(table):
    mu = np.full((4, 4, 4, 4), np.nan, dtype=np.float64)
    with table.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    zero = [row for row in rows if row['methylation_level'] == '0']
    if len(zero) != 96:
        raise ValueError(f'expected 96 methylation-bin-0 entries, got {len(zero)}')
    for row in zero:
        context = [CODE[x] for x in row['context']]
        ref, alt = CODE[row['ref']], CODE[row['alt']]
        if context[1] != ref:
            raise ValueError(f'bad table row: {row}')
        rate = float(row['mu_snp'])
        keys = [(context[0], ref, context[2], alt),
                (COMPLEMENT[context[2]], COMPLEMENT[ref],
                 COMPLEMENT[context[0]], COMPLEMENT[alt])]
        for key in keys:
            if np.isfinite(mu[key]):
                raise ValueError(f'duplicate oriented context {key}')
            mu[key] = rate
    for left in range(4):
        for ref in range(4):
            for right in range(4):
                for alt in range(4):
                    if alt != ref and not np.isfinite(mu[left, ref, right, alt]):
                        raise ValueError(f'missing rate {(left, ref, right, alt)}')
    return mu


def score_gene(seq, protein, gene_scores, choices, mu):
    synonyms, native, ddg_table, kind_table = mutation_tables(seq, protein, gene_scores)
    assert np.array_equal(choices[0], native)
    codon_codes = np.zeros((len(synonyms), 6, 3), dtype=np.uint8)
    for site, options in enumerate(synonyms):
        for j, codon in enumerate(options):
            codon_codes[site, j] = [CODE[b] for b in codon]
    all_means = np.empty(len(choices), dtype=float)
    all_stop = np.empty(len(choices), dtype=float)
    all_syn = np.empty(len(choices), dtype=float)
    nt_positions = np.arange(1, 3*len(protein)-1)
    sites, slots = nt_positions//3, nt_positions%3
    ranks = np.arange(3)
    for start in range(0, len(choices), 40):
        end = min(start+40, len(choices))
        block = choices[start:end]
        bases = codon_codes[np.arange(len(protein))[None, :], block].reshape(len(block), -1)
        left, ref, right = bases[:, nt_positions-1], bases[:, nt_positions], bases[:, nt_positions+1]
        alt = ranks[None, None, :] + (ranks[None, None, :] >= ref[:, :, None])
        weights = mu[left[:, :, None], ref[:, :, None], right[:, :, None], alt]
        effects = ddg_table[sites[None, :, None], block[:, sites, None],
                            slots[None, :, None], ranks[None, None, :]]
        kinds = kind_table[sites[None, :, None], block[:, sites, None],
                           slots[None, :, None], ranks[None, None, :]]
        if np.any(kinds < 0) or np.any(np.isnan(effects[kinds != 2])):
            raise ValueError('missing effect')
        total = weights.sum(axis=(1, 2))
        non_stop = kinds != 2
        cost = np.minimum(np.maximum(effects, 0), 10)**2
        all_means[start:end] = np.nansum(weights*cost, axis=(1, 2)) / np.sum(weights*non_stop, axis=(1, 2))
        all_stop[start:end] = np.sum(weights*(kinds == 2), axis=(1, 2))/total
        all_syn[start:end] = np.sum(weights*(kinds == 0), axis=(1, 2))/total
    return all_means, all_stop, all_syn


def main(input_dir, data_dir, rate_table):
    cds, proteins, scores = load_inputs(input_dir)
    mu = rate_lookup(rate_table)
    rows=[]
    for gene, seq in cds.items():
        data=np.load(data_dir/f'{gene}.npz')
        means, stop, syn = score_gene(seq, proteins[gene], scores[gene], data['codon_choices'], mu)
        uniform_means, uniform_stop, uniform_syn = score_gene(
            seq, proteins[gene], scores[gene], data['codon_choices'], np.ones_like(mu))
        percentile=100*float(np.mean(means[1:]<means[0]))
        uniform_percentile=100*float(np.mean(uniform_means[1:]<uniform_means[0]))
        rows.append(dict(gene=gene, residues=len(proteins[gene]),
                         uniform_exact_percentile=uniform_percentile,
                         context_weighted_exact_percentile=percentile,
                         exact_native_mean_cost=float(means[0]),
                         exact_recoding_mean_cost=float(np.mean(means[1:])),
                         exact_native_syn_fraction=float(syn[0]),
                         exact_recoding_syn_fraction=float(np.mean(syn[1:])),
                         exact_native_stop_fraction=float(stop[0]),
                         exact_recoding_stop_fraction=float(np.mean(stop[1:]))))
        np.savez_compressed(data_dir/f'{gene}_weighted_exact.npz',mean_cost=means,
                            stop_fraction=stop,syn_fraction=syn,
                            uniform_mean_cost=uniform_means,
                            uniform_stop_fraction=uniform_stop,
                            uniform_syn_fraction=uniform_syn)
        print(gene,'uniform exact',round(uniform_percentile,1),
              'weighted exact',round(percentile,1),flush=True)
    with (data_dir/'weighted_exact_comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    names=[r['gene'] for r in sorted(rows,key=lambda r:r['context_weighted_exact_percentile'])]
    lookup={r['gene']:r for r in rows}
    fig,ax=plt.subplots(figsize=(8.8,6.2));y=np.arange(len(names))
    old=[lookup[g]['uniform_exact_percentile'] for g in names]
    new=[lookup[g]['context_weighted_exact_percentile'] for g in names]
    for yi,a,b in zip(y,old,new):ax.plot([a,b],[yi,yi],color='0.7',lw=1,zorder=1)
    ax.scatter(old,y,facecolors='white',edgecolors='black',s=45,label='Uniform exact',zorder=3)
    ax.scatter(new,y,color='black',marker='s',s=28,label='Context-weighted exact',zorder=4)
    ax.axvline(50,color='0.3',ls='--',lw=.8);ax.set_yticks(y,names,fontsize=8);ax.invert_yaxis()
    ax.set_xlim(0,100);ax.set_xlabel('Native mean-cost percentile among 1,000 recodings')
    ax.set_title('Twenty genes: uniform vs context-weighted enumeration')
    ax.legend(frameon=False,loc='lower right');ax.grid(axis='x',color='0.9',lw=.5)
    fig.tight_layout();fig.savefig(data_dir/'uniform_vs_weighted_exact.png',dpi=180);plt.close(fig)
    print(data_dir/'weighted_exact_comparison.csv')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=ROOT)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--rate-table', type=Path, default=ROOT / 'mutation_rate_methylation_bins.txt')
    args = parser.parse_args()
    main(args.input_dir, args.data_dir, args.rate_table)
