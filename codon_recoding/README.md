# Independent synonymous recoding with FoldX lookup scores

These scripts reproduce the 20-gene pilot presented in November 2026. They are separate from the older within-gene codon shuffle: each non-start amino-acid position independently receives a uniformly chosen synonymous codon. The encoded protein stays unchanged, while total codon usage may change.

## Inputs

The three pilot input files are in this directory:

- `new20_cds.fasta`: nucleotide coding sequences, with IDs matching the protein FASTA.
- `new20_protein.fasta`: translated protein sequences, same IDs.
- `foldx_20_ddg.csv`: columns `gene,mutation,ddg`; `mutation` is an amino-acid change such as `A12V`, using one-based residue positions. Include every position-specific amino-acid substitution reachable by one nucleotide change from **any** synonymous codon. FoldX scores were computed separately and are not recalculated by these scripts.

The CDS and protein FASTA files were selected from the earlier 40-protein AlphaFold3 batch, originally recorded in `genes_info_seq.csv.gz`. Each FASTA header includes the Ensembl gene ID and transcript ID. The FoldX table contains 81,175 position-specific substitutions for these 20 proteins, calculated with FoldX 5.1 from sequence-validated AlphaFold3 structures. Both scripts require Python with `numpy`, `biopython`, and `matplotlib` (the first script does not use matplotlib).

## Run

```bash
python3 codon_recoding/run_recoding_pilot.py --recodings 1000 --mutations 5000 --input-dir codon_recoding --output-dir /path/to/results
python3 codon_recoding/score_recoding_weighted_exact.py --input-dir codon_recoding --data-dir /path/to/results
```

The first script samples 5,000 nucleotide changes per sequence, using the same nucleotide positions and alternative-base ranks across the natural sequence and 1,000 recodings. The alternative base and resulting amino-acid change can differ because the codons can differ. Each actual amino-acid substitution is looked up in the precomputed FoldX table. The first output is a Monte Carlo pilot; its per-gene choices and draws are saved in NPZ files.

The second script reuses those saved recodings but **enumerates all single-base changes** for each sequence, yielding exact uniform and local sequence-context-weighted mean costs. It uses the bundled `mutation_rate_methylation_bins.txt` table, methylation bin 0, and reverse-complemented entries for G/T-centered contexts. The table's provenance and suitability for the study's target population should be verified before treating its weights as measured mutation probabilities. As only CDSs are supplied, the first and last coding nucleotides are excluded from the context-weighted calculation, and from the matching uniform exact calculation.

Both scripts use `max(min(ΔΔG, 10), 0)^2` as predicted destabilization cost. Synonymous outcomes have zero FoldX cost. Stop-gain events have no FoldX stability score; they are excluded from the mean cost and reported separately. The result does not estimate effects of synonymous recoding on translation or protein expression.
