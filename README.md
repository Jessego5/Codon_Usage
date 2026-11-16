# Codon_Usage
This is for Ragsdale Lab's Codon Usage Project Fall 2025

## Gene table cleaning and FoldX gene selection

Run from the repository root (standard-library Python only):

```bash
python3 clean_genes_info.py      # genes_info.csv.gz -> data/genes_info_clean.csv.gz
python3 select_foldx_genes.py    # -> data/foldx_gene_selection.csv (queries the PDBe API)
```

- `data/genes_info_clean.csv.gz`: `genes_info.csv.gz` with 312 rows removed whose CDS fails basic checks (length not a multiple of 3, no ATG start, no terminal stop, internal stop, or non-ACGT bases). 15,855 genes remain; all 77 columns are kept.
- `data/genes_info_excluded.csv`: the 312 removed genes and the reason(s) for each.
- `data/foldx_gene_selection.csv`: 189 genes proposed for FoldX. Candidates have an X-ray or cryo-EM structure, 50-500 residues, and no membrane GO terms. They are split into tertiles of median TPM across tissues (low < 5.87, mid 5.87-25.93, high > 25.93). 60 per tertile are drawn at random (seed 42) among genes whose best PDBe/SIFTS chain covers at least 80% of the UniProt sequence at 3.0 Å resolution or better, taking single-polypeptide entries first. Nine of the earlier ten genes are added and labelled `existing_ours` (POLR2A is absent from `genes_info`). The `reason` column marks sampled and existing genes; keep the existing genes out of the tertile comparison because they were not randomly sampled.
