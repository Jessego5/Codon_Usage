"""Drop genes_info rows whose CDS fails basic checks.

Writes:
  data/genes_info_clean.csv.gz  - all original columns, valid CDS only
  data/genes_info_excluded.csv  - gene IDs and reasons for every dropped row
"""
import csv
import gzip
import os
import re
import sys

csv.field_size_limit(sys.maxsize)
STOPS = {"TAA", "TAG", "TGA"}


def cds_problems(seq):
    s = seq.strip().upper()
    if not s:
        return ["empty"]
    bad = []
    if len(s) % 3:
        bad.append("len_not_multiple_of_3")
    if s[:3] != "ATG":
        bad.append("no_ATG_start")
    if s[-3:] not in STOPS:
        bad.append("no_stop_at_end")
    if not re.fullmatch("[ACGT]+", s):
        bad.append("non_ACGT")
    last_full = len(s) - len(s) % 3 - 3
    if any(s[i:i + 3] in STOPS for i in range(0, last_full, 3)):
        bad.append("internal_stop")
    return bad


def main(src=None,
         out="data/genes_info_clean.csv.gz",
         excl="data/genes_info_excluded.csv"):
    if src is None:  # repo keeps the raw table at top level; local copy is in data/
        src = next(p for p in ("data/genes_info.csv.gz", "genes_info.csv.gz")
                   if os.path.exists(p))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    kept = dropped = 0
    with gzip.open(src, "rt", newline="") as fin, \
         gzip.open(out, "wt", newline="") as fout, \
         open(excl, "w", newline="") as fex:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        ex = csv.writer(fex)
        ex.writerow(["ensembl_gene_id", "ensembl_transcript_id",
                     "uniprotswissprot", "cds_length", "reasons"])
        for row in reader:
            bad = cds_problems(row["cds"])
            if bad:
                dropped += 1
                ex.writerow([row["ensembl_gene_id"], row["ensembl_transcript_id"],
                             row["uniprotswissprot"], len(row["cds"].strip()),
                             ";".join(bad)])
            else:
                row["cds"] = row["cds"].strip().upper()
                writer.writerow(row)
                kept += 1
    print(f"kept {kept}, dropped {dropped}")


if __name__ == "__main__":
    main(*sys.argv[1:])
