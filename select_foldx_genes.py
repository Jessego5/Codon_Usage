"""Pick genes for the FoldX expansion.

Steps
1. Candidate pool from data/genes_info_clean.csv.gz:
   pdb_count > 0, X-ray or cryo-EM structure, 50-500 aa,
   no membrane GO terms (GO:0016021 integral membrane, GO:0005886 plasma membrane).
2. Split the pool into expression tertiles (low / mid / high) by the median
   TPM across all GTEx tissue columns.
3. For a random sample of genes per bin, ask PDBe (SIFTS best_structures) for
   their structures. A gene passes if some X-ray/EM chain covers
   >= MIN_COVERAGE of the UniProt sequence at resolution <= MAX_RESOLUTION.
   The best chain is the one with the highest coverage, then the best resolution.
4. Within each bin, take single-polypeptide entries first (homo-oligomers count
   as single), then fill with multi-chain entries, up to PER_BIN genes.
5. Always add the existing genes (our 10 + Jiarui's list, if present).

Outputs
  data/foldx_gene_selection.csv   one row per selected gene, with the reason
  data/pdbe_cache/                cached API responses, so reruns are offline
"""
import csv
import gzip
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(sys.maxsize)

SRC = "data/genes_info_clean.csv.gz"
OUT = "data/foldx_gene_selection.csv"
CACHE = "data/pdbe_cache"
JIARUI_FILE = "data/jiarui_genes.txt"  # one UniProt ID per line

SEED = 42
PER_BIN = 60
SAMPLE_PER_BIN = 200  # genes queried per bin before choosing PER_BIN
MIN_COVERAGE = 0.80
MAX_RESOLUTION = 3.0
MIN_AA, MAX_AA = 50, 500
MEMBRANE_GO = ("GO:0016021", "GO:0005886")

# Our 10 genes from the earlier runs (UniProt IDs).
OUR_GENES = {
    "P68871": "HBB", "P69905": "HBA1", "P02144": "MB", "P99999": "CYCS",
    "P60709": "ACTB", "P04406": "GAPDH", "P00338": "LDHA",
    "P25705": "ATP5F1A", "P34931": "HSPA1L", "P24928": "POLR2A",
}

API = "https://www.ebi.ac.uk/pdbe/api"


def fetch_json(url, cache_name):
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:  # PDBe answers 404 when there is no mapping
                data = {}
                break
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"failed: {url}")
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def best_chain(uniprot):
    data = fetch_json(f"{API}/mappings/best_structures/{uniprot}",
                      f"best_{uniprot}.json")
    hits = [
        s for s in data.get(uniprot, [])
        if s.get("experimental_method", "").lower().startswith(("x-ray", "electron"))
        and s.get("resolution") is not None
        and s["resolution"] <= MAX_RESOLUTION
        and s.get("coverage", 0) >= MIN_COVERAGE
    ]
    if not hits:
        return None
    return sorted(hits, key=lambda s: (-s["coverage"], s["resolution"]))[0]


def n_protein_entities(pdb_id):
    data = fetch_json(f"{API}/pdb/entry/molecules/{pdb_id}", f"mol_{pdb_id}.json")
    mols = data.get(pdb_id, [])
    return sum(1 for m in mols if m.get("molecule_type", "").startswith("polypeptide"))


def evaluate(gene):
    hit = best_chain(gene["uniprotswissprot"])
    if hit is None:
        return gene, None, None
    return gene, hit, n_protein_entities(hit["pdb_id"])


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    os.makedirs(CACHE, exist_ok=True)
    with gzip.open(SRC, "rt", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    tissues = cols[cols.index("cds") + 1:]

    for g in rows:
        g["aa_length"] = len(g["cds"]) // 3 - 1
        vals = [v for v in (to_float(g[t]) for t in tissues) if v is not None]
        g["median_tpm"] = statistics.median(vals) if vals else None

    jiarui = set()
    if os.path.exists(JIARUI_FILE):
        with open(JIARUI_FILE) as f:
            jiarui = {line.strip() for line in f if line.strip()}
    existing = set(OUR_GENES) | jiarui

    pool = [
        g for g in rows
        if (to_float(g["pdb_count"]) or 0) > 0
        and ("X-ray" in g["methods"] or "Electron" in g["methods"])
        and MIN_AA <= g["aa_length"] <= MAX_AA
        and not any(t in g["go_ids"] for t in MEMBRANE_GO)
        and g["median_tpm"] is not None
        and g["uniprotswissprot"]
        and g["uniprotswissprot"] not in existing
    ]
    tpm = sorted(g["median_tpm"] for g in pool)
    t1, t2 = tpm[len(tpm) // 3], tpm[2 * len(tpm) // 3]
    print(f"pool {len(pool)} genes; tertile cut-offs: {t1:.2f}, {t2:.2f} TPM")

    def bin_of(x):
        return "low" if x < t1 else ("mid" if x < t2 else "high")

    rng = random.Random(SEED)
    selected = []
    for b in ("low", "mid", "high"):
        genes = [g for g in pool if bin_of(g["median_tpm"]) == b]
        rng.shuffle(genes)
        sample = genes[:SAMPLE_PER_BIN]
        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(evaluate, sample))
        passed = [(g, h, n) for g, h, n in results if h is not None]
        single = [r for r in passed if r[2] == 1]
        multi = [r for r in passed if r[2] != 1]
        chosen = (single + multi)[:PER_BIN]
        print(f"{b}: {len(genes)} in bin, queried {len(sample)}, "
              f"passed {len(passed)} ({len(single)} single-chain), chose {len(chosen)}")
        for g, h, n in chosen:
            selected.append((g, h, n, b, "sampled"))

    by_uniprot = {g["uniprotswissprot"]: g for g in rows}
    for uid in sorted(existing):
        g = by_uniprot.get(uid)
        reason = "existing_ours" if uid in OUR_GENES else "existing_jiarui"
        if g is None:
            print(f"warning: existing gene {uid} not in clean table")
            continue
        _, h, n = evaluate(g)
        b = bin_of(g["median_tpm"]) if g["median_tpm"] is not None else ""
        selected.append((g, h, n, b, reason))

    fields = ["uniprot", "gene_label", "ensembl_gene_id", "ensembl_transcript_id",
              "aa_length", "median_tpm", "expr_bin", "pdb_id", "chain_id",
              "method", "resolution", "coverage", "n_protein_entities", "reason"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g, h, n, b, reason in selected:
            uid = g["uniprotswissprot"]
            w.writerow({
                "uniprot": uid,
                "gene_label": OUR_GENES.get(uid, ""),
                "ensembl_gene_id": g["ensembl_gene_id"],
                "ensembl_transcript_id": g["ensembl_transcript_id"],
                "aa_length": g["aa_length"],
                "median_tpm": round(g["median_tpm"], 3) if g["median_tpm"] is not None else "",
                "expr_bin": b,
                "pdb_id": h["pdb_id"] if h else "",
                "chain_id": h["chain_id"] if h else "",
                "method": h["experimental_method"] if h else "",
                "resolution": h["resolution"] if h else "",
                "coverage": h["coverage"] if h else "",
                "n_protein_entities": n if n is not None else "",
                "reason": reason,
            })
    print(f"wrote {len(selected)} genes to {OUT}")


if __name__ == "__main__":
    main()
