import re

#NOTE: one genebank file may contain multiple CDS (code for different proteins)
def parse_cds_with_flanks(paths):
    # allow a single path or a list of paths
    if isinstance(paths, str):
        paths = [paths]

    #build complement map and reverse-complement helper
    comp = str.maketrans("acgtnACGTN", "tgcANtgcaN")
    def revcomp(s): return s.translate(comp)[::-1]

    out = []
    for path in paths:
        #read the whole file
        text = open(path, "r", encoding="utf-8").read()

        #extract raw sequence between ORIGIN and //; keep only A/C/G/T/N; make lowercase
        seq = re.sub(r"[^acgtnACGTN]", "", re.search(r"ORIGIN(.*)//", text, re.DOTALL).group(1)).lower()
        n = len(seq)

        # extract the FEATURES block
        features = re.search(r"FEATURES(.*?)ORIGIN", text, re.DOTALL).group(1)

        #collect complete cds location expressions
        cds_locs = []
        for block in re.finditer(r"^\s*CDS\s+([^\n]+)", features, re.MULTILINE):
            loc_line = block.group(1).strip()
            pos = block.end()
            #pull in subsequent indented lines until hit a qualifier ('/', (like /gene=..., /product=...))
            while True:
                cm = re.match(r"\n\s+([^/\n][^\n]*)", features[pos:])
                if not cm: break
                loc_line += cm.group(1).strip()
                pos += cm.end()
            cds_locs.append(loc_line)

        #parse each cds location to exon intervals
        for i, loc in enumerate(cds_locs):
            #drop qualifiers after '/', remove whitespace
            s = re.sub(r"\s+", "", loc.split("/", 1)[0])
            # detect complement (minus strand) and strip wrapper
            is_comp = s.startswith("complement(")
            if is_comp: s = s[len("complement("):-1]

            # detect multi-exon join(...) and strip wrapper
            if s.startswith("join("): s = s[5:-1]

            # split into exon tokens
            parts = s.split(",")
            for j, token in enumerate(parts):
                m = re.match(r"(\d+)\.\.(\d+)", token)
                if not m: continue  # skip anything not in start..end form

                # genbank is 1-based inclusive; add ±1-nt flanks, clamped to [1, n]
                start, end = int(m.group(1)), int(m.group(2))
                fs, fe = max(1, start-1), min(n, end+1)

                #slice sequence
                frag = seq[fs-1:fe]

                #if minus strand, return fragment in coding orientation
                if is_comp: frag = revcomp(frag)
                out.append({
                    "cds_index": i,
                    "exon_index": j,
                    "strand": "-" if is_comp else "+", #reverse-complement needed if -
                    "start_1based": fs,
                    "end_1based": fe,
                    "seq": frag
                })
    return out


if __name__ == "__main__":
    print(parse_cds_with_flanks("sequence-hbb.gb"))
    print(parse_cds_with_flanks("sequence-hba1.gb"))
    print(parse_cds_with_flanks("sequence-hspa1l.gb"))