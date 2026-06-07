import argparse
import shutil
from collections import Counter
from pathlib import Path


def copy_filtered_fasta(src, dst, keep_ids):
    write = False
    with open(src, "r", encoding="utf-8") as inp, open(dst, "w", encoding="utf-8") as out:
        for line in inp:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0]
                write = seq_id in keep_ids
            if write:
                out.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="NPInter4.0_network")
    parser.add_argument("--top-rna", type=int, default=1000)
    parser.add_argument("--top-protein", type=int, default=150)
    parser.add_argument("--max-edges", type=int, default=8000)
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args()

    src = Path(args.data_dir) / args.dataset
    pairs = []
    with open(src / "interactions.txt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                pairs.append(tuple(line.strip().split("\t")[:2]))
    rna_degree = Counter(r for r, _ in pairs)
    pro_degree = Counter(p for _, p in pairs)
    keep_rna = {x for x, _ in rna_degree.most_common(args.top_rna)}
    keep_pro = {x for x, _ in pro_degree.most_common(args.top_protein)}
    sub = [(r, p) for r, p in pairs if r in keep_rna and p in keep_pro]
    sub.sort(key=lambda x: (-(rna_degree[x[0]] + pro_degree[x[1]]), x[0], x[1]))
    sub = sub[: args.max_edges]
    rna_ids = sorted({r for r, _ in sub})
    pro_ids = sorted({p for _, p in sub})

    suffix = args.suffix or f"degree_r{args.top_rna}_p{args.top_protein}_e{args.max_edges}"
    dst = Path(args.data_dir) / f"{args.dataset}_{suffix}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / "c1c2c3" / "dataset").mkdir(parents=True)
    (dst / "c1c2c3" / "save").mkdir(parents=True)
    (dst / "c1c2c3" / "result").mkdir(parents=True)

    with open(dst / "interactions.txt", "w", encoding="utf-8") as fh:
        for r, p in sub:
            fh.write(f"{r}\t{p}\n")
    (dst / "lncRNA_ID.txt").write_text("\n".join(rna_ids) + "\n", encoding="utf-8")
    (dst / "protein_ID.txt").write_text("\n".join(pro_ids) + "\n", encoding="utf-8")
    copy_filtered_fasta(src / "lncRNA_seq.fa", dst / "lncRNA_seq.fa", set(rna_ids))
    copy_filtered_fasta(src / "protein_seq.fa", dst / "protein_seq.fa", set(pro_ids))
    print(dst)
    print(f"edges={len(sub)} rna={len(rna_ids)} protein={len(pro_ids)}")
    print(f"top_rna={args.top_rna} top_protein={args.top_protein}")


if __name__ == "__main__":
    main()
