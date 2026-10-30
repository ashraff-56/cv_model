"""Extract selected Food-101 classes from the Hugging Face parquet mirror (ethz/food101).

The parquet rows keep the original JPEG bytes and file names, so the output layout
`<out>/images/<class>/<id>.jpg` matches the official tarball and the Kaggle copy
(kmader/food41). That keeps manifest paths identical on every machine.

    python tools/extract_food101_parquet.py --src data/raw/hf_food101 --out data/raw/food101 \
        --classes samosa pizza hamburger
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def class_names(parquet_file: Path) -> list[str]:
    meta = pq.ParquetFile(parquet_file).schema_arrow.metadata[b"huggingface"]
    return json.loads(meta)["info"]["features"]["label"]["names"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--classes", nargs="*", help="Food-101 class names; default: all")
    args = ap.parse_args()

    shards = sorted(args.src.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards in {args.src}")
    names = class_names(shards[0])
    wanted = set(args.classes or names)
    unknown = wanted - set(names)
    if unknown:
        raise SystemExit(f"unknown Food-101 classes: {sorted(unknown)}")

    written = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["image", "label"])
            for row in table.to_pylist():
                cls = names[row["label"]]
                if cls not in wanted:
                    continue
                dest = args.out / "images" / cls / Path(row["image"]["path"]).name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(row["image"]["bytes"])
                written += 1
        print(f"{shard.name}: total written {written}")
    (args.out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
