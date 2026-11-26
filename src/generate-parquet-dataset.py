import argparse
import json
import struct
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = {"train": "train", "val": "validation", "test": "test"}

IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
OBJECTS_TYPE = pa.struct([
    ("category", pa.list_(pa.int64())),
    ("obb", pa.list_(pa.list_(pa.list_(pa.int32())))),    # [x1,y1,...,x4,y4] pixels
    ("bbox", pa.list_(pa.list_(pa.float32()))),   # COCO [x, y, w, h] pixels
])
SCHEMA = pa.schema([
    ("image", IMAGE_TYPE),
    ("image_id", pa.string()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("objects", OBJECTS_TYPE),
])


def hf_metadata(classes):
    seq_f32 = {"feature": {"dtype": "float32", "_type": "Value"}, "_type": "Sequence"}
    features = {
        "image": {"_type": "Image"},
        "image_id": {"dtype": "string", "_type": "Value"},
        "width": {"dtype": "int32", "_type": "Value"},
        "height": {"dtype": "int32", "_type": "Value"},
        "objects": {
            "feature": {
                "category": {"names": classes, "_type": "ClassLabel"},
                "obb": seq_f32,
                "bbox": seq_f32,
            },
            "_type": "Sequence",
        },
    }
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode()}


def png_size(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a valid PNG")
    return struct.unpack(">II", data[16:24])


def parse_json(path: Path, classes, width, height, normalized):
    """
    Asume:
      {"shapes": [{"label": "car", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]}
    """
    ann = json.loads(path.read_text(encoding="utf-8"))
    cats, obbs, bboxes = [], [], []
    for o in ann["shapes"]:
        pts = o["points"]
        if normalized:
            pts = [[x * width, y * height] for x, y in pts]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cats.append(classes.index(o["label"]))
        obbs.append([[int(p[0]), int(p[1])] for p in pts])
        bboxes.append([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)])
    return {"category": cats, "obb": obbs, "bbox": bboxes}


def plan_shards(files, max_bytes):
    shards, cur, size = [], [], 0
    for f in files:
        s = f.stat().st_size
        if cur and size + s > max_bytes:
            shards.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += s
    if cur:
        shards.append(cur)
    return shards


def build_row(img: Path, classes, normalized, missing):
    data = img.read_bytes()
    w, h = png_size(data)
    js = img.with_suffix(".json")
    if js.exists():
        objects = parse_json(js, classes, w, h, normalized)
    else:
        missing.append(img.name)
        objects = {"category": [], "obb": [], "bbox": []}
    return {
        "image": {"bytes": data, "path": img.name},
        "image_id": img.stem,
        "width": w,
        "height": h,
        "objects": objects,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="folder with train/ val/ test/")
    ap.add_argument("--out", required=True, type=Path, help="output folder")
    ap.add_argument("--normalized", action="store_true",
                    help="JSON coords are normalized 0-1")
    args = ap.parse_args()

    out_dir = args.out / "parquets"
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA.with_metadata(hf_metadata(['small vehicle', 'large vehicle']))
    max_bytes = 1024 * 1024 * 1024 * 100

    for folder, split in SPLITS.items():
        src = args.root / folder
        if not src.is_dir():
            print(f"[INFO] skipping {src}", file=sys.stderr)
            continue
        files = sorted(src.glob("*.png"))
        shards = plan_shards(files, max_bytes)
        missing = []

        for i, shard in enumerate(shards):
            name = f"{split}-{i:05d}-of-{len(shards):05d}.parquet"
            with pq.ParquetWriter(out_dir / name, schema, compression="snappy") as writer:
                for j in range(0, len(shard), 32):
                    rows = [build_row(f, ['small vehicle', 'large vehicle'], args.normalized, missing)
                            for f in shard[j:j + 32]]
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            print(f"[{split}] {name}: {len(shard)} images")

        if missing:
            print(f"[INFO] {split}: {len(missing)} images without JSON: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
