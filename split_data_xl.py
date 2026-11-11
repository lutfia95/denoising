"""Stream both annotation schemas into a shared store and leakage-safe manifests.

Run from any directory: python split_data_xl.py --config configs/train_xl.yml
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.xl_data import (XLFeatureConfig, arrow_rows, canonical_run, make_features, prepare_arrays,
                         read_config, resolve_path, write_json)


SCHEMAS = {
    "legacy": {"file": "PeakListFileName", "scan": "scan", "mz": "mz_arr",
               "intensity": "int_arr", "charge": "Charge", "precursor": "exp m/z"},
    "annotations": {"file": "run_name", "scan": "scan_number", "mz": "peak_mz",
                    "intensity": "peak_intensity", "charge": "precursor_charge", "precursor": "precursor_mz"},
}
STORE_SCHEMA = pa.schema([
    ("mz_arr", pa.list_(pa.float64())), ("int_arr", pa.list_(pa.float64())),
    ("annotation_mask", pa.list_(pa.bool_())), ("Charge", pa.int32()), ("exp m/z", pa.float64()),
])


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            # Stable representatives, independent of hash iteration order.
            self.parent[max(a, b)] = min(a, b)


def assign_groups(frame, fractions, seed, trials=256):
    names = ["train", "val", "test"]
    fractions = np.asarray(fractions, dtype=float)
    if len(fractions) != 3 or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("Three positive split fractions summing to 1 are required")
    counts = pd.crosstab(frame.group_id, frame.source).sort_index()
    if len(counts) < 3 or (counts.gt(0).sum(axis=0) < 3).any():
        raise ValueError("Need at least three independent groups per source after leakage grouping")
    # Search group-preserving assignments using source row counts, never targets.
    rng = np.random.default_rng(seed)
    sizes = np.maximum(1, np.floor(len(counts) * fractions[:2]).astype(int))
    if sizes.sum() >= len(counts):
        sizes[0] = len(counts) - sizes[1] - 1
    values = counts.to_numpy()
    total = values.sum(0)
    best, best_score = None, float("inf")
    for _ in range(trials):
        order = rng.permutation(len(counts))
        parts = np.split(order, np.cumsum(sizes))
        amounts = np.stack([values[p].sum(0) for p in parts])
        if (amounts == 0).any():
            continue
        score = float(np.square(amounts / total - fractions[:, None]).sum())
        if score < best_score:
            best_score = score
            best = {counts.index[j]: name for name, part in zip(names, parts) for j in part}
    if best is None:
        raise ValueError("Could not put every source into every split; inspect connected groups")
    return frame.group_id.map(best), best_score


def write_conflict_examples(out, metadata, pairs):
    store = pq.ParquetFile(out / "spectra.parquet")
    examples = []
    for a, b in pairs:
        items = [metadata[a], metadata[b]]
        masks = []
        detail = {}
        for suffix, item in zip(("a", "b"), items):
            column = store.read_row_group(item["row_group"], columns=["annotation_mask"]).column(0)
            masks.append(np.asarray(column[item["row_offset"]].as_py(), dtype=bool))
            for key in ("source", "source_row", "project", "run_id", "scan", "fdr", "row_group", "row_offset"):
                detail[f"{key}_{suffix}"] = item[key]
        detail.update(peaks=len(masks[0]), positive_a=int(masks[0].sum()), positive_b=int(masks[1].sum()),
                      shared_positive=int((masks[0] & masks[1]).sum()),
                      disagreeing_peaks=int((masks[0] != masks[1]).sum()))
        examples.append(detail)
    pd.DataFrame(examples).to_parquet(out / "annotation_conflict_examples.parquet", index=False)


def split_data(config, root, output_dir=None, limit_per_source=None):
    settings = config["split"]
    features = XLFeatureConfig(**config.get("features", {}))
    out = resolve_path(root, output_dir or config["data"]["split_dir"])
    # Never silently reuse partially written or unrelated split artifacts.
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")
    if settings.get("group_by", "project") not in ("project", "run"):
        raise ValueError("split.group_by must be project or run")
    policy = settings.get("conflicting_labels", "best_fdr")
    if policy not in ("best_fdr", "exclude"):
        raise ValueError("conflicting_labels must be best_fdr or exclude")
    sources = config["data"]["sources"]
    if len({s["name"] for s in sources}) != len(sources):
        raise ValueError("Source names must be unique")
    row_group_size = int(settings.get("row_group_size", 128))
    if row_group_size < 1 or not 0 < float(settings.get("minimum_weight", 0.5)) <= 1:
        raise ValueError("row_group_size must be positive and minimum_weight must be in (0, 1]")
    uf, runs, fingerprints, identities, peak_groups = UnionFind(), {}, {}, {}, {}
    metadata, store_rows = [], []
    audit = Counter()
    examples, conflicts = [], set()
    conflict_pairs, conflict_example_counts = [], Counter()
    writer = pq.ParquetWriter(out / "spectra.parquet", STORE_SCHEMA, compression="zstd")
    row_group = 0

    def flush():
        nonlocal row_group
        if store_rows:
            writer.write_table(pa.Table.from_pylist(store_rows, schema=STORE_SCHEMA), row_group_size=len(store_rows))
            store_rows.clear()
            row_group += 1

    try:
        for source in sources:
            path = resolve_path(root, source["path"])
            schema = SCHEMAS[source["schema"]]
            # Do not prefetch multi-GB row groups into RAM.
            parquet = pq.ParquetFile(path, memory_map=True, pre_buffer=False, buffer_size=65536)
            columns = list(dict.fromkeys([*schema.values(), "annotation_mask", "SearchID", "fdr",
                                          *( ["dataset_id"] if source["schema"] == "annotations" else [])]))
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            print(f"Reading {source['name']}: {parquet.metadata.num_rows:,} rows", flush=True)
            seen = 0
            for batch in parquet.iter_batches(batch_size=row_group_size, columns=columns, use_threads=False):
                for row in arrow_rows(batch):
                    if limit_per_source is not None and seen >= limit_per_source:
                        break
                    original_row = seen
                    seen += 1
                    audit[f"{source['name']}:input_rows"] += 1
                    try:
                        if row["annotation_mask"] is None:
                            raise ValueError("Missing annotation mask")
                        mz, intensity, target, _ = prepare_arrays(row[schema["mz"]], row[schema["intensity"]], row["annotation_mask"])
                        charge, precursor = float(row[schema["charge"]]), float(row[schema["precursor"]])
                        peak, spectrum, hist = make_features(mz, intensity, charge, precursor, features)
                        del peak
                        scan_value = float(row[schema["scan"]])
                        if not np.isfinite(scan_value) or scan_value < 0 or int(scan_value) != scan_value:
                            raise ValueError("scan number must be a nonnegative integer")
                        scan = int(scan_value)
                        if row[schema["file"]] is None or not canonical_run(row[schema["file"]]):
                            raise ValueError("Missing run identity")
                        if row["SearchID"] is None or not str(row["SearchID"]).strip():
                            raise ValueError("Missing project identity")
                        fdr = float(row["fdr"])
                        if not np.isfinite(fdr) or fdr > 1:
                            raise ValueError("FDR must be a finite fraction, not a percentage")
                    except (ValueError, TypeError, OverflowError) as exc:
                        audit[f"{source['name']}:invalid_rows"] += 1
                        if len(examples) < 20:
                            examples.append({"source": source["name"], "row": original_row, "reason": str(exc)})
                        continue
                    if fdr < 0:
                        audit[f"{source['name']}:negative_fdr_clipped"] += 1
                    fdr = max(0.0, fdr)
                    basename = canonical_run(row[schema["file"]])
                    project = str(row["SearchID"])
                    namespace = str(row.get("dataset_id", ""))
                    run_id = f"{source['name']}::{namespace}::{basename}"
                    group = f"{source['name']}::{project}" if settings.get("group_by", "project") == "project" else run_id
                    uf.find(group)
                    # The same canonical run across searches, exports or sources
                    # must not land in different splits, even if masses shifted.
                    if basename in runs:
                        uf.union(group, runs[basename])
                    else:
                        runs[basename] = group
                    digest = hashlib.blake2b(digest_size=20)
                    digest.update(mz.astype("<f4").tobytes())
                    digest.update(intensity.astype("<f4").tobytes())
                    peak_hash = digest.digest()
                    if peak_hash in peak_groups:
                        uf.union(group, peak_groups[peak_hash])
                    else:
                        peak_groups[peak_hash] = group
                    label_hash = hashlib.blake2b(target.astype(np.uint8).tobytes(), digest_size=16).digest()
                    identity = (run_id, scan, int(charge))
                    previous = identities.get(identity)
                    if previous is not None:
                        uf.union(group, metadata[previous]["original_group"])
                    key = (peak_hash, int(charge), round(precursor, 3))
                    old = fingerprints.get(key)
                    if old is not None:
                        uf.union(group, metadata[old]["original_group"])
                        audit["exact_spectrum_duplicates"] += 1
                        if metadata[old]["label_hash"] != label_hash:
                            audit["duplicate_label_disagreements"] += 1
                            conflicts.add(key)
                            if conflict_example_counts[source["name"]] < 10:
                                conflict_pairs.append((old, len(metadata)))
                                conflict_example_counts[source["name"]] += 1
                    priority = (fdr, -int(source.get("priority", 0)), len(metadata))
                    i = len(metadata)
                    # Mild reliability weighting. A q-value is not a peak PEP;
                    # avoid the old collapse of 1-5% rows to one minimum weight.
                    weight = max(float(settings.get("minimum_weight", 0.5)), 1.0 - fdr)
                    metadata.append({
                        "row_group": row_group, "row_offset": len(store_rows),
                        "source": source["name"], "source_row": original_row,
                        "project": project, "run_id": run_id, "scan": scan,
                        "original_group": group, "num_peaks": len(mz), "num_positive": int(target.sum()),
                        "fdr": fdr, "weight": weight, "spectrum_features": spectrum,
                        "histogram": hist, "fingerprint": key, "label_hash": label_hash,
                        "priority": priority, "identity": identity,
                    })
                    if old is None or priority < metadata[old]["priority"]:
                        fingerprints[key] = i
                    if previous is None or priority < metadata[previous]["priority"]:
                        identities[identity] = i
                    store_rows.append({"mz_arr": mz, "int_arr": intensity,
                                       "annotation_mask": target.astype(bool),
                                       "Charge": int(charge), "exp m/z": precursor})
                    if len(store_rows) >= row_group_size:
                        flush()
                if seen % 10000 < row_group_size:
                    print(f"  {source['name']}: {seen:,} rows processed", flush=True)
                if limit_per_source is not None and seen >= limit_per_source:
                    break
            audit[f"{source['name']}:available_rows"] = parquet.metadata.num_rows
        flush()
    finally:
        writer.close()
    if not metadata:
        raise ValueError("No valid spectra")
    # Keep one identification per run/scan/charge and one exact spectrum globally.
    # Resolve jointly so transitive duplicate chains cannot choose two survivors.
    selected, used_identity, used_fingerprint = [], set(), set()
    for i in sorted(range(len(metadata)), key=lambda j: metadata[j]["priority"]):
        item = metadata[i]
        key, identity = item["fingerprint"], item["identity"]
        if policy == "exclude" and key in conflicts:
            audit["excluded_conflicting_rows"] += 1
            continue
        if key in used_fingerprint or identity in used_identity:
            audit["deduplicated_rows"] += 1
            continue
        selected.append(i)
        used_fingerprint.add(key)
        used_identity.add(identity)
    frame = pd.DataFrame([metadata[i] for i in sorted(selected)])
    frame["group_id"] = frame.original_group.map(uf.find)
    frame["split"], score = assign_groups(frame, settings.get("fractions", [0.7, 0.15, 0.15]), settings.get("seed", 42))
    frame = frame.drop(columns=["fingerprint", "label_hash", "priority", "identity", "original_group"])
    summaries = []
    for name in ("train", "val", "test"):
        part = frame[frame.split == name].reset_index(drop=True)
        part.to_parquet(out / f"{name}.parquet", index=False)
        for source, rows in part.groupby("source"):
            summaries.append({"split": name, "source": source, "spectra": len(rows),
                              "groups": int(rows.group_id.nunique()), "peaks": int(rows.num_peaks.sum()),
                              "positive_fraction": float(rows.num_positive.sum() / rows.num_peaks.sum()),
                              "max_peaks": int(rows.num_peaks.max())})
    summary = pd.DataFrame(summaries)
    summary.to_parquet(out / "split_summary.parquet", index=False)
    write_conflict_examples(out, metadata, conflict_pairs)
    write_json(out / "split_metadata.json", {"feature_config": asdict(features), "sources": sources,
               "split_config": settings, "audit": dict(audit), "invalid_examples": examples,
               "assignment_error": score, "limited_to_rows_per_source": limit_per_source,
               "summary": summaries})
    print(summary.to_string(index=False), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_xl.yml")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit-per-source", type=int, help="Smoke test only; recorded in metadata")
    args = parser.parse_args()
    config, root = read_config(args.config)
    split_data(config, root, args.output_dir, args.limit_per_source)


if __name__ == "__main__":
    main()
