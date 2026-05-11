#!/usr/bin/env python3
"""Convert Alembic curve hair to the renderer's binary strand cache format."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import struct
import sys
from pathlib import Path


BINARY_MAGIC = b"SSRSTRD\0"
BINARY_VERSION = 1


def safe_file_stem(name: str) -> str:
    stem = name.strip().replace(" ", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)


def load_alembic_modules():
    try:
        from alembic.Abc import IArchive, WrapExistingFlag
        from alembic.AbcGeom import ICurves, IXform
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing VFX Alembic Python bindings. The PyPI package named "
            "'alembic' is a database migration library; this converter needs "
            "bindings that provide alembic.Abc and alembic.AbcGeom."
        ) from exc

    return IArchive, WrapExistingFlag, ICurves, IXform


def child_count(obj) -> int:
    return int(obj.getNumChildren())


def children(obj):
    for index in range(child_count(obj)):
        yield obj.getChild(index)


def point3(value) -> list[float]:
    try:
        return [float(value.x), float(value.y), float(value.z)]
    except AttributeError:
        return [float(value[0]), float(value[1]), float(value[2])]


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_rows(matrix) -> list[list[float]]:
    try:
        return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]
    except Exception:
        return identity_matrix()


def mul_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    x, y, z = point
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def xform_matrix(obj, wrap_flag, ixform) -> list[list[float]]:
    if not ixform.matches(obj.getMetaData()):
        return identity_matrix()
    try:
        schema = ixform(obj, wrap_flag).getSchema()
        return matrix_to_rows(schema.getValue().getMatrix())
    except Exception:
        return identity_matrix()


def collect_curve_objects(archive_path: Path):
    iarchive, wrap_flag, icurves, ixform = load_alembic_modules()
    archive = iarchive(str(archive_path))
    found = []

    def walk(obj, path, parent_matrix):
        local_matrix = xform_matrix(obj, wrap_flag.kWrapExisting, ixform)
        world_matrix = mul_matrix(parent_matrix, local_matrix)
        if icurves.matches(obj.getMetaData()):
            found.append((path, obj, world_matrix))
        for child in children(obj):
            child_name = child.getName()
            child_path = f"{path}/{child_name}" if path else f"/{child_name}"
            walk(child, child_path, world_matrix)

    walk(archive.getTop(), "", identity_matrix())
    return found, wrap_flag, icurves


def extract_curves(obj, world_matrix, wrap_existing, icurves):
    schema = icurves(obj, wrap_existing.kWrapExisting).getSchema()
    sample = schema.getValue()
    positions = [transform_point(world_matrix, point3(point)) for point in sample.getPositions()]
    curve_counts = [int(count) for count in sample.getCurvesNumVertices()]

    offset = 0
    strands = []
    for count in curve_counts:
        if count >= 2:
            strands.append(list(range(offset, offset + count)))
        offset += count

    if offset != len(positions):
        raise ValueError(
            f"curve vertex counts sum to {offset}, but positions contain {len(positions)} points"
        )

    return positions, strands


def bounds_from_flat(vertices):
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for offset in range(0, len(vertices), 3):
        for axis, value in enumerate(vertices[offset : offset + 3]):
            mins[axis] = min(mins[axis], value)
            maxs[axis] = max(maxs[axis], value)
    return mins, maxs


def write_binary_cache(output: Path, scale: float, vertices, strand_meta, indices):
    with output.open("wb") as file:
        file.write(BINARY_MAGIC)
        file.write(struct.pack("<IfQQQ", BINARY_VERSION, scale, len(vertices) // 3, len(strand_meta) // 2, len(indices)))

        vertex_payload = array("f", vertices)
        meta_payload = array("I", strand_meta)
        index_payload = array("I", indices)
        if sys.byteorder != "little":
            vertex_payload.byteswap()
            meta_payload.byteswap()
            index_payload.byteswap()

        vertex_payload.tofile(file)
        meta_payload.tofile(file)
        index_payload.tofile(file)


def write_json_cache(output: Path, source: Path, scale: float, vertices, strand_meta, indices):
    points = [
        [vertices[offset], vertices[offset + 1], vertices[offset + 2]]
        for offset in range(0, len(vertices), 3)
    ]
    strands = [
        indices[offset : offset + count]
        for count, offset in zip(strand_meta[0::2], strand_meta[1::2])
    ]
    cache = {
        "version": BINARY_VERSION,
        "source": str(source),
        "scale": scale,
        "vertices": points,
        "strands": strands,
    }
    output.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")


def build_cache(curve_items, wrap_existing, icurves, max_strands: int | None):
    vertices = []
    strand_meta = []
    indices = []
    strand_count = 0
    for _path, obj, matrix in curve_items:
        curve_vertices, curve_strands = extract_curves(obj, matrix, wrap_existing, icurves)
        for strand in curve_strands:
            if max_strands is not None and strand_count >= max_strands:
                break
            base = len(vertices)
            for index in strand:
                vertices.extend(curve_vertices[index])
            index_offset = len(indices)
            vertex_base = base // 3
            indices.extend(range(vertex_base, vertex_base + len(strand)))
            strand_meta.extend((len(strand), index_offset))
            strand_count += 1
        if max_strands is not None and strand_count >= max_strands:
            break

    if strand_count == 0:
        raise ValueError("No valid strands with at least two points were found.")

    return vertices, strand_meta, indices, strand_count


def write_cache(output: Path, fmt: str, source: Path, scale: float, vertices, strand_meta, indices):
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "binary":
        write_binary_cache(output, scale, vertices, strand_meta, indices)
    else:
        write_json_cache(output, source, scale, vertices, strand_meta, indices)


def print_cache_summary(output: Path, strand_count: int, vertices, scale: float):
    mins, maxs = bounds_from_flat(vertices)
    print(
        f"wrote {output} with {strand_count} strands, {len(vertices) // 3} vertices, "
        f"bounds min={mins} max={maxs}, loader scale={scale}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--list", action="store_true", help="List curve objects and exit")
    parser.add_argument("--curve", action="append", default=[], help="Only import paths containing this substring")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied by the Bevy cache loader")
    parser.add_argument("--max-strands", type=int, default=None, help="Import only the first N strands")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Write one cache file per Alembic curve object. Output is treated as a directory.",
    )
    parser.add_argument(
        "--format",
        choices=("binary", "json"),
        default="binary",
        help="Output cache format",
    )
    args = parser.parse_args()

    curve_objects, wrap_existing, icurves = collect_curve_objects(args.input)
    if args.list:
        for path, obj, _matrix in curve_objects:
            schema = icurves(obj, wrap_existing.kWrapExisting).getSchema()
            sample = schema.getValue()
            count = len(sample.getCurvesNumVertices())
            points = len(sample.getPositions())
            print(f"{path}: curves={count} points={points}")
        return 0

    selected = [
        item
        for item in curve_objects
        if not args.curve or any(needle in item[0] for needle in args.curve)
    ]
    if not selected:
        raise SystemExit("No Alembic curve objects matched the requested filters.")

    suffix = ".strands" if args.format == "binary" else ".strands.json"
    if args.split:
        output_dir = args.output or args.input.with_suffix(".strands.d")
        output_dir.mkdir(parents=True, exist_ok=True)
        total_strands = 0
        total_vertices = 0
        for path, obj, matrix in selected:
            vertices, strand_meta, indices, strand_count = build_cache(
                [(path, obj, matrix)],
                wrap_existing,
                icurves,
                args.max_strands,
            )
            output = output_dir / f"{safe_file_stem(path.rsplit('/', 1)[-1])}{suffix}"
            write_cache(output, args.format, args.input, args.scale, vertices, strand_meta, indices)
            print_cache_summary(output, strand_count, vertices, args.scale)
            total_strands += strand_count
            total_vertices += len(vertices) // 3
        print(
            f"wrote {len(selected)} split caches under {output_dir} "
            f"with {total_strands} total strands and {total_vertices} total vertices"
        )
        return 0

    vertices, strand_meta, indices, strand_count = build_cache(
        selected,
        wrap_existing,
        icurves,
        args.max_strands,
    )
    output = args.output or args.input.with_suffix(suffix)
    write_cache(output, args.format, args.input, args.scale, vertices, strand_meta, indices)
    print_cache_summary(output, strand_count, vertices, args.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
