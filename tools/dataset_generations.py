"""Explicit binary-generation history and non-destructive uncommitted retirement.

The original batch_plan.json and completed data/receipts are immutable.
All functions run under the caller's existing .batch.lock. No numerical work.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Callable

FORMAT = "rainbow.dataset_generations.v1"
LAYOUT = "records/i{incident_index:04d}/w{wavelength_index:04d}"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 16 << 20:
        raise ValueError(f"Generation JSON is too large: {path}")
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token: {value}")
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def validate_plan(plan: dict[str, Any], module_names: tuple[str, ...]) -> None:
    if set(plan) != {"version", "conditions", "binary_sha256", "signature", "record_layout"}:
        raise ValueError("Unexpected generation plan fields")
    if plan["version"] != "rainbow.condition_batch.v1" or plan["record_layout"] != LAYOUT:
        raise ValueError("Unsupported generation plan format/layout")
    binary = plan["binary_sha256"]
    if (not isinstance(binary, dict) or set(binary) != {"generator", *module_names}
            or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
                   for v in binary.values())):
        raise ValueError("Malformed generation binary hashes")
    spec = {k: plan[k] for k in ("version", "conditions", "binary_sha256")}
    if digest(spec) != plan["signature"]:
        raise ValueError("Generation plan signature/content mismatch")


def authorize(root: Path, current: dict[str, Any], module_names: tuple[str, ...],
              *, resume: bool, reason: str | None, previous_generator: Path | None
              ) -> dict[str, dict[str, Any]]:
    """Return authorized plans, rejecting any implicit change of binaries.

    Registering requires the preserved previous executable AND all five modules.
    A recorded reason authorizes preservation of old results; it is not proof
    that the new and old numerical outputs are equivalent.
    """
    validate_plan(current, module_names)
    path = root / "batch_plan.json"
    registry_path = root / "generation_history.json"
    if not path.exists():
        if reason is not None or previous_generator is not None:
            raise ValueError("Registering a generation requires an existing dataset plan")
        if any(p.name != ".batch.lock" for p in root.iterdir()):
            raise ValueError("Refusing to adopt a nonempty directory without this batch plan")
        write_json(path, current)
        return {current["signature"]: current}
    if not resume:
        raise FileExistsError("Dataset plan exists; use --resume")
    base = read_json(path)
    validate_plan(base, module_names)
    for key in ("version", "conditions", "record_layout"):
        if current[key] != base[key]:
            raise ValueError("Resume rejected: scientific conditions/layout differ from batch_plan.json")
    base_sig = base["signature"]
    if registry_path.exists():
        registry = read_json(registry_path)
        if (registry.get("format") != FORMAT or registry.get("base_signature") != base_sig
                or not isinstance(registry.get("generations"), list)):
            raise ValueError("Invalid generation history identity")
        plans: dict[str, dict[str, Any]] = {}
        parent = None
        for entry in registry["generations"]:
            p = entry.get("plan", {})
            validate_plan(p, module_names)
            sig = p["signature"]
            if (sig in plans or entry.get("parent_signature") != parent
                    or not isinstance(entry.get("reason"), str) or not entry["reason"].strip()):
                raise ValueError("Broken generation history chain")
            for key in ("version", "conditions", "record_layout"):
                if p[key] != base[key]:
                    raise ValueError("Generation history contains incompatible conditions")
            if not plans and p != base:
                raise ValueError("Generation history root differs from original batch plan")
            plans[sig] = p
            parent = sig
        if not plans or registry.get("active_signature") != parent:
            raise ValueError("Invalid active generation")
        active = parent
    else:
        plans = {base_sig: base}
        active = base_sig
        registry = {"format": FORMAT, "base_signature": base_sig,
                    "active_signature": base_sig, "generations": [
                        {"plan": base, "parent_signature": None,
                         "reason": "original batch_plan.json; kept unchanged",
                         "registered_time_ns": time.time_ns()}]}
    sig = current["signature"]
    if sig == active:
        if current != plans[active]:
            raise ValueError("Active generation plan mismatch")
        if reason is not None:
            print("[generation] current binaries already registered; no history rewrite.", flush=True)
        return plans
    if sig in plans:
        raise ValueError("Refusing implicit rollback to an older generation")
    if reason is None or not reason.strip() or previous_generator is None:
        raise ValueError("Resume rejected: binaries changed. After validation, use "
                         "--register-generation REASON --previous-generator PATH_TO_OLD_EXE")
    if len(reason) > 2048:
        raise ValueError("Generation reason is too long")
    previous_generator = previous_generator.resolve(strict=True)
    previous = {"generator": previous_generator,
                **{name: previous_generator.parent / "modules" / name for name in module_names}}
    previous_hashes = {name: file_hash(p) for name, p in previous.items()}
    if previous_hashes != plans[active]["binary_sha256"]:
        raise ValueError("Preserved previous binaries do not match the previous active generation")
    registry["generations"].append({"plan": current, "parent_signature": active,
        "reason": reason.strip(), "registered_time_ns": time.time_ns(),
        "previous_generator_backup": str(previous_generator),
        "numerical_equivalence_certified": False})
    registry["active_signature"] = sig
    write_json(registry_path, registry)
    plans[sig] = current
    print(f"[generation] registered {sig[:12]} after {active[:12]}; "
          "old plan/data/receipts preserved.", flush=True)
    return plans


def latest_signature(failure: dict[str, Any]) -> str:
    return failure["attempts"][-1].get("generation_signature", failure["signature"])


def _inventory(path: Path) -> dict[str, dict[str, Any]]:
    """Hash a flat unfinished record; never follow links/reparse points."""
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise ValueError(f"Unsafe uncommitted record directory: {path}")
    result = {}
    for child in path.iterdir():
        info = child.lstat()
        if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"Uncommitted record contains a non-regular file: {child}")
        result[child.name] = {"size": info.st_size, "sha256": file_hash(child)}
    return result


def retire_uncommitted(root: Path, ii: int, ll: int, known_signatures: set[str],
                       rename: Callable[[Path, Path], Any], *, enabled: bool) -> bool:
    """Move unfinished/orphan output aside, never delete or adopt it.

    The explicit option acknowledges that the old writer has stopped. A durable
    journal makes interrupted moves replayable. Unreceipted output has unknown
    provenance; it is regenerated, even if its values happen to be well formed.
    """
    job = f"i{ii:04d}_w{ll:04d}"
    record = root / f"records/i{ii:04d}/w{ll:04d}"
    receipt = root / "receipts" / (job + ".json")
    source_part = Path(str(record) + ".part")
    archive_root = root / "retired_outputs" / job
    prepared = []
    if archive_root.exists():
        if archive_root.is_symlink() or getattr(archive_root.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe retirement archive")
        for candidate in sorted(archive_root.glob("*/retirement.json")):
            entry = read_json(candidate)
            if entry.get("state") == "prepared":
                prepared.append((candidate, entry))
    if len(prepared) > 1:
        raise ValueError(f"Multiple unfinished retirement journals at {job}")
    if prepared:
        journal_path, journal = prepared[0]
        archive = journal_path.parent
        if (journal.get("format") != "rainbow.retired_output.v1" or journal.get("job") != job
                or not isinstance(journal.get("moves"), list)
                or not set(journal.get("known_generation_signatures", [])) <= known_signatures):
            raise ValueError("Invalid retired-output journal")
    else:
        sources = [p for p in (source_part, record) if os.path.lexists(p)]
        if not sources or receipt.exists():
            return False
        if not enabled:
            raise ValueError(f"Uncommitted output at {job}; preserve it. After all writers stop, "
                             "--retire-uncommitted moves it aside for regeneration")
        archive = archive_root / str(time.time_ns())
        journal_path = archive / "retirement.json"
        if archive.exists():
            raise ValueError(f"Retirement archive already exists: {archive}")
        moves = []
        for source in sources:
            moves.append({"source": source.relative_to(root).as_posix(),
                          "destination": (archive / source.name).relative_to(root).as_posix(),
                          "files": _inventory(source)})
        journal = {"format": "rainbow.retired_output.v1", "job": job, "state": "prepared",
                   "created_time_ns": time.time_ns(), "known_generation_signatures": sorted(known_signatures),
                   "provenance": "unreceipted_output_not_accepted", "moves": moves}
        write_json(journal_path, journal)
    if not enabled:
        raise ValueError(f"Unfinished retirement at {job}; --retire-uncommitted required")
    if receipt.exists():
        raise ValueError("Refusing to retire any receipted output")
    allowed_sources = {record.relative_to(root).as_posix(), source_part.relative_to(root).as_posix()}
    seen = set()
    for move in journal["moves"]:
        source_name = move.get("source")
        if source_name not in allowed_sources or source_name in seen:
            raise ValueError("Retirement source does not match the job")
        seen.add(source_name)
        source = root / source_name
        expected_dest = archive / source.name
        if move.get("destination") != expected_dest.relative_to(root).as_posix():
            raise ValueError("Retirement destination does not match the job")
        dest = expected_dest
        a, b = os.path.lexists(source), os.path.lexists(dest)
        if a == b:
            raise ValueError("Retirement requires exactly one source or archived copy")
        if _inventory(source if a else dest) != move["files"]:
            raise ValueError("Uncommitted content changed since retirement began")
        if a:
            rename(source, dest)
        if _inventory(dest) != move["files"]:
            raise ValueError("Retirement content changed during rename")
    journal["state"] = "retired"
    journal["retired_time_ns"] = time.time_ns()
    write_json(journal_path, journal)
    print(f"[retire] preserved {job} under {archive}; condition will be regenerated.", flush=True)
    return True
