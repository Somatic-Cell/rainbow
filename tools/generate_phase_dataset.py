#!/usr/bin/env python3
"""Sequential, resumable condition-grid generation; no point sampling or learning.

Standard library only. Every condition invokes the real rainbow_generate once.
This orchestrator does not batch OptiX queries or share GPU contexts across jobs.
It never changes resolution/tolerances to get a failed condition to pass.
Opt-in --continue-on-error defers recognized pre-CDF input rejections ONLY.
No rejected distribution is saved or counted as complete. Other failures stop.
Opt-in --recover-commits verifies completed .part data and retries final rename.
It does not recompute, repair, or modify numerical data or the batch signature.
Binary changes require explicit registration; old plan/data provenance is retained.
--retry-failed-and-pending selects missing conditions without replacing successes.
--retire-uncommitted preserves interrupted output outside records before retrying.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
import stat
import re
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
VERSION = "rainbow.condition_batch.v1"
RUNNER_VERSION = "rainbow.condition_batch.generations.v1"
# Also importable by unittest via spec_from_file_location, outside tools/.
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
import dataset_generations as generations
FAILURE_VERSION = "rainbow.condition_failure.v1"
GENERATOR_VERSION = "numpy_cdf_water_geometry_v1"
MODULES = ("raindrop_trace.optixir", "patch_build.fatbin", "patch_query.optixir",
           "patch_optics.fatbin", "phase_cdf.fatbin")
ARRAYS = ("phi_cdf.npy", "theta_given_phi_cdf.npy", "u_edges.npy")


def fail_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def load_json(path: Path, limit: int = 8 << 20) -> dict[str, Any]:
    if path.stat().st_size > limit:
        raise ValueError(f"JSON too large: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream, parse_constant=fail_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def exact(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numerical parameter")
    try:
        d = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal: {value}") from exc
    if not d.is_finite():
        raise ValueError("Nonfinite parameter")
    return d


def scalar(value: Any, low: float, high: float) -> float:
    d = exact(value)
    if not Decimal(str(low)) <= d <= Decimal(str(high)):
        raise ValueError(f"Parameter {value} is outside [{low}, {high}]")
    return float(d)


def integer(value: Any, low: int, high: int) -> int:
    d = exact(value)
    if d != d.to_integral_value() or not low <= d <= high:
        raise ValueError(f"Expected integer in [{low}, {high}]")
    return int(d)


def flag(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("Expected JSON true or false")
    return value


def decimal_string(d: Decimal) -> str:
    return format(d, "f")


def wavelengths(spec: dict[str, Any]) -> list[str]:
    if set(spec) != {"start_nm", "stop_nm", "step_nm"}:
        raise ValueError("wavelengths needs start_nm, stop_nm, step_nm")
    start, stop, step = (exact(spec[k]) for k in ("start_nm", "stop_nm", "step_nm"))
    if not 380 <= start <= stop <= 830 or step <= 0:
        raise ValueError("Require 380 <= start <= stop <= 830 and positive step")
    intervals = (stop-start)/step
    if intervals != intervals.to_integral_value():
        raise ValueError("The wavelength range must contain an integer number of steps; no silent endpoint truncation")
    count = int(intervals)+1
    if count > 100000:
        raise ValueError("Too many wavelengths")
    values = [decimal_string(start+step*i) for i in range(count)]
    packed = [struct.pack("<f", float(v)) for v in values]
    if len(set(packed)) != len(packed):
        raise ValueError("Distinct wavelengths collapse to the same FP32 solver value")
    return values


def inclinations(spec: dict[str, Any]) -> list[str]:
    if set(spec) != {"min_degrees", "max_degrees", "count", "placement"}:
        raise ValueError("inclinations needs min_degrees, max_degrees, count, placement")
    if spec["placement"] != "linear_angle_cell_centers":
        raise ValueError("Only explicit linear_angle_cell_centers is supported")
    low, high = exact(spec["min_degrees"]), exact(spec["max_degrees"])
    count = integer(spec["count"], 1, 100000)
    if not -90 <= low < high <= 90:
        raise ValueError("Canonical inclination domain is [-90,90], NOT [0,180]")
    return [decimal_string(low+(high-low)*(Decimal(i)+Decimal("0.5"))/count)
            for i in range(count)]


def normalized_config(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {"output_root", "generator", "radius_mm", "force_sphere", "temperature_c",
               "pressure_pa", "wavelengths", "inclinations", "solver", "reserve_gib"}
    if set(raw) != allowed:
        raise ValueError(f"Unexpected/missing config keys: {set(raw)^allowed}")
    s = raw["solver"]
    sk = {"grid", "query_theta", "query_phi", "cdf_theta", "cdf_phi", "stage", "device",
          "focal_offsets", "allow_underresolved", "maximum_coarsening_tv",
          "cdf_max_l1", "cdf_max_lost_mass"}
    if not isinstance(s, dict) or set(s) != sk:
        raise ValueError("Invalid solver keys")
    out = {k: integer(s[k], 2 if k == "grid" else 1, 32767)
           for k in ("grid", "query_theta", "query_phi", "cdf_theta", "cdf_phi")}
    if out["grid"]**2*4 > 0xFFFFFFFF:
        raise ValueError("Source index range exceeds the existing solver contract")
    for axis in ("theta", "phi"):
        if out["query_"+axis] % out["cdf_"+axis] or out["cdf_"+axis] > out["query_"+axis]:
            raise ValueError("CDF grids must divide query grids")
    if s["stage"] not in ("diffraction", "focal", "path", "incoherent"):
        raise ValueError("Unknown stage")
    out["stage"] = s["stage"]
    out["device"] = integer(s["device"], 0, 128)
    out["allow_underresolved"] = flag(s["allow_underresolved"])
    out["maximum_coarsening_tv"] = scalar(s["maximum_coarsening_tv"], 0, 1)
    out["cdf_max_l1"] = scalar(s["cdf_max_l1"], 0, 1)
    out["cdf_max_lost_mass"] = scalar(s["cdf_max_lost_mass"], 0, 1)
    offsets = s["focal_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 4:
        raise ValueError("Four focal offsets required")
    out["focal_offsets"] = [integer(v, 0, 3) for v in offsets]
    radius = scalar(raw["radius_mm"], 1e-9, 3)
    if s["stage"] in ("focal", "diffraction") and not .1 <= radius <= 1:
        raise ValueError("Table-II diffraction path currently accepts radii 0.1..1.0 mm only")
    return {"radius_mm": radius, "force_sphere": flag(raw["force_sphere"]),
            "temperature_c": scalar(raw["temperature_c"], 0, 60),
            "pressure_pa": scalar(raw["pressure_pa"], 50000, 100000000),
            "wavelength_nm": wavelengths(raw["wavelengths"]),
            "inclination_degrees": inclinations(raw["inclinations"]),
            "inclination_placement": "linear_angle_cell_centers",
            "material": "water", "solver": out}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


@contextmanager
def batch_lock(root: Path) -> Iterator[None]:
    lock = root / ".batch.lock"
    try:
        with lock.open("x", encoding="ascii") as f:
            import os
            f.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError("Batch lock exists. Do not run two writers. After verifying that no batch is running, remove the stale .batch.lock manually.") from exc
    try:
        yield
    finally:
        lock.unlink()


def npy_size(path: Path, expected_shape: tuple[int, ...]) -> int:
    """Header and length validation only; no full CDF read, no sampling."""
    with path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not NPY: {path}")
        ver = f.read(2)
        if ver == b"\x01\x00":
            raw = f.read(2)
            if len(raw) != 2: raise ValueError("Truncated NPY")
            length = struct.unpack("<H", raw)[0]
        elif ver == b"\x02\x00":
            raw = f.read(4)
            if len(raw) != 4: raise ValueError("Truncated NPY")
            length = struct.unpack("<I", raw)[0]
        else:
            raise ValueError("Unsupported NPY version")
        if not 0 < length <= 65536: raise ValueError("Invalid NPY header size")
        text = f.read(length)
        if len(text) != length: raise ValueError("Truncated NPY header")
        h = ast.literal_eval(text.decode("latin1"))
        if (h.get("descr") != "<f8" or h.get("fortran_order") is not False
                or h.get("shape") != expected_shape):
            raise ValueError(f"Wrong NPY layout: {path}")
        expected = f.tell()+8*math.prod(expected_shape)
    if path.stat().st_size != expected:
        raise ValueError(f"Truncated or trailing NPY data: {path}")
    return expected


def close(a: Any, b: float, tol: float = 1e-12) -> bool:
    return (type(a) in (int, float) and math.isfinite(a)
            and abs(a-b) <= tol*max(1.0, abs(b)))


def fp32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def verify_record(path: Path, plan: dict[str, Any], ii: int, ll: int) -> dict[str, Any]:
    c, s = plan["conditions"], plan["conditions"]["solver"]
    m = load_json(path/"metadata.json", 1 << 20)
    def require(ok: bool, message: str) -> None:
        if not ok: raise ValueError(f"{path}: {message}")
    expected = {"complete": True, "schema": "rainbow.phase_cdf.numpy.v2",
                "generator_version": GENERATOR_VERSION,
                "coordinate_contract_id": "rainbow.phase_cdf.coordinates.v1",
                "direction_convention": "physical_propagation", "dtype": "<f8", "order": "C",
                "density_measure": "solid_angle_sr", "cdf_axis_order": ["phi_cell", "theta_edge"],
                "theta_count": s["cdf_theta"], "phi_count": s["cdf_phi"],
                "query_grid": [s["query_theta"], s["query_phi"]],
                "incident_grid": [s["grid"], s["grid"]], "stage": s["stage"],
                "force_sphere": c["force_sphere"], "focal_quarter_turn_offsets": s["focal_offsets"]}
    for key, value in expected.items(): require(m.get(key) == value, f"mismatched {key}")
    require(close(m.get("radius_mm"), fp32(c["radius_mm"])), "radius")
    require(close(m.get("wavelength_nm"), fp32(float(c["wavelength_nm"][ll]))), "wavelength")
    alpha = float(c["inclination_degrees"][ii])
    require(close(m.get("incident_inclination_degrees"), alpha), "inclination")
    material = m.get("material", {})
    require(material.get("model") == "water_iapws_r9_97", "material")
    require(close(material.get("temperature_kelvin"), c["temperature_c"]+273.15), "temperature")
    require(close(material.get("pressure_pascal"), c["pressure_pa"]), "pressure")
    mat = m.get("sampling_frame_columns", [])
    require(isinstance(mat, list) and len(mat)==3 and all(isinstance(row,list) and len(row)==3 for row in mat), "frame shape")
    require(all(type(v) in (int,float) and math.isfinite(v) for row in mat for v in row), "frame values")
    cols = [[mat[r][j] for r in range(3)] for j in range(3)]
    for j in range(3):
        for k in range(3):
            require(close(sum(cols[j][r]*cols[k][r] for r in range(3)), float(j==k), 2e-12), "frame orthogonality")
    cross = [cols[0][1]*cols[1][2]-cols[0][2]*cols[1][1],
             cols[0][2]*cols[1][0]-cols[0][0]*cols[1][2],
             cols[0][0]*cols[1][1]-cols[0][1]*cols[1][0]]
    require(sum(cross[r]*cols[2][r] for r in range(3)) > 1-2e-12, "frame handedness")
    angle = math.radians(alpha)
    ki = [math.cos(angle), -math.sin(angle), 0.0]
    require(all(abs(cols[2][r]-ki[r]) < 2e-6 for r in range(3)), "incident direction/sign")
    geometry, cs = m.get("geometry", {}), m.get("cross_sections", {})
    area = geometry.get("projected_area_mm2")
    require(type(area) in (int,float) and math.isfinite(area) and area>0, "projected area")
    require(geometry.get("projected_area_converged") is True and geometry.get("convexity_verified") is True, "area validation")
    require(cs.get("model")=="geometric_nonabsorbing" and cs.get("wave_scattering_cross_section_computed") is False, "cross-section model")
    require(cs.get("scattering_mm2")==area and cs.get("extinction_mm2")==area and cs.get("absorption_mm2")==0, "cross sections")
    q = m.get("quality", {})
    require(q.get("invalid_values")==0 and q.get("incomplete_directions")==0, "invalid optical data")
    require(q.get("allow_underresolved") is s["allow_underresolved"], "underresolution policy")
    require(s["allow_underresolved"] or q.get("underresolved_directions")==0, "underresolution not authorized")
    for key, cfg in (("cdf_l1_mass_error","cdf_max_l1"),("cdf_lost_probability_mass","cdf_max_lost_mass")):
        value = q.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=s[cfg], key)
    sp = m.get("storage_processing", {})
    require(sp.get("additional_filter")=="none", "unexpected global smoothing")
    tv = sp.get("coarsening_tv")
    require(type(tv) in (int,float) and math.isfinite(tv) and 0<=tv<=s["maximum_coarsening_tv"], "coarsening error")
    hg = m.get("hg", {})
    require(hg.get("target")=="saved_cdf" and type(hg.get("g")) in (float,int) and -1<hg["g"]<1, "HG target")
    nt, np_ = s["cdf_theta"], s["cdf_phi"]
    sizes = {name: npy_size(path/name, shape) for name,shape in
             zip(ARRAYS, ((np_+1,), (np_,nt+1), (nt+1,)))}
    return {"metadata_sha256": sha256(path/"metadata.json"), "npy_sizes": sizes,
            "hg_g": hg["g"], "projected_area_mm2": area,
            "underresolved_directions": q.get("underresolved_directions"), "coarsening_tv": tv}


def command(exe: Path, out: Path, c: dict[str, Any], ii: int, ll: int) -> list[str]:
    s = c["solver"]
    values = {"out": str(out), "material": "water", "radius-mm": str(c["radius_mm"]),
              "wavelength-nm": c["wavelength_nm"][ll], "inclination-deg": c["inclination_degrees"][ii],
              "temperature-c": str(c["temperature_c"]), "pressure-pa": str(c["pressure_pa"]),
              "grid": s["grid"], "query-theta": s["query_theta"], "query-phi": s["query_phi"],
              "cdf-theta": s["cdf_theta"], "cdf-phi": s["cdf_phi"], "stage": s["stage"],
              "device": s["device"], "max-coarsening-tv": s["maximum_coarsening_tv"],
              "cdf-max-l1": s["cdf_max_l1"], "cdf-max-lost-mass": s["cdf_max_lost_mass"],
              "focal-offsets": ",".join(map(str,s["focal_offsets"]))}
    args = [str(exe)]
    for k,v in values.items(): args += ["--"+k,str(v)]
    if c["force_sphere"]: args.append("--sphere")
    if s["allow_underresolved"]: args.append("--allow-underresolved")
    return args



# This is recovery of a completed on-disk transaction, NOT a numerical retry.
COMMIT_RECOVERY_VERSION = "rainbow.commit_recovery.v1"
COMMIT_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)


def log_tail_lines(log: Path) -> list[str]:
    with log.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - (256 << 10)))
        return stream.read().decode("utf-8", errors="replace").splitlines()


def rename_failure(log: Path, returncode: int) -> dict[str, str] | None:
    """Recognize the two-path std::filesystem::rename error, not arbitrary I/O.

    File names in this application have no quotes. Both paths must subsequently
    match the job derived from the verified batch plan; logs never choose paths.
    The OS message is retained, not interpreted as proof of a temporary lock.
    """
    if returncode != 1:
        return None
    errors = [line for line in log_tail_lines(log) if line.startswith("rainbow_generate:")]
    if not errors:
        return None
    match = re.fullmatch(r'rainbow_generate: rename: (.+): "([^"\r\n]+)", "([^"\r\n]+)"', errors[-1])
    if not match:
        return None
    return {"message": errors[-1], "os_message": match[1],
            "source": match[2], "destination": match[3]}


def log_path_key(value: str) -> str:
    # std::filesystem implementations may quote Windows backslashes twice.
    # Never use the parsed path for filesystem access; this is comparison only.
    import ntpath
    if re.match(r"^[A-Za-z]:", value) or value.startswith("\\\\"):
        value = value.replace("\\\\", "\\") if re.match(r"^[A-Za-z]:", value) else value
        return ntpath.normcase(ntpath.normpath(value))
    return os.path.normcase(os.path.abspath(value))


def is_plain_path(path: Path, *, directory: bool) -> None:
    """Recovery refuses symbolic links/junctions and unexpected object types."""
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400):  # FILE_ATTRIBUTE_REPARSE_POINT
        raise ValueError(f"Recovery refuses a linked/reparse path: {path}")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError(f"Wrong recovery object type: {path}")


def recovery_log(root: Path, failure: dict[str, Any]) -> Path:
    # A per-job failure ledger is already signature/index-checked by the caller.
    from pathlib import PurePosixPath
    name = PurePosixPath(failure["attempts"][-1]["log"])
    if name.is_absolute() or len(name.parts) != 2 or name.parts[0] != "logs" or "\\" in str(name):
        raise ValueError("Recovery log must be a direct file under logs/")
    log = root.joinpath(*name.parts)
    is_plain_path(log.parent, directory=True)
    is_plain_path(log, directory=False)
    return log


def record_hashes(path: Path) -> dict[str, str]:
    is_plain_path(path.parent.parent, directory=True)
    is_plain_path(path.parent, directory=True)
    is_plain_path(path, directory=True)
    names = (*ARRAYS, "metadata.json")
    if {entry.name for entry in path.iterdir()} != set(names):
        raise ValueError(f"Recovery requires exactly three NPY files and metadata.json: {path}")
    for name in names:
        is_plain_path(path/name, directory=False)
    return {name: sha256(path/name) for name in names}


def read_f64_rows(path: Path, rows: int, width: int) -> Iterator[list[float]]:
    """Read closed-at-exit, row-sized blocks; header/size is checked beforehand.

    No mmap: an outstanding mmap/file handle can itself prevent a Windows rename.
    The fixed-endian data are decoded without NumPy or a PDF/point buffer.
    """
    from array import array
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if prefix[:6] != b"\x93NUMPY" or prefix[6:] not in (b"\x01\x00", b"\x02\x00"):
            raise ValueError(f"Invalid NPY prefix: {path}")
        count_bytes = 2 if prefix[6:] == b"\x01\x00" else 4
        raw = stream.read(count_bytes)
        if len(raw) != count_bytes:
            raise ValueError(f"Truncated NPY header: {path}")
        header_bytes = int.from_bytes(raw, "little")
        if not 0 < header_bytes <= 65536 or len(stream.read(header_bytes)) != header_bytes:
            raise ValueError(f"Invalid NPY header length: {path}")
        for _ in range(rows):
            raw = stream.read(8*width)
            if len(raw) != 8*width:
                raise ValueError(f"Truncated NPY payload: {path}")
            data = array("d")
            if data.itemsize != 8:
                raise RuntimeError("Recovery requires an 8-byte Python array('d')")
            data.frombytes(raw)
            if sys.byteorder != "little":
                data.byteswap()
            yield data.tolist()
        if stream.read(1):
            raise ValueError(f"Trailing NPY payload: {path}")


def audit_saved_cdf(path: Path, plan: dict[str, Any]) -> dict[str, float]:
    """Full-value validation of the saved distribution, NOT optical convergence.

    Uses the saved edge coordinates. Does not normalize, clamp, repair, or sample.
    Tolerances match the existing reader: mass 1e-10; saved-CDF HG g 5e-12.
    """
    from contextlib import closing
    s = plan["conditions"]["solver"]
    nt, np_ = s["cdf_theta"], s["cdf_phi"]

    def vector(name: str, width: int) -> list[float]:
        with closing(read_f64_rows(path/name, 1, width)) as it:
            data = next(it)
            # Exhaust the iterator: verify the final payload length, then close.
            if next(it, None) is not None:
                raise ValueError("Unexpected additional row")
            return data

    def cdf_row(data: list[float], label: str) -> list[float]:
        if (data[0] != 0.0 or data[-1] != 1.0
                or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in data)):
            raise ValueError(f"Recovery: invalid CDF endpoints/values in {label}")
        delta = [b-a for a, b in zip(data, data[1:])]
        if any(v < 0.0 for v in delta):
            raise ValueError(f"Recovery: decreasing CDF in {label}")
        return delta

    u = vector("u_edges.npy", nt+1)
    du = cdf_row(u, "u_edges.npy")
    if any(v <= 0.0 for v in du):
        raise ValueError("Recovery: u_edges must be strictly increasing")
    q = cdf_row(vector("phi_cdf.npy", np_+1), "phi_cdf.npy")
    mu = [1.0-a-b for a, b in zip(u, u[1:])]
    masses, moments = [], []
    with closing(read_f64_rows(path/"theta_given_phi_cdf.npy", np_, nt+1)) as rows:
        for j, row in enumerate(rows):
            delta = cdf_row(row, f"theta_given_phi_cdf.npy row {j}")
            masses.append(q[j]*math.fsum(delta))
            moments.append(q[j]*math.fsum(w*x for w, x in zip(delta, mu)))
    mass = math.fsum(masses)
    if not math.isfinite(mass) or abs(mass-1.0) > 1e-10:
        raise ValueError("Recovery: saved CDF mass differs from 1")
    g = math.fsum(moments)/mass
    expected_g = load_json(path/"metadata.json", 1 << 20)["hg"]["g"]
    if not math.isfinite(g) or abs(g-expected_g) > 5e-12:
        raise ValueError("Recovery: metadata HG g disagrees with saved CDF")
    return {"mass": mass, "g": g, "mass_tolerance": 1e-10, "g_tolerance": 5e-12}


def rename_with_retry(source: Path, destination: Path) -> dict[str, Any]:
    """Bounded permission/share-lock retries; never delete or replace output.

    Operates under the batch's exclusive-writer lock. All validation handles must
    have been closed by the caller. Windows Path.rename refuses an existing target.
    On other systems the target check also assumes the documented single writer.
    """
    import errno
    attempts, waited, errors = 0, 0.0, []
    while True:
        if os.path.lexists(destination):
            raise FileExistsError(f"Recovery refuses an existing destination: {destination}")
        is_plain_path(source, directory=True)
        attempts += 1
        try:
            source.rename(destination)
            return {"rename_attempts": attempts, "waited_seconds": waited, "rename_errors": errors}
        except OSError as exc:
            code = getattr(exc, "winerror", None)
            retryable = (code in (5, 32, 33) if code is not None
                         else exc.errno in (errno.EACCES, errno.EPERM, errno.EBUSY))
            if not retryable or attempts > len(COMMIT_RETRY_DELAYS):
                raise
            delay = COMMIT_RETRY_DELAYS[attempts-1]
            errors.append({"attempt": attempts, "errno": exc.errno, "winerror": code, "message": str(exc)})
            print(f"[commit-recovery] rename attempt {attempts} failed; retry in {delay:g}s: {exc}", flush=True)
            time.sleep(delay)
            waited += delay


def recover_commit(root: Path, plan: dict[str, Any], ii: int, ll: int,
                   path: Path, receipt: Path, failure: dict[str, Any] | None) -> bool:
    """Adopt only a verified final-rename failure tied to this signed batch plan.

    A durable hash journal makes a crash between rename and receipt recoverable.
    Arbitrary .part directories / records without receipts remain errors.
    No child process is launched here; all four numerical files are unchanged.
    """
    if not failure:
        return False
    if (generations.latest_signature(failure) != plan["signature"] or failure.get("incident_index") != ii
            or failure.get("wavelength_index") != ll):
        raise ValueError("Commit recovery failure-ledger signature/index mismatch")
    source = Path(str(path)+".part")
    log = recovery_log(root, failure)
    detail = rename_failure(log, failure["attempts"][-1]["returncode"])
    if not detail or (log_path_key(detail["source"]) != log_path_key(str(source))
                      or log_path_key(detail["destination"]) != log_path_key(str(path))):
        return False
    if os.path.lexists(receipt):
        raise FileExistsError("Commit recovery cannot replace an existing receipt")
    has_source, has_final = os.path.lexists(source), os.path.lexists(path)
    if has_source == has_final:
        raise ValueError("Commit recovery requires exactly one of .part and final directory")
    job = f"i{ii:04d}_w{ll:04d}"
    journal_path = root/"commit_recoveries"/(job+".json")
    journal = load_json(journal_path) if journal_path.exists() else None
    identity = {"format": COMMIT_RECOVERY_VERSION, "signature": plan["signature"],
                "incident_index": ii, "wavelength_index": ll,
                "source": source.relative_to(root).as_posix(),
                "destination": path.relative_to(root).as_posix(),
                "failure_log": log.relative_to(root).as_posix(), "failure_log_sha256": sha256(log)}
    if journal:
        if any(journal.get(k) != v for k, v in identity.items()):
            raise ValueError("Commit recovery journal identity/log mismatch")
        if journal.get("state") not in ("validated", "renamed", "committed"):
            raise ValueError("Invalid commit recovery journal state")
    elif has_final:
        # A final directory is accepted ONLY if this recovery's pre-rename journal
        # recorded its exact content. No general orphan-output adoption is added.
        raise ValueError("Final output without receipt and without a recovery journal")
    candidate = source if has_source else path
    before = record_hashes(candidate)
    checked = verify_record(candidate, plan, ii, ll)  # original validator unchanged
    audit = audit_saved_cdf(candidate, plan)
    if record_hashes(candidate) != before:
        raise ValueError("Record changed during commit recovery validation")
    if journal and (journal.get("files_sha256") != before or journal.get("record") != checked):
        raise ValueError("Recovery journal content hashes no longer match")
    if not journal:
        journal = {**identity, "state": "validated", "created_time_ns": time.time_ns(),
                   "files_sha256": before, "record": checked, "cdf_validation": audit,
                   "original_generator_returncode": failure["attempts"][-1]["returncode"]}
        write_json(journal_path, journal)
    if has_source:
        try:
            journal.update(rename_with_retry(source, path))
        except OSError as exc:
            journal["last_rename_error"] = {"time_ns": time.time_ns(), "message": str(exc),
                                           "errno": exc.errno, "winerror": getattr(exc, "winerror", None)}
            write_json(journal_path, journal)
            raise
        journal["state"] = "renamed"
        write_json(journal_path, journal)
    if record_hashes(path) != before:
        raise ValueError("Record content changed after directory rename; no receipt accepted")
    write_json(receipt, {"signature": plan["signature"], "incident_index": ii,
                        "wavelength_index": ll, "record": checked})
    journal["state"] = "committed"
    journal["committed_time_ns"] = time.time_ns()
    write_json(journal_path, journal)
    failure["state"] = "resolved"
    failure["resolved_time_ns"] = time.time_ns()
    failure["resolution"] = "validated_directory_commit_recovery"
    failure["recovery_journal"] = journal_path.relative_to(root).as_posix()
    write_json(root/"failures"/(job+".json"), failure)
    print(f"[commit-recovery] RECOVERED {job}; saved CDF validated; no GPU recomputation.", flush=True)
    return True


def inspect_job_failure(log: Path, returncode: int) -> dict[str, Any]:
    """Classify an explicit rejection by the unchanged solver, not a crash.

    Read at most 256 KiB of the log tail. Missing/unrecognized diagnostics fail
    closed: they must not turn a driver, filesystem or CLI error into a skipped
    numerical condition. Retain strings rather than parsing NaN into JSON.
    """
    lines = log_tail_lines(log)
    generator_errors = [line for line in lines if line.startswith("rainbow_generate:")]
    diagnostics = [line for line in lines if line.startswith("[cdf-input]")]
    rejection = next((line for line in reversed(lines)
                      if line.startswith("rainbow_generate: CDF input rejected:")), "")
    fields = dict(re.findall(r"([A-Za-z_]+)=([^, ]+)", rejection))
    def positive_count(key: str) -> bool:
        value = fields.get(key, "")
        return value.isdecimal() and int(value) > 0
    before_cdf = any(re.fullmatch(
        r"\[cdf-input\] stage=(?:incoherent|path|focal|diffraction) "
        r"directions=[1-9][0-9]* allow_underresolved=[01] CDF_accumulation_started=0",
        line) for line in diagnostics)
    continuable = (returncode == 1 and bool(rejection) and before_cdf
                   and (positive_count("invalid_values") or positive_count("incomplete")))
    # This is a diagnostic category, not a proof of the underlying geometry.
    boundary = any(re.search(r"^\[cdf-input\] optical flags=64(?: |$)", line)
                   and re.search(r"\bboundary_hits=[1-9][0-9]*(?: |$)", line)
                   for line in diagnostics)
    return {"continuable": continuable,
            "category": ("cdf_input_boundary_pending" if boundary else "cdf_input_rejected")
                        if continuable else "unclassified_generator_failure",
            "message": rejection or (generator_errors[-1] if generator_errors else
                                      "Nonzero generator exit without a recognized safe-to-defer rejection."),
            "generator_error_lines": generator_errors[-8:],
            "cdf_diagnostic_lines": diagnostics[-64:]}


def load_job_failure(path: Path, signature: str | set[str], ii: int, ll: int) -> dict[str, Any]:
    allowed = {signature} if isinstance(signature, str) else set(signature)
    value = load_json(path, 4 << 20)
    if (value.get("format") != FAILURE_VERSION
            or value.get("signature") not in allowed
            or type(value.get("incident_index")) is not int or value["incident_index"] != ii
            or type(value.get("wavelength_index")) is not int or value["wavelength_index"] != ll
            or value.get("state") not in ("unresolved", "resolved")
            or not isinstance(value.get("attempts"), list) or not value["attempts"]):
        raise ValueError(f"Invalid/mismatched failure record: {path}")
    for attempt in value["attempts"]:
        if (not isinstance(attempt, dict) or type(attempt.get("continuable")) is not bool
                or type(attempt.get("returncode")) is not int
                or not isinstance(attempt.get("log"), str)
                or attempt.get("generation_signature", value["signature"]) not in allowed):
            raise ValueError(f"Malformed failure attempt: {path}")
    return value


def save_job_failure(root: Path, plan: dict[str, Any], ii: int, ll: int,
                     log: Path, returncode: int, detail: dict[str, Any],
                     known_signatures: set[str] | None = None) -> dict[str, Any]:
    job = f"i{ii:04d}_w{ll:04d}"
    file = root/"failures"/(job+".json")
    if file.exists():
        value = load_job_failure(file, known_signatures or plan["signature"], ii, ll)
    else:
        c = plan["conditions"]
        value = {"format": FAILURE_VERSION, "signature": plan["signature"],
                 "incident_index": ii, "wavelength_index": ll,
                 "inclination_degrees": c["inclination_degrees"][ii],
                 "wavelength_nm": c["wavelength_nm"][ll], "attempts": []}
    value["state"] = "unresolved"
    value["runner_version"] = RUNNER_VERSION
    value["attempts"].append({"time_ns": time.time_ns(), "returncode": returncode,
                              "log": log.relative_to(root).as_posix(), **detail,
                              "generation_signature": plan["signature"]})
    write_json(file, value)
    return value


def publish_progress(root: Path, signature: str, total: int, completed: int,
                     unresolved: dict[str, dict[str, Any]], attempted: int,
                     start: float, *, last_record: str | None = None,
                     status: str = "running") -> None:
    # A failed or unattempted condition can never count as a completed record.
    waiting = total - completed - len(unresolved)
    if waiting < 0:
        raise RuntimeError("Internal batch accounting error")
    complete = completed == total and not unresolved
    summary = {"runner_version": RUNNER_VERSION, "signature": signature,
               "completed": completed, "total": total, "failed": len(unresolved),
               "unattempted": waiting, "attempted_this_run": attempted,
               "complete": complete, "status": status,
               "elapsed_seconds_this_run": time.monotonic()-start}
    if last_record is not None:
        summary["last_record"] = last_record
    failures = []
    for job, value in sorted(unresolved.items()):
        attempt = value["attempts"][-1]
        failures.append({"job": job, "incident_index": value["incident_index"],
                         "wavelength_index": value["wavelength_index"],
                         "inclination_degrees": value["inclination_degrees"],
                         "wavelength_nm": value["wavelength_nm"],
                         "category": attempt["category"], "log": attempt["log"],
                         "attempt_count": len(value["attempts"]),
                         "generation_signature": generations.latest_signature(value),
                         "failure_record": f"failures/{job}.json"})
    # Per-job failure/receipt records are authoritative; these two files are
    # regenerated summaries. A crash between replacements does not adopt data.
    write_json(root/"failures.json", {"format": "rainbow.condition_failures.v1",
               "signature": signature, "count": len(failures), "jobs": failures})
    write_json(root/"progress.json", summary)

def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-root", type=Path)
    ap.add_argument("--generator", type=Path)
    ap.add_argument("--wavelength-range", nargs=3, metavar=("START_NM","STOP_NM","STEP_NM"))
    ap.add_argument("--radius-mm")
    ap.add_argument("--inclination-count", type=int)
    ap.add_argument("--allow-underresolved", action="store_true", help="explicitly permit recorded underresolution warnings")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, help="attempt at most this many selected conditions; failures also count")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="defer recognized pre-CDF numerical rejections; never accept their data")
    ap.add_argument("--recover-commits", action="store_true",
                    help="validate known final-rename failures and retry directory commit without GPU work")
    ap.add_argument("--retry-failed", action="store_true",
                    help="with --resume, attempt ONLY recorded unresolved failures once each")
    ap.add_argument("--register-generation", metavar="REASON",
                    help="explicitly register changed binaries with the same scientific conditions")
    ap.add_argument("--previous-generator", type=Path,
                    help="preserved previous .exe, with its modules/ directory; used to verify migration")
    ap.add_argument("--retry-failed-and-pending", action="store_true",
                    help="with --resume, retry all failures once and also generate unattempted conditions")
    ap.add_argument("--retire-uncommitted", action="store_true",
                    help="after writers stop, preserve unreceipted interrupted output under retired_outputs and regenerate")
    ap.add_argument("--max-job-failures", type=int, default=100,
                    help="stop after this many rejected jobs in this run (default: 100)")
    ap.add_argument("--max-consecutive-failures", type=int, default=10,
                    help="stop after this many consecutive rejected jobs (default: 10)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="actually start the simulations; default is a dry run")
    mode.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit<=0: raise ValueError("limit must be positive")
    if args.max_job_failures <= 0 or args.max_consecutive_failures <= 0:
        raise ValueError("Failure limits must be positive")
    if args.retry_failed and args.retry_failed_and_pending:
        raise ValueError("Choose --retry-failed OR --retry-failed-and-pending")
    if (args.retry_failed or args.retry_failed_and_pending or args.register_generation is not None
            or args.retire_uncommitted) and not args.resume:
        raise ValueError("Retry/generation-change/retirement options require --resume")
    if args.previous_generator is not None and args.register_generation is None:
        raise ValueError("--previous-generator requires --register-generation")
    raw = load_json(args.config)
    if args.wavelength_range: raw["wavelengths"] = dict(zip(("start_nm","stop_nm","step_nm"),args.wavelength_range))
    if args.radius_mm is not None: raw["radius_mm"]=args.radius_mm
    if args.inclination_count is not None: raw["inclinations"]["count"]=args.inclination_count
    if args.allow_underresolved: raw["solver"]["allow_underresolved"]=True
    c = normalized_config(raw)
    root = (args.out_root or Path(raw["output_root"]))
    exe = (args.generator or Path(raw["generator"]))
    root = (root if root.is_absolute() else ROOT/root).resolve()
    exe = (exe if exe.is_absolute() else ROOT/exe).resolve()
    reserve = int(scalar(raw["reserve_gib"], 0, 1000000)*(1<<30))
    nt,np_ = c["solver"]["cdf_theta"],c["solver"]["cdf_phi"]
    payload = 8*((np_+1)+np_*(nt+1)+(nt+1))
    estimate_per_record = payload+(128<<10) # allowance for headers/metadata/receipt; logs extra
    ni,nl = len(c["inclination_degrees"]),len(c["wavelength_nm"])
    print(f"conditions={ni} x {nl} = {ni*nl}; CDF={nt}x{np_}; array_bytes_per_record={payload}")
    print(f"array_total_TB={payload*ni*nl/1e12:.6f}; output={root}")
    print(f"inclination_centers=[{c['inclination_degrees'][0]}, {c['inclination_degrees'][-1]}] degrees")
    print(f"wavelengths=[{c['wavelength_nm'][0]}, {c['wavelength_nm'][-1]}] nm; inclusive exact-step endpoints")
    print("One subprocess per condition; one GPU job at a time. No context reuse / query batching added.")
    if args.recover_commits:
        print("Commit recovery: verify known final-rename failures, preserve data bytes, "
              "retry permission/share errors up to 7 attempts (6.3s total wait).")
    if args.continue_on_error:
        print("Failure policy: defer recognized pre-CDF rejections; no invalid CDF is saved. "
              f"Limits per run: {args.max_job_failures} failures / "
              f"{args.max_consecutive_failures} consecutive. Remaining failures give exit 2.")
    if c["solver"]["allow_underresolved"]:
        print("WARNING: underresolved diffraction is explicitly permitted, not certified accurate.")
    if c["solver"]["maximum_coarsening_tv"]==1:
        print("WARNING: maximum_coarsening_tv=1 is not an accuracy acceptance threshold.")
    if not args.execute:
        print("DRY RUN: no simulations or files written. Add --execute to start.")
        print(subprocess.list2cmdline(command(exe,root/"records/i0000/w0000",c,0,0)))
        return 0
    binaries = {"generator": exe, **{name: exe.parent/"modules"/name for name in MODULES}}
    for name,path in binaries.items():
        if not path.is_file(): raise FileNotFoundError(f"Missing {name}: {path}")
    spec = {"version": VERSION, "conditions": c,
            "binary_sha256": {name: sha256(path) for name,path in binaries.items()}}
    signature = object_hash(spec)
    plan = {**spec, "signature": signature, "record_layout": "records/i{incident_index:04d}/w{wavelength_index:04d}"}
    root.mkdir(parents=True,exist_ok=True)
    with batch_lock(root):
        known_plans = generations.authorize(root, plan, MODULES, resume=args.resume,
            reason=args.register_generation, previous_generator=args.previous_generator)
        known_signatures = set(known_plans)
        pending=[]; skipped=0
        unresolved: dict[str, dict[str, Any]] = {}
        for ii in range(ni):
            for ll in range(nl):
                job=f"i{ii:04d}_w{ll:04d}";path=root/f"records/i{ii:04d}/w{ll:04d}"
                receipt=root/"receipts"/(job+".json")
                failure_path=root/"failures"/(job+".json")
                failure = load_job_failure(failure_path, known_signatures, ii, ll) if failure_path.exists() else None
                has_part = os.path.lexists(Path(str(path)+".part"))
                orphan_final = os.path.lexists(path) and not os.path.lexists(receipt)
                if args.recover_commits and (has_part or orphan_final) and failure:
                    origin_plan = known_plans[generations.latest_signature(failure)]
                    recover_commit(root, origin_plan, ii, ll, path, receipt, failure)
                # Unreceipted interrupted output is NEVER silently adopted.
                # Opt-in retirement preserves its bytes and triggers recomputation.
                generations.retire_uncommitted(root, ii, ll, known_signatures,
                    rename_with_retry, enabled=args.retire_uncommitted)
                if os.path.lexists(Path(str(path)+".part")):
                    raise ValueError(f"Incomplete .part exists: {path}.part; no recognized/validated commit recovery. "
                                     "Preserve it and inspect the log (use --recover-commits for a known final-rename failure).")
                if path.exists():
                    if not args.resume or not receipt.exists():
                        raise ValueError(f"Existing record without authorized resume/receipt: {path}")
                    saved=load_json(receipt)
                    origin_plan=known_plans.get(saved.get("signature"))
                    if origin_plan is None:
                        raise ValueError(f"Receipt refers to an unregistered generation: {receipt}")
                    actual=verify_record(path,origin_plan,ii,ll)
                    if saved.get("incident_index")!=ii or saved.get("wavelength_index")!=ll or saved.get("record")!=actual:
                        raise ValueError(f"Resume receipt/content mismatch: {path}")
                    skipped+=1
                    if failure and failure["state"] == "unresolved":
                        # Recover a crash after receipt commit but before ledger update.
                        failure["state"] = "resolved"
                        write_json(failure_path, failure)
                else:
                    if receipt.exists(): raise ValueError(f"Receipt exists but record missing: {path}")
                    if failure:
                        if failure["state"] != "unresolved":
                            raise ValueError(f"Resolved failure but record missing: {failure_path}")
                        unresolved[job] = failure
                        if not (args.retry_failed or args.retry_failed_and_pending):
                            if not args.continue_on_error or not failure["attempts"][-1]["continuable"]:
                                raise RuntimeError(f"Unresolved failure: {job}. Use --retry-failed to retry; "
                                                   "only known pre-CDF rejections may be deferred with --continue-on-error.")
                            continue
                    elif args.retry_failed:
                        continue
                    pending.append((ii,ll,job,path,receipt))
        selected=pending if args.limit is None else pending[:args.limit]
        needed=len(selected)*estimate_per_record+reserve
        free=shutil.disk_usage(root).free
        print(f"verified_existing={skipped}; queued={len(pending)}; failed_existing={len(unresolved)}; "
              f"this_run={len(selected)}; free_TB={free/1e12:.3f}")
        if needed>free: raise RuntimeError(f"Insufficient free space: estimated {needed} bytes including reserve; available {free}")
        start=time.monotonic()
        completed = skipped
        failed_this_run = consecutive_failures = attempted = 0
        publish_progress(root, signature, ni*nl, completed, unresolved, attempted, start)
        for n,(ii,ll,job,path,receipt) in enumerate(selected,1):
            free=shutil.disk_usage(root).free
            if free<estimate_per_record+reserve:
                publish_progress(root, signature, ni*nl, completed, unresolved, attempted, start,
                                 status="stopped_free_space")
                raise RuntimeError("Free-space reserve reached; stopping without lowering precision")
            (root/"logs").mkdir(exist_ok=True)
            log=root/"logs"/(job+".log")
            if log.exists():
                log=root/"logs"/(job+f".{time.time_ns()}.log")
            path.parent.mkdir(parents=True,exist_ok=True)
            cmd=command(exe,path,c,ii,ll)
            print(f"[{n}/{len(selected)}] {job}: alpha={c['inclination_degrees'][ii]} lambda={c['wavelength_nm'][ll]}",flush=True)
            with log.open("xb") as stream:
                rc=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode
            attempted += 1
            if rc!=0:
                detail = inspect_job_failure(log, rc)
                # Only a failure BEFORE any dataset writer activity may be deferred.
                # No automatic cleanup or adoption of incomplete/unverified output.
                if path.exists() or Path(str(path)+".part").exists() or receipt.exists():
                    detail["continuable"] = False
                    detail["category"] = "failure_with_output_artifacts"
                unresolved[job] = save_job_failure(root, plan, ii, ll, log, rc, detail, known_signatures)
                recovered = (args.recover_commits
                             and recover_commit(root, plan, ii, ll, path, receipt, unresolved[job]))
                if recovered:
                    unresolved.pop(job)
                    completed += 1
                    consecutive_failures = 0
                    publish_progress(root, signature, ni*nl, completed, unresolved, attempted, start,
                                     last_record=job)
                    continue
                failed_this_run += 1
                consecutive_failures += 1
                continuing = args.continue_on_error and detail["continuable"]
                limit_hit = (failed_this_run >= args.max_job_failures
                             or consecutive_failures >= args.max_consecutive_failures)
                publish_progress(root, signature, ni*nl, completed, unresolved, attempted, start,
                                 last_record=job, status="running" if continuing and not limit_hit else "stopped_error")
                if not continuing or limit_hit:
                    reason = " Failure limit reached." if continuing and limit_hit else ""
                    raise RuntimeError(f"Generator exit={rc}; stopped at {job}. See {log}.{reason} "
                                       "No failed record was accepted or deleted.")
                print(f"  FAILED (deferred): {detail['category']}; see {log}. "
                      "No receipt/data accepted; continuing next condition.", flush=True)
                continue
            # Output errors/metadata mismatches still stop. This verification is
            # unchanged; --continue-on-error is not an integrity bypass.
            checked=verify_record(path,plan,ii,ll)
            write_json(receipt,{"signature":signature,"incident_index":ii,"wavelength_index":ll,"record":checked})
            if job in unresolved:
                value = unresolved.pop(job)
                value["state"] = "resolved"
                value["resolved_time_ns"] = time.time_ns()
                value["resolved_generation_signature"] = signature
                write_json(root/"failures"/(job+".json"), value)
            completed += 1
            consecutive_failures = 0
            publish_progress(root, signature, ni*nl, completed, unresolved, attempted, start, last_record=job)
            print(f"  saved; g={checked['hg_g']:.9g}; Aproj_mm2={checked['projected_area_mm2']:.9g}",flush=True)
        complete = completed == ni*nl and not unresolved
        status = "complete" if complete else ("incomplete_failures" if unresolved else "subset_finished")
        publish_progress(root, signature, ni*nl, completed, unresolved, attempted, start, status=status)
        if complete:
            print("Dataset complete.")
        elif unresolved:
            print(f"Dataset INCOMPLETE: {completed}/{ni*nl} valid records, {len(unresolved)} failed, "
                  f"{ni*nl-completed-len(unresolved)} unattempted. See failures.json. Exit code: 2.")
        else:
            print("Requested subset finished. Repeat with --resume to continue the same plan.")
        return 2 if unresolved else 0



if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("Interrupted. No automatic cleanup of records/.part directories.",file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, struct.error) as exc:
        print(f"generate_phase_dataset: {exc}",file=sys.stderr)
        raise SystemExit(1)
