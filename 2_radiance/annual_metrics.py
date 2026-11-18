#!/usr/bin/env python3
# coding: utf-8
"""
Annual sDA and ASE with Radiance, for the 560 sampling cases or for one new design.

Paths come from config.py and the scene from radiance_scene.py.

sDA300/50%  : Radiance rcontrib daylight coefficients (reinhart.cal, MF=1,
              sky + ground hemispheres) x gendaymtx -m 1 annual sky matrix.
              gendaymtx row 0 is the ground and aligns with rbin 0, so the
              sky matrix is used without reordering.
ASE1000/250h: direct sun only (-ab 0). gendaymtx -D writes one sun source per
              daylit hour; rcontrib -V+ returns each sun's contribution,
              which is that hour's direct illuminance.

Occupied hours are 08:00-18:00 local clock time every day (3650 h), with EU
daylight saving time applied to the standard-time weather data. No dynamic blinds.

Usage
-----
All 560 cases (a model whose metrics.json exists is skipped; --force reruns):
    python 2_radiance/annual_metrics.py --workers 8
Some cases only:
    python 2_radiance/annual_metrics.py --models model_040 model_418
One new design (a case dict in the same format as fixed_design_cases_560.json):
    python 2_radiance/annual_metrics.py --case-json design.json --out thesis_data/results/mcp_runs/my_design
    The metrics are written to <out>/metrics.json and printed as the last line.
"""

import argparse
import csv
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from radiance_scene import (NX, NY, WORKPLANE_Z, CEILING_RHO, WALL_RHO, FLOOR_RHO,  # noqa: E402
                            OVERHANG_RHO, GROUND_RHO, GROUND_EXTENT, GLASS_VT, write_scene)

RAD_BIN = config.RAD_BIN
RAD_LIB = config.RAD_LIB
CASES = config.CASES_JSON
WEA = config.WEA
OUT_ROOT = config.RADIANCE_560
COMMON_ROOT = config.RADIANCE_COMMON

OCC_START, OCC_END = 8, 18           # local clock time
# The EPW data period starts on Sunday 1 Jan; EU DST runs from the last Sunday of
# March to the last Sunday of October (here 26 Mar - 29 Oct).
EPW_JAN1_WEEKDAY = 6                 # Monday=0 ... Sunday=6
SDA_LUX, SDA_TARGET = 300.0, 0.50
ASE_LUX, ASE_HOURS = 1000.0, 250
RGB_W = np.array([47.4, 120.0, 11.6], dtype=np.float32)   # 179 lm/W x (0.265, 0.670, 0.065)

BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="default: every case in CASES")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ab", type=int, default=5)
    parser.add_argument("--ad", type=int, default=512)
    parser.add_argument("--asamp", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--case-json", type=Path, help="simulate one design case from this JSON file")
    parser.add_argument("--out", type=Path, help="output folder for --case-json")
    return parser.parse_args()


def rad(name):
    exe = RAD_BIN / (f"{name}.exe" if os.name == "nt" else name)
    if not exe.exists():
        raise FileNotFoundError(f"Radiance program not found: {exe}. "
                                "Install Radiance or set RADIANCE_ROOT (see README).")
    return str(exe)


def run(cmd, stdin=None, stdout=None, cwd=None):
    start = time.perf_counter()
    sin = open(stdin, "rb") if stdin else None
    sout = open(stdout, "wb") if stdout else subprocess.PIPE
    try:
        proc = subprocess.run(cmd, stdin=sin, stdout=sout, stderr=subprocess.PIPE, cwd=cwd,
                              creationflags=BELOW_NORMAL)
    finally:
        if sin:
            sin.close()
        if stdout:
            sout.close()
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{proc.stderr.decode('utf-8', errors='replace')[:2000]}")
    return time.perf_counter() - start


def peak_memory_mb():
    """Peak working set of this Python process (Windows)."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32.K32GetProcessMemoryInfo(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    except Exception:
        return float("nan")


# -- common inputs ------------------------------------------------------

def last_sunday(month_end_doy):
    """0-based day of year of the last Sunday on or before month_end_doy."""
    return month_end_doy - (month_end_doy + EPW_JAN1_WEEKDAY - 6) % 7


def occupied_mask():
    """8760 mask of occupied hours in weather-file (standard) time."""
    doy = np.repeat(np.arange(365), 24)
    hour = np.tile(np.arange(24), 365)
    dst = (doy >= last_sunday(89)) & (doy < last_sunday(303))   # 31 Mar, 31 Oct
    shift = dst.astype(int)                                     # clock = standard + 1 h
    return (hour + shift >= OCC_START) & (hour + shift < OCC_END)


def load_sky(path):
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if not line.strip():
                break
        vals = np.fromiter((v for ln in f for v in ln.split()), dtype=np.float32)
    return vals.reshape(146, 8760, 3)


def prepare_common():
    """Sky matrix (occupied hours only) and occupied sun sources, shared by all models."""
    COMMON_ROOT.mkdir(parents=True, exist_ok=True)
    sky_path = COMMON_ROOT / f"sky_m1_g{GROUND_RHO:.2f}.smx"
    sky_occ_path = COMMON_ROOT / "sky_occupied.npy"
    suns_rad = COMMON_ROOT / "suns_all.rad"
    suns_mod = COMMON_ROOT / "suns_all.mod"
    occ_suns = COMMON_ROOT / "suns_occ_dst.rad"
    occ_mod = COMMON_ROOT / "suns_occ_dst.mod"
    occ_hours = COMMON_ROOT / "suns_occ_dst_hours.npy"
    occ = occupied_mask()

    if not sky_path.exists():
        g = f"{GROUND_RHO}"
        run([rad("gendaymtx"), "-m", "1", "-O0", "-g", g, g, g, str(WEA)], stdout=str(sky_path))
    if not sky_occ_path.exists():
        np.save(sky_occ_path, np.ascontiguousarray(load_sky(sky_path)[:, occ, :]))
    if not suns_rad.exists():
        run([rad("gendaymtx"), "-n", "-D", str(suns_rad), "-M", str(suns_mod),
             "-m", "1", "-d", str(WEA)], stdout=os.devnull, cwd=str(COMMON_ROOT))

    # Keep suns that are both occupied and non-zero; solarN is minutes from Jan 1 00:00.
    if not occ_suns.exists():
        blocks = re.split(r"\n(?=void light )", suns_rad.read_text(encoding="ascii").strip())
        keep_txt, keep_mod, keep_hoy = [], [], []
        for block in blocks:
            name = re.search(r"void light (solar(\d+))", block)
            rgb = re.search(r"\n3 (\S+) (\S+) (\S+)\n", block)
            hoy = (int(name.group(2)) - 30) // 60
            if float(rgb.group(2)) <= 0 or not occ[hoy]:
                continue
            keep_txt.append(block)
            keep_mod.append(name.group(1))
            keep_hoy.append(hoy)
        occ_suns.write_text("\n".join(keep_txt) + "\n", encoding="ascii", newline="\n")
        occ_mod.write_text("\n".join(keep_mod) + "\n", encoding="ascii", newline="\n")
        np.save(occ_hours, np.array(keep_hoy, dtype=np.int32))
    return {"sky_occ": str(sky_occ_path), "suns": str(occ_suns), "suns_mod": str(occ_mod),
            "n_suns": int(len(np.load(occ_hours))), "n_occ": int(occ.sum())}


# -- per-model simulation -----------------------------------------------

def read_ascii(path, n_rows, n_cols):
    # Binary (-ff) stdout is corrupted by Windows text-mode newline translation, so use ascii.
    data = np.fromfile(path, dtype=np.float32, sep=" ")
    if data.size != n_rows * n_cols * 3:
        raise ValueError(f"{path.name}: {data.size} floats, expected {n_rows}x{n_cols}x3")
    return data.reshape(n_rows, n_cols, 3)


def compute_model(case, common, dims, args, wd):
    model_start = time.perf_counter()
    W, D, H = dims
    model_id = case["model_id"]
    wd = Path(wd)
    if wd.exists():
        shutil.rmtree(wd)
    room, skies, sensors = write_scene(wd, case, W, D, H)
    n = NX * NY
    t = {}

    sky_oct = wd / "sky_scene.oct"
    sun_oct = wd / "sun_scene.oct"
    t["oconv_s"] = run([rad("oconv"), str(skies), str(room)], stdout=str(sky_oct), cwd=str(wd))
    t["oconv_s"] += run([rad("oconv"), str(room), common["suns"]], stdout=str(sun_oct), cwd=str(wd))

    dc_sky = wd / "dc_sky.txt"
    t["rcontrib_sky_s"] = run([
        rad("rcontrib"), "-I+", "-h",
        "-ab", str(args.ab), "-ad", str(args.ad), "-as", str(args.asamp), "-lw", "0.0001",
        "-e", "MF:1", "-f", "reinhart.cal", "-b", "rbin", "-bn", "Nrbins",
        "-m", "skyglow", str(sky_oct),
    ], stdin=str(sensors), stdout=str(dc_sky), cwd=str(wd))

    dc_sun = wd / "dc_sun_direct.txt"
    t["rcontrib_sun_s"] = run([
        rad("rcontrib"), "-I+", "-h", "-V+",
        "-ab", "0", "-dt", "0", "-dc", "1", "-dj", "0", "-dr", "0",
        "-M", common["suns_mod"], str(sun_oct),
    ], stdin=str(sensors), stdout=str(dc_sun), cwd=str(wd))

    # Occupied hours only: sky columns were pre-filtered, and every sun source is occupied.
    start = time.perf_counter()
    sky = np.load(common["sky_occ"], mmap_mode="r")
    coeff = read_ascii(dc_sky, n, 146)
    lux = np.zeros((n, common["n_occ"]), dtype=np.float32)
    for c in range(3):
        lux += RGB_W[c] * (coeff[:, :, c] @ sky[:, :, c])
    np.maximum(lux, 0.0, out=lux)
    del coeff, sky
    da300 = np.mean(lux >= SDA_LUX, axis=1)
    mean_lux = float(lux.mean())
    del lux

    sun = read_ascii(dc_sun, n, common["n_suns"])
    direct_lux = np.tensordot(sun, RGB_W, axes=([2], [0]))
    del sun
    ase_hours = np.sum(direct_lux > ASE_LUX, axis=1)
    del direct_lux
    t["matrix_s"] = time.perf_counter() - start

    dc_sky.unlink()
    dc_sun.unlink()
    for f in (sky_oct, sun_oct, skies, sensors):
        f.unlink()

    sda = float(np.mean(da300 >= SDA_TARGET))
    ase = float(np.mean(ase_hours > ASE_HOURS))
    metrics = {
        "model_id": model_id,
        "stratum_id": case.get("stratum_id", ""),
        "facade_configuration": case.get("facade_configuration", ""),
        "reference_orientation": case.get("reference_orientation", ""),
        "active_facades": ";".join(case["active_facades"]),
        "window_type": case["window_type"],
        "sill_height_m": case["sill_height_m"],
        "window_width_m": case["window_width_m"],
        "window_height_m": case["window_height_m"],
        "overhang_present": case["overhang_present"],
        "overhang_depth_m": case["overhang_depth_m"],
        "sensor_grid": f"{NX}x{NY}",
        "workplane_z_m": WORKPLANE_Z,
        "occupied_hours": common["n_occ"],
        "sun_sources": common["n_suns"],
        "mean_lux_occupied": mean_lux,
        "sDA_300_50_percent": sda * 100.0,
        "ASE_1000_250_percent": ase * 100.0,
        "max_direct_sun_hours_over_1000": int(ase_hours.max()),
        "radiance_params": f"sky: -ab {args.ab} -ad {args.ad} -as {args.asamp} -lw 0.0001; sun: -ab 0",
        "fixed_inputs": {"ceiling": CEILING_RHO, "walls": WALL_RHO, "floor": FLOOR_RHO,
                         "overhang": OVERHANG_RHO, "ground": GROUND_RHO,
                         "ground_extent_m": GROUND_EXTENT, "glazing_vt": GLASS_VT},
        "timings_s": t,
    }
    np.save(wd / "da300_map.npy", da300.reshape(NY, NX))
    np.save(wd / "ase_hours_map.npy", ase_hours.reshape(NY, NX))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ext = [0, W, 0, D]
    gx, gy = np.linspace(0, W, NX), np.linspace(0, D, NY)
    im0 = axes[0].imshow(da300.reshape(NY, NX) * 100, origin="lower", extent=ext, vmin=0, vmax=100, cmap="viridis")
    if da300.min() < SDA_TARGET < da300.max():
        axes[0].contour(gx, gy, da300.reshape(NY, NX), levels=[SDA_TARGET], colors="w")
    axes[0].set_title(f"{model_id}  DA300 [% occ. hours]\nsDA300/50% = {sda * 100:.1f}%")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(ase_hours.reshape(NY, NX), origin="lower", extent=ext,
                         vmin=0, vmax=max(500, int(ase_hours.max())), cmap="inferno")
    if ase_hours.min() < ASE_HOURS < ase_hours.max():
        axes[1].contour(gx, gy, ase_hours.reshape(NY, NX), levels=[ASE_HOURS], colors="c")
    axes[1].set_title(f"{model_id}  direct-sun hours > 1000 lux\nASE1000/250h = {ase * 100:.1f}%")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xlabel("x [m]  (east +)")
        ax.set_ylabel("y [m]  (north +)")
    fig.tight_layout()
    fig.savefig(wd / "sda_ase_maps.png", dpi=150)
    plt.close(fig)
    metrics["total_model_s"] = time.perf_counter() - model_start
    metrics["worker_peak_memory_mb"] = peak_memory_mb()
    (wd / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# -- parallel driver ----------------------------------------------------

def init_worker():
    os.environ["PATH"] = str(RAD_BIN) + os.pathsep + os.environ.get("PATH", "")
    os.environ["RAYPATH"] = str(RAD_LIB)
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass(ctypes.c_void_p(k32.GetCurrentProcess()), 0x00004000)  # below normal
    except Exception:
        pass


def worker(case, common, dims, args):
    try:
        return case["model_id"], compute_model(case, common, dims, args, OUT_ROOT / case["model_id"]), None
    except Exception:
        return case["model_id"], None, traceback.format_exc()


SUMMARY_FIELDS = [
    "model_id", "stratum_id", "facade_configuration", "reference_orientation", "active_facades",
    "window_type", "sill_height_m", "window_width_m", "window_height_m", "overhang_present",
    "overhang_depth_m", "mean_lux_occupied", "sDA_300_50_percent", "ASE_1000_250_percent",
    "max_direct_sun_hours_over_1000", "total_model_s", "worker_peak_memory_mb",
]


def write_summary(model_ids):
    with open(config.RADIANCE_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for model_id in model_ids:
            path = OUT_ROOT / model_id / "metrics.json"
            if path.exists():
                m = json.loads(path.read_text(encoding="utf-8"))
                writer.writerow({k: m.get(k, "") for k in SUMMARY_FIELDS})


def log(line):
    print(line, flush=True)
    with open(OUT_ROOT / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def room_dims():
    room = json.loads(CASES.read_text(encoding="utf-8"))["metadata"]["room_dimensions_m"]
    return room["width"], room["depth"], room["height"]


def run_single(args):
    """Simulate one design case (used by the MCP server)."""
    init_worker()
    args.out = args.out.resolve()          # Radiance runs with the output folder as working directory
    case = json.loads(args.case_json.read_text(encoding="utf-8"))
    common = prepare_common()
    metrics = compute_model(case, common, room_dims(), args, args.out)
    print(json.dumps({"sDA_300_50_percent": metrics["sDA_300_50_percent"],
                      "ASE_1000_250_percent": metrics["ASE_1000_250_percent"],
                      "out": str(args.out), "total_model_s": metrics["total_model_s"]}))


def main():
    args = parse_args()
    if args.case_json:
        if not args.out:
            raise SystemExit("--case-json needs --out")
        return run_single(args)

    init_worker()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = json.loads(CASES.read_text(encoding="utf-8"))
    dims = room_dims()
    cases = {c["model_id"]: c for c in data["cases"]}
    model_ids = args.models or list(cases)

    run_start = time.perf_counter()
    common = prepare_common()
    todo = [m for m in model_ids if args.force or not (OUT_ROOT / m / "metrics.json").exists()]
    log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] start: {len(todo)} to run, "
        f"{len(model_ids) - len(todo)} already done, workers={args.workers}, "
        f"shared setup {time.perf_counter() - run_start:.1f}s")

    done, failed = 0, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futures = [pool.submit(worker, cases[m], common, dims, args) for m in todo]
        for fut in as_completed(futures):
            model_id, m, err = fut.result()
            done += 1
            if err:
                failed.append(model_id)
                log(f"[{done}/{len(todo)}] {model_id} FAILED\n{err}")
                continue
            log(f"[{done}/{len(todo)}] {model_id}: sDA={m['sDA_300_50_percent']:.1f}%  "
                f"ASE={m['ASE_1000_250_percent']:.1f}%  {m['total_model_s']:.1f}s  "
                f"peak {m['worker_peak_memory_mb']:.0f} MB  elapsed {time.perf_counter() - run_start:.0f}s")

    # the summary always lists every case with results, not only the ones run now,
    # so a partial rerun cannot shrink the training labels
    write_summary([m for m in cases if (OUT_ROOT / m / "metrics.json").exists()])
    elapsed = time.perf_counter() - run_start
    info = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "workers": args.workers,
            "models_run": len(todo), "failed": failed, "wall_time_s": elapsed,
            "wall_time_per_model_s": elapsed / max(len(todo), 1)}
    (OUT_ROOT / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"finished {len(todo)} models in {elapsed:.0f}s with {args.workers} workers; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
