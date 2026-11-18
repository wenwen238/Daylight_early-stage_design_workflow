#!/usr/bin/env python3
# coding: utf-8
"""Radiance scene of one design case: room, windows, overhangs, ground and sensors.

Used by annual_metrics.py (the simulation) and by the MCP server (drawing the same
geometry in Rhino), so what is drawn is exactly what is simulated.

make_room_geom() builds the room geometry; write_scene() assigns the thesis fixed
inputs as materials.

Fixed inputs (thesis Table 4.2, IES LM-83-23): ceiling 0.70, walls 0.50,
floor 0.20, overhang 0.30, clear glazing VT 0.65, work plane 0.76 m.
Ground (LM-83-23 4.9): modelled polygon 30 m beyond the room perimeter with
reflectance 0.10.
"""

import os
import re
from pathlib import Path

import numpy as np

NX, NY = 32, 32
WORKPLANE_Z = 0.76
CEILING_RHO, WALL_RHO, FLOOR_RHO = 0.70, 0.50, 0.20
OVERHANG_RHO = 0.30
GROUND_RHO = 0.10
GROUND_EXTENT = 30.0          # m beyond the room perimeter
GLASS_VT = 0.65
# Radiance glass transmissivity from visible transmittance
GLASS_TN = GLASS_VT * 1.0895


def make_room_geom(wd, W, D, H, windows):
    """
    Write room.rad for an arbitrary room.

    Parameters
    ----------
    wd      : working directory
    W, D, H : room width(X), depth(Y), height(Z)  in metres
    windows : list of dicts, each with keys:
                face  : 'south'|'north'|'east'|'west'
                x0,x1 : horizontal extent (along face local axis)
                z0,z1 : vertical extent
              For south/north, local axis = X (0..W).
              For east/west,   local axis = Y (0..D).

    Returns
    -------
    Path to room.rad
    """
    os.makedirs(wd, exist_ok=True)
    geom = os.path.join(wd, "room.rad")
    if os.path.exists(geom):
        return geom

    lines = []

    # ── materials ───────────────────────────────────────────────────
    lines += [
        "void plastic wall_mat\n0\n0\n5 0.50 0.50 0.50 0 0\n",
        "void plastic floor_mat\n0\n0\n5 0.30 0.30 0.30 0 0\n",
        "void trans glass\n0\n0\n7 0.60 0.60 0.65 0 0 0.65 0\n",
    ]

    # ── floor & ceiling ─────────────────────────────────────────────
    lines.append(f"floor_mat polygon floor\n0\n0\n12\n"
                 f"  0 0 0  {W} 0 0  {W} {D} 0  0 {D} 0\n")
    lines.append(f"wall_mat polygon ceiling\n0\n0\n12\n"
                 f"  0 0 {H}  0 {D} {H}  {W} {D} {H}  {W} 0 {H}\n")

    # ── solid walls (we punch windows into them) ─────────────────────
    # Build each face as one or more opaque polygons around windows.
    # faces: south(Y=0), north(Y=D), east(X=W), west(X=0)

    def _face_windows(face):
        return [w for w in windows if w["face"].lower() == face]

    def _write_south(lines, W, H, wins):
        """South face (Y=0), horizontal axis=X."""
        cuts = sorted([(w["x0"], w["x1"], w["z0"], w["z1"]) for w in wins])
        _punch_face_x(lines, "south", W, H, 0, cuts,
                      lambda x0, z0, x1, z1: f"  {x0} 0 {z0}  {x0} 0 {z1}  {x1} 0 {z1}  {x1} 0 {z0}\n",
                      lambda x0, z0, x1, z1: f"  {x0} 0 {z0}  {x1} 0 {z0}  {x1} 0 {z1}  {x0} 0 {z1}\n",
                      # solid polygon vertices (Y=0, outward = -Y, so list CW from outside)
                      solid_pts=lambda: f"  0 0 {H}  0 0 0  {W} 0 0  {W} 0 {H}\n")

    def _write_north(lines, W, D, H, wins):
        """North face (Y=D), horizontal axis=X."""
        cuts = sorted([(w["x0"], w["x1"], w["z0"], w["z1"]) for w in wins])
        _punch_face_x(lines, "north", W, H, D, cuts,
                      lambda x0, z0, x1, z1: f"  {x0} {D} {z1}  {x0} {D} {z0}  {x1} {D} {z0}  {x1} {D} {z1}\n",
                      lambda x0, z0, x1, z1: f"  {x0} {D} {z0}  {x1} {D} {z0}  {x1} {D} {z1}  {x0} {D} {z1}\n",
                      solid_pts=lambda: f"  0 {D} {H}  0 {D} 0  {W} {D} 0  {W} {D} {H}\n")

    def _write_east(lines, D, W, H, wins):
        """East face (X=W), horizontal axis=Y."""
        cuts = sorted([(w["x0"], w["x1"], w["z0"], w["z1"]) for w in wins])
        _punch_face_y(lines, "east", D, H, W, cuts,
                      lambda y0, z0, y1, z1: f"  {W} {y0} {z0}  {W} {y0} {z1}  {W} {y1} {z1}  {W} {y1} {z0}\n",
                      lambda y0, z0, y1, z1: f"  {W} {y0} {z0}  {W} {y1} {z0}  {W} {y1} {z1}  {W} {y0} {z1}\n",
                      solid_pts=lambda: f"  {W} 0 {H}  {W} 0 0  {W} {D} 0  {W} {D} {H}\n")

    def _write_west(lines, D, H, wins):
        """West face (X=0), horizontal axis=Y."""
        cuts = sorted([(w["x0"], w["x1"], w["z0"], w["z1"]) for w in wins])
        _punch_face_y(lines, "west", D, H, 0, cuts,
                      lambda y0, z0, y1, z1: f"  0 {y1} {z0}  0 {y1} {z1}  0 {y0} {z1}  0 {y0} {z0}\n",
                      lambda y0, z0, y1, z1: f"  0 {y0} {z0}  0 {y1} {z0}  0 {y1} {z1}  0 {y0} {z1}\n",
                      solid_pts=lambda: f"  0 0 {H}  0 0 0  0 {D} 0  0 {D} {H}\n")

    def _punch_face_x(lines, name, span, H, y_const, cuts, win_pts_fn, _win_pts_fn2, solid_pts):
        """Generic: face whose horizontal axis is X (0..span)."""
        if not cuts:
            lines.append(f"wall_mat polygon {name}\n0\n0\n12\n" + solid_pts())
            return
        # horizontal bands: z=0..z_min, z=z_max..H; lateral strips left/right of each window
        # Simplest robust approach: write left/right/top/bottom strips for each window
        # (works for non-overlapping windows)
        xs_cuts = sorted(set([0] + [c[0] for c in cuts] + [c[1] for c in cuts] + [span]))
        zs_cuts = sorted(set([0] + [c[2] for c in cuts] + [c[3] for c in cuts] + [H]))
        idx = 0
        for ix in range(len(xs_cuts) - 1):
            x0, x1 = xs_cuts[ix], xs_cuts[ix + 1]
            for iz in range(len(zs_cuts) - 1):
                z0, z1 = zs_cuts[iz], zs_cuts[iz + 1]
                cx = (x0 + x1) / 2; cz = (z0 + z1) / 2
                is_win = any(w[0] <= cx <= w[1] and w[2] <= cz <= w[3] for w in cuts)
                if is_win:
                    lines.append(f"glass polygon {name}_win{idx}\n0\n0\n12\n"
                                 + win_pts_fn(x0, z0, x1, z1))
                else:
                    lines.append(f"wall_mat polygon {name}_wall{idx}\n0\n0\n12\n"
                                 + win_pts_fn(x0, z0, x1, z1))
                idx += 1

    def _punch_face_y(lines, name, span, H, x_const, cuts, win_pts_fn, _win_pts_fn2, solid_pts):
        """Generic: face whose horizontal axis is Y (0..span)."""
        ys_cuts = sorted(set([0] + [c[0] for c in cuts] + [c[1] for c in cuts] + [span]))
        zs_cuts = sorted(set([0] + [c[2] for c in cuts] + [c[3] for c in cuts] + [H]))
        idx = 0
        for iy in range(len(ys_cuts) - 1):
            y0, y1 = ys_cuts[iy], ys_cuts[iy + 1]
            for iz in range(len(zs_cuts) - 1):
                z0, z1 = zs_cuts[iz], zs_cuts[iz + 1]
                cy = (y0 + y1) / 2; cz = (z0 + z1) / 2
                is_win = any(w[0] <= cy <= w[1] and w[2] <= cz <= w[3] for w in cuts)
                if is_win:
                    lines.append(f"glass polygon {name}_win{idx}\n0\n0\n12\n"
                                 + win_pts_fn(y0, z0, y1, z1))
                else:
                    lines.append(f"wall_mat polygon {name}_wall{idx}\n0\n0\n12\n"
                                 + win_pts_fn(y0, z0, y1, z1))
                idx += 1

    _write_south(lines, W, H, _face_windows("south"))
    _write_north(lines, W, D, H, _face_windows("north"))
    _write_east(lines, D, W, H, _face_windows("east"))
    _write_west(lines, D, H, _face_windows("west"))

    with open(geom, "w") as f:
        f.write("\n".join(lines) + "\n")

    return geom


def overhang_polygon(shade, W, D):
    u0, u1, z, dep = shade["x0_m"], shade["x1_m"], shade["z_m"], shade["depth_m"]
    face = shade["face"]
    if face == "south":
        pts = [(u0, 0, z), (u1, 0, z), (u1, -dep, z), (u0, -dep, z)]
    elif face == "north":
        pts = [(u0, D, z), (u1, D, z), (u1, D + dep, z), (u0, D + dep, z)]
    elif face == "west":
        pts = [(0, u0, z), (0, u1, z), (-dep, u1, z), (-dep, u0, z)]
    else:
        pts = [(W, u0, z), (W, u1, z), (W + dep, u1, z), (W + dep, u0, z)]
    coords = "  ".join(f"{x:g} {y:g} {zz:g}" for x, y, zz in pts)
    return f"overhang_mat polygon overhang_{face}\n0\n0\n12\n  {coords}\n"


def ground_polygons(W, D):
    """Ground ring at z=0, 30 m beyond the perimeter, with the room footprint left open."""
    e = GROUND_EXTENT
    strips = {
        "south": (-e, -e, W + e, 0),
        "north": (-e, D, W + e, D + e),
        "west": (-e, 0, 0, D),
        "east": (W, 0, W + e, D),
    }
    out = []
    for name, (x0, y0, x1, y1) in strips.items():
        out.append(f"ground_mat polygon ground_{name}\n0\n0\n12\n"
                   f"  {x0:g} {y0:g} 0  {x1:g} {y0:g} 0  {x1:g} {y1:g} 0  {x0:g} {y1:g} 0\n")
    return "\n".join(out)


def write_scene(wd, case, W, D, H):
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    windows = [{"face": w["face"], "x0": w["x0_m"], "x1": w["x1_m"],
                "z0": w["z0_m"], "z1": w["z1_m"]} for w in case["windows"]]
    room = Path(make_room_geom(str(wd), W, D, H, windows))
    text = room.read_text()
    if "ground_mat" not in text:
        # Swap make_room_geom's V1 materials for the thesis fixed inputs.
        swaps = {
            "void plastic wall_mat\n0\n0\n5 0.50 0.50 0.50 0 0\n":
                f"void plastic wall_mat\n0\n0\n5 {WALL_RHO} {WALL_RHO} {WALL_RHO} 0 0\n"
                f"\nvoid plastic ceiling_mat\n0\n0\n5 {CEILING_RHO} {CEILING_RHO} {CEILING_RHO} 0 0\n"
                f"\nvoid plastic overhang_mat\n0\n0\n5 {OVERHANG_RHO} {OVERHANG_RHO} {OVERHANG_RHO} 0 0\n"
                f"\nvoid plastic ground_mat\n0\n0\n5 {GROUND_RHO} {GROUND_RHO} {GROUND_RHO} 0 0\n",
            "void plastic floor_mat\n0\n0\n5 0.30 0.30 0.30 0 0\n":
                f"void plastic floor_mat\n0\n0\n5 {FLOOR_RHO} {FLOOR_RHO} {FLOOR_RHO} 0 0\n",
            "void trans glass\n0\n0\n7 0.60 0.60 0.65 0 0 0.65 0\n":
                f"void glass glass\n0\n0\n3 {GLASS_TN:.4f} {GLASS_TN:.4f} {GLASS_TN:.4f}\n",
            "wall_mat polygon ceiling\n": "ceiling_mat polygon ceiling\n",
        }
        for old, new in swaps.items():
            if old not in text:
                raise RuntimeError(f"room.rad: expected text not found: {old.splitlines()[0]}")
            text = text.replace(old, new)
        text += "\n" + "\n".join(overhang_polygon(s, W, D) for s in case.get("shading", []))
        text += "\n" + ground_polygons(W, D)
        room.write_text(text, encoding="ascii", newline="\n")

    skies = wd / "skyglow.rad"
    skies.write_text(
        "void glow skyglow\n0\n0\n4 1 1 1 0\n"
        "skyglow source sky\n0\n0\n4 0 0 1 180\n"
        "skyglow source ground\n0\n0\n4 0 0 -1 180\n",
        encoding="ascii", newline="\n")

    sensors = wd / "sensors.pts"
    xs = np.linspace(W / (2 * NX), W - W / (2 * NX), NX)
    ys = np.linspace(D / (2 * NY), D - D / (2 * NY), NY)
    with sensors.open("w", newline="\n") as f:
        for y in ys:
            for x in xs:
                f.write(f"{x:.6f} {y:.6f} {WORKPLANE_Z:.6f} 0 0 1\n")
    return room, skies, sensors


def read_polygons(room_rad):
    """Polygons of a room.rad as [{material, name, points}], for drawing in Rhino."""
    lines = Path(room_rad).read_text(encoding="ascii").splitlines()
    polygons = []
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 3 and parts[1] == "polygon":
            count = int(lines[i + 3].split()[0])
            nums, j = [], i + 4
            while len(nums) < count:
                nums.extend(float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lines[j]))
                j += 1
            polygons.append({"material": parts[0], "name": parts[2],
                             "points": [nums[k:k + 3] for k in range(0, count, 3)]})
            i = j
        else:
            i += 1
    return polygons
