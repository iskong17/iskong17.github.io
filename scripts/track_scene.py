#!/usr/bin/env python3
"""Real surroundings for the cockpit view, from an OpenStreetMap extract (ODbL, © OpenStreetMap contributors).

OSM has no line for the Albert Park race track, so the track stays the car position outline already on the
circuit page. It is placed on the map by fitting it to the park roads the circuit runs on, and the water,
buildings and trees are written in the same metres as the game's track.

  track_scene.py albert_park OSM.json
"""

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# circuit id: lap length in metres, and the OSM road names the lap runs on
CIRCUITS = {"albert_park": (5278, {"Lakeside Drive", "Aughtie Drive"})}
RADIUS = 750  # keep scenery this close to the track


def outline(cid, length):
    text = (ROOT / "_circuits" / f"{cid}.md").read_text(encoding="utf-8")
    d = re.search(r'track_path: "([^"]+)"', text).group(1)
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]
    pts = list(zip(nums[::2], nums[1::2]))
    total = sum(math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
    k = length / total
    return [(x * k, y * k) for x, y in pts]


def nearest_cost(track, targets, cell=40):
    grid = {}
    for p in track:
        grid.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(p)
    cost = []
    for t in targets:
        cx, cy = int(t[0] // cell), int(t[1] // cell)
        best = 200.0
        for a in range(-3, 4):
            for b in range(-3, 4):
                for p in grid.get((cx + a, cy + b), ()):
                    best = min(best, math.dist(p, t))
        cost.append(best)
    cost.sort()
    return cost


def densify(line, step=15):
    out = []
    for a, b in zip(line, line[1:]):
        n = max(1, int(math.dist(a, b) / step))
        out += [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n)]
    return out


def fit(track, roads):
    """Rotation and shift taking map metres (east, -north) onto the track. Median distance is the score."""
    tx = sum(p[0] for p in track) / len(track)
    ty = sum(p[1] for p in track) / len(track)
    rx = sum(p[0] for p in roads) / len(roads)
    ry = sum(p[1] for p in roads) / len(roads)
    sparse = track[::2]

    def score(deg, dx, dy):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        moved = [((p[0] - rx) * c - (p[1] - ry) * s + tx + dx, (p[0] - rx) * s + (p[1] - ry) * c + ty + dy) for p in roads]
        cost = nearest_cost(sparse, moved)
        return cost[int(len(cost) * .7)]  # robust: a part of these roads lies off the lap

    best = (1e9, 0, 0, 0)
    for deg in range(0, 360, 6):
        for dx in range(-300, 301, 100):
            for dy in range(-300, 301, 100):
                v = score(deg, dx, dy)
                if v < best[0]:
                    best = (v, deg, dx, dy)
    for span, step in ((6, 2), (2, .5)):
        _, d0, x0, y0 = best
        for deg in [d0 + i * step for i in range(int(-span / step), int(span / step) + 1)]:
            for dx in range(int(x0) - 60, int(x0) + 61, 15 if span > 2 else 5):
                for dy in range(int(y0) - 60, int(y0) + 61, 15 if span > 2 else 5):
                    v = score(deg, dx, dy)
                    if v < best[0]:
                        best = (v, deg, dx, dy)
    return best, (rx, ry, tx, ty)


def stitch(lines):
    """Join a relation's outer ways, which can come in any order and direction, into closed rings."""
    key = lambda g: (g["lat"], g["lon"])
    rings, lines = [], [list(l) for l in lines]
    while lines:
        ring = lines.pop()
        grown = True
        while grown and key(ring[0]) != key(ring[-1]):
            grown = False
            for i, l in enumerate(lines):
                if key(l[0]) == key(ring[-1]):
                    ring += l[1:]
                elif key(l[-1]) == key(ring[-1]):
                    ring += l[-2::-1]
                else:
                    continue
                lines.pop(i)
                grown = True
                break
        rings.append(ring)
    return rings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuit")
    ap.add_argument("osm")
    a = ap.parse_args()
    length, names = CIRCUITS[a.circuit]
    track = outline(a.circuit, length)
    els = json.loads(Path(a.osm).read_text(encoding="utf-8"))["elements"]

    lat0, lon0 = -37.8497, 144.968
    def metres(lat, lon):
        return ((lon - lon0) * 111320 * math.cos(math.radians(lat0)), -(lat - lat0) * 110540)

    roads = []
    for e in els:
        if e.get("tags", {}).get("name") in names and "geometry" in e:
            roads += densify([metres(g["lat"], g["lon"]) for g in e["geometry"]])
    (med, deg, dx, dy), (rx, ry, tx, ty) = fit(track, roads)
    print(f"fit: {len(roads)} road samples, 70th percentile distance {med:.1f} m, rotation {deg} deg")
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))

    def world(lat, lon):
        p = metres(lat, lon)
        return ((p[0] - rx) * c - (p[1] - ry) * s + tx + dx, (p[0] - rx) * s + (p[1] - ry) * c + ty + dy)

    cell = 100
    near = {(int(p[0] // cell), int(p[1] // cell)) for p in track}
    span = RADIUS // cell + 1
    def close(p):
        cx, cy = int(p[0] // cell), int(p[1] // cell)
        return any((cx + i, cy + j) in near for i in range(-span, span + 1) for j in range(-span, span + 1))

    def dist_track(p):
        return min(math.dist(p, q) for q in track[::3])

    out = {"water": [], "buildings": [], "trees": []}
    for e in els:
        t = e.get("tags", {})
        if t.get("natural") == "tree" and "lat" in e:
            p = world(e["lat"], e["lon"])
            if close(p) and dist_track(p) > 15:
                out["trees"].append([round(p[0], 1), round(p[1], 1)])
        elif t.get("natural") == "water" and "geometry" in e:
            out["water"].append([[round(v, 1) for v in world(g["lat"], g["lon"])] for g in e["geometry"]])
        elif t.get("natural") == "water" and "members" in e:
            for ring in stitch([m["geometry"] for m in e["members"] if m.get("role") == "outer" and "geometry" in m]):
                out["water"].append([[round(v, 1) for v in world(g["lat"], g["lon"])] for g in ring])
        elif "building" in t and "geometry" in e:
            poly = [world(g["lat"], g["lon"]) for g in e["geometry"]]
            mid = (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
            if not close(mid) or dist_track(mid) < 12:
                continue
            try:
                h = float(str(t.get("height", "")).replace("m", "").strip())
            except ValueError:
                h = 0
            if not h:
                try:
                    h = float(t.get("building:levels", 0)) * 3.2
                except ValueError:
                    h = 0
            out["buildings"].append({"h": round(h or 6.5, 1), "p": [[round(v, 1) for v in p] for p in poly[:-1]]})
    print({k: len(v) for k, v in out.items()})
    path = ROOT / "assets" / "data" / f"track-{a.circuit}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(path.name, path.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
