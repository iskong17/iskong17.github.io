#!/usr/bin/env python3
"""Data for the blog's Formula 1 tabs that is not tied to one race.

  --live                 _data/f1_live.json: current standings and calendar
  --progress [SEASON]    _data/f1_charts/<season>.json: season charts as inline SVG
                         (the current season, or every season from 2021 with --progress all)

Data comes from the Jolpica F1 API. Colors come from the site's CSS variables.
"""

import argparse
import json
from pathlib import Path

from f1api import jolpica, jolpica_all

DRIVER = "max_verstappen"
ROOT = Path(__file__).resolve().parent.parent
FIRST_SEASON = 2021


def full_name(d):
    return f"{d['givenName']} {d['familyName']}"


def live():
    ds = jolpica("current/driverStandings")["StandingsTable"]["StandingsLists"]
    cs = jolpica("current/constructorStandings")["StandingsTable"]["StandingsLists"]
    races = jolpica("current/races")["RaceTable"]
    season = races["season"]
    out = {
        "season": season,
        "round": int(ds[0]["round"]) if ds else 0,
        "drivers": [
            {"pos": s.get("position") or s.get("positionText"), "id": s["Driver"]["driverId"], "name": full_name(s["Driver"]),
             "team": s["Constructors"][-1]["name"] if s["Constructors"] else "", "points": float(s["points"]), "wins": int(s["wins"])}
            for s in (ds[0]["DriverStandings"] if ds else [])
        ],
        "constructors": [
            {"pos": s.get("position") or s.get("positionText"), "id": s["Constructor"]["constructorId"], "name": s["Constructor"]["name"],
             "points": float(s["points"]), "wins": int(s["wins"])}
            for s in (cs[0]["ConstructorStandings"] if cs else [])
        ],
        "races": [
            {"round": int(r["round"]), "name": r["raceName"], "circuit": r["Circuit"]["circuitName"], "circuit_id": r["Circuit"]["circuitId"],
             "locality": r["Circuit"]["Location"]["locality"], "country": r["Circuit"]["Location"]["country"],
             "start": f"{r['date']}T{r.get('time', '00:00:00Z')}", "sprint": "Sprint" in r}
            for r in races["Races"]
        ],
    }
    path = ROOT / "_data" / "f1_live.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.name}", flush=True)


def svg_open(w, h, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">'


def figure(body, caption):
    return f'<figure class="race-chart">{body}<figcaption>{caption}</figcaption></figure>'


def round_ticks(x, n, y):
    step = 5 if n > 10 else 1
    ticks = sorted({1, *range(step, n + 1, step)})
    return "".join(f'<text x="{x(r):.1f}" y="{y}" text-anchor="middle" style="fill: var(--muted)" font-size="11">{r}</text>' for r in ticks)


def points_chart(rows, rival):
    w, h = 720, 280
    left, right, top, bottom = 44, 92, 14, 30
    n = rows[-1]["round"]
    peak = max(max(r["max"], r["rival"]) for r in rows) or 1
    unit = next(u for u in (10, 25, 50, 100, 200) if peak / u <= 6)

    def x(r):
        return left + (w - left - right) * r / n

    def y(v):
        return h - bottom - (h - top - bottom) * v / peak

    g = []
    v = 0
    while v <= peak:
        g.append(f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" style="stroke: var(--line-strong)" stroke-width="1"/>')
        g.append(f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end" style="fill: var(--muted)" font-size="11">{v}</text>')
        v += unit
    g.append(round_ticks(x, n, h - 10))
    series = [("rival", "var(--text-2)", rival.split()[-1]), ("max", "var(--red)", "Verstappen")]
    ends = {key: y(rows[-1][key]) for key, _, _ in series}
    if abs(ends["max"] - ends["rival"]) < 14:  # keep the two end labels from overlapping
        hi, lo = ("max", "rival") if ends["max"] <= ends["rival"] else ("rival", "max")
        ends[hi], ends[lo] = ends[hi] - 7, ends[lo] + 7
    for key, color, label in series:
        pts = " ".join(f"{x(r['round']):.1f},{y(r[key]):.1f}" for r in rows)
        g.append(f'<polyline points="{x(0):.1f},{y(0):.1f} {pts}" fill="none" style="stroke: {color}" stroke-width="2.5" stroke-linejoin="round"/>')
        g.append(f'<text x="{x(n) + 6:.1f}" y="{ends[key] + 4:.1f}" style="fill: {color}" font-size="11" font-weight="700">{label}</text>')
    return figure(svg_open(w, h, "Cumulative points by round") + "".join(g) + "</svg>",
                  f"Cumulative points by round: Max Verstappen and {rival}.")


def gap_chart(rows, rival):
    w, h = 720, 200
    left, right, top, bottom = 44, 16, 12, 30
    n = rows[-1]["round"]
    gaps = [r["max"] - r["rival"] for r in rows]
    hi, lo = max(gaps + [0]), min(gaps + [0])  # only draw the side of the zero line the season used
    span = max(1, hi - lo)

    def x(r):
        return left + (w - left - right) * (r - 0.5) / n

    scale = (h - top - bottom) / span
    mid = top + hi * scale
    bw = max(3, (w - left - right) / n - 4)
    g = [f'<line x1="{left}" x2="{w - right}" y1="{mid:.1f}" y2="{mid:.1f}" style="stroke: var(--line-strong)" stroke-width="1"/>']
    for value, y_ in {(hi, top), (0, mid), (lo, h - bottom)}:
        label = f"+{value:g}" if value > 0 else f"{value:g}"
        g.append(f'<text x="{left - 8}" y="{y_ + 4:.1f}" text-anchor="end" style="fill: var(--muted)" font-size="11">{label}</text>')
    for r, gap in zip(rows, gaps):
        top_ = mid - gap * scale if gap > 0 else mid
        color = "var(--red)" if gap >= 0 else "var(--text-2)"
        g.append(f'<rect x="{x(r["round"]) - bw / 2:.1f}" y="{top_:.1f}" width="{bw:.1f}" height="{abs(gap) * scale:.1f}" rx="1" style="fill: {color}"/>')
    g.append(round_ticks(x, n, h - 10))
    return figure(svg_open(w, h, "Points gap by round") + "".join(g) + "</svg>",
                  f"Max's points gap to {rival} after each round. Above the line, Max is ahead.")


def finish_chart(results):
    w, h = 720, 240
    left, right, top, bottom = 44, 16, 14, 30
    n = results[-1]["round"]
    worst = max([r["pos"] for r in results if r["pos"]] + [10])
    worst = 20 if worst > 10 else 10

    def x(r):
        return left + (w - left - right) * (r - 0.5) / n

    def y(p):
        return top + (h - top - bottom - 22) * (p - 1) / (worst - 1)

    g = []
    for p in sorted({1, 5, 10, 15, 20} & set(range(1, worst + 1))):
        g.append(f'<line x1="{left}" x2="{w - right}" y1="{y(p):.1f}" y2="{y(p):.1f}" style="stroke: var(--line-strong)" stroke-width="1"/>')
        g.append(f'<text x="{left - 8}" y="{y(p) + 4:.1f}" text-anchor="end" style="fill: var(--muted)" font-size="11">P{p}</text>')
    out_y = h - bottom - 4
    if any(not r["pos"] for r in results):
        g.append(f'<text x="{left - 8}" y="{out_y + 4:.1f}" text-anchor="end" style="fill: var(--muted)" font-size="11">Out</text>')
    for r in results:
        if r["pos"]:
            color = "var(--yellow)" if r["pos"] == 1 else ("var(--red)" if r["pos"] <= 3 else "var(--text-2)")
            g.append(f'<circle cx="{x(r["round"]):.1f}" cy="{y(r["pos"]):.1f}" r="5" style="fill: {color}"/>')
        else:
            cx = x(r["round"])
            g.append(f'<path d="M{cx - 4:.1f} {out_y - 4:.1f}l8 8m0 -8l-8 8" fill="none" style="stroke: var(--muted)" stroke-width="2"/>')
    g.append(round_ticks(x, n, h - 10))
    return figure(svg_open(w, h, "Finishing position by round") + "".join(g) + "</svg>",
                  "Max's finishing position in each round. Yellow is a win, red a podium; a cross means he was not classified.")


def progress(season):
    results = jolpica_all(f"{season}/drivers/{DRIVER}/results", "RaceTable", "Races")
    if not results:
        print(f"{season}: no results yet", flush=True)
        return
    finishes = []
    for race in results:
        res = race["Results"][0]
        classified = res["positionText"].isdigit()
        finishes.append({"round": int(race["round"]), "pos": int(res["position"]) if classified else None})

    last_round = finishes[-1]["round"]
    final = jolpica(f"{season}/{last_round}/driverStandings")["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    # The other line is the leader, or the driver in P2 when Max leads.
    other = next(s for s in final if s["Driver"]["driverId"] != DRIVER)
    other_id, rival = other["Driver"]["driverId"], full_name(other["Driver"])

    rows = []
    for rnd in range(1, last_round + 1):
        lists = jolpica(f"{season}/{rnd}/driverStandings")["StandingsTable"]["StandingsLists"]
        if not lists:
            continue
        points = {s["Driver"]["driverId"]: float(s["points"]) for s in lists[0]["DriverStandings"]}
        rows.append({"round": rnd, "max": points.get(DRIVER, 0.0), "rival": points.get(other_id, 0.0)})

    out = {"rival": rival, "points": points_chart(rows, rival), "gap": gap_chart(rows, rival), "finish": finish_chart(finishes)}
    folder = ROOT / "_data" / "f1_charts"
    folder.mkdir(exist_ok=True)
    (folder / f"{season}.json").write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote f1_charts/{season}.json", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--progress", nargs="?", const="current", help="a season, 'current' (default) or 'all'")
    a = ap.parse_args()
    if not a.live and not a.progress:
        ap.error("use --live and/or --progress")
    if a.live:
        live()
    if a.progress:
        current = int(jolpica("current/races")["RaceTable"]["season"])
        if a.progress == "all":
            for season in range(FIRST_SEASON, current + 1):
                progress(season)
        else:
            progress(current if a.progress == "current" else int(a.progress))


if __name__ == "__main__":
    main()
