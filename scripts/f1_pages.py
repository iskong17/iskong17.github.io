#!/usr/bin/env python3
"""Generate circuit and driver pages for the blog's Race section.

Circuits (_circuits/<id>.md), for every circuit on the current calendar:
  track outline drawn from OpenF1 car position data, the fastest race lap on
  record, the most successful driver, and the last five winners.
Drivers (_drivers/<id>.md), for every driver on the current grid:
  nationality, championship results of the last three seasons, and team history.

Usage:
  python3 scripts/f1_pages.py --circuits --drivers
  python3 scripts/f1_pages.py --circuit marina_bay
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from f1api import jolpica, jolpica_all, merge_races, openf1

ROOT = Path(__file__).resolve().parent.parent
CIRCUITS = ROOT / "_circuits"
DRIVERS = ROOT / "_drivers"


def yaml_str(s):
    return json.dumps(str(s), ensure_ascii=False)


def to_sec(s):
    try:
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(s.split(":"))))
    except (AttributeError, ValueError):
        return None


def full_name(d):
    return f"{d['givenName']} {d['familyName']}"


# ---------- circuits ----------

def track_outline(circuit_id):
    """SVG path of one fast lap, from OpenF1 location data of the most recent race here."""
    today = datetime.now(timezone.utc).date().isoformat()
    for season in range(datetime.now().year, 2022, -1):
        races = jolpica(f"{season}/races")["RaceTable"]["Races"]
        race = next((r for r in races if r["Circuit"]["circuitId"] == circuit_id and r["date"] <= today), None)
        if not race:
            continue
        sessions = openf1("sessions", year=season, session_name="Race")
        session = next((s for s in sessions if s["date_start"][:10] == race["date"]), None)
        if not session:
            continue
        winner = jolpica(f"{season}/{race['round']}/results/1")["RaceTable"]["Races"]
        if not winner:
            continue
        number = winner[0]["Results"][0]["number"]
        laps = [l for l in openf1("laps", session_key=session["session_key"], driver_number=number) if l.get("lap_duration") and l.get("date_start")]
        if not laps:
            continue
        lap = min(laps, key=lambda l: l["lap_duration"])
        start = datetime.fromisoformat(lap["date_start"])
        end = start + timedelta(seconds=lap["lap_duration"])
        pts = openf1("location", session_key=session["session_key"], driver_number=number,
                     date__gt=start.isoformat(), date__lt=end.isoformat())
        pts = [(p["x"], p["y"]) for p in pts if p.get("x") is not None]
        if len(pts) < 50:
            continue
        return svg_path(pts), season
    return None, None


def path_viewbox(path, pad_ratio=0.06):
    nums = [float(n) for n in path.replace("M", " ").replace("L", " ").replace("Z", " ").replace(",", " ").split()]
    xs, ys = nums[0::2], nums[1::2]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    pad = max(w, h) * pad_ratio
    return f"{min(xs) - pad:.1f} {min(ys) - pad:.1f} {w + 2 * pad:.1f} {h + 2 * pad:.1f}"


def svg_path(pts, size=400, pad=16):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scale = (size - 2 * pad) / max(w, h)
    ox = pad + (size - 2 * pad - w * scale) / 2
    oy = pad + (size - 2 * pad - h * scale) / 2
    coords = [(ox + (x - min(xs)) * scale, oy + (max(ys) - y) * scale) for x, y in pts]  # flip y for SVG
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords) + " Z"


def circuit_page(circuit):
    cid = circuit["circuitId"]
    wins = merge_races(jolpica_all(f"circuits/{cid}/results/1", "RaceTable", "Races"))
    wins = [r for r in wins if r.get("Results")]
    wins.sort(key=lambda r: (int(r["season"]), int(r["round"])))

    counts = Counter(full_name(r["Results"][0]["Driver"]) for r in wins)
    top = counts.most_common(1)[0][1] if counts else 0
    most = sorted(n for n, c in counts.items() if c == top)

    recent = list(reversed(wins))[:5]

    fastest = merge_races(jolpica_all(f"circuits/{cid}/fastest/1/results", "RaceTable", "Races"))
    record = None
    for r in fastest:
        res = r["Results"][0] if r.get("Results") else None
        t = to_sec(res.get("FastestLap", {}).get("Time", {}).get("time")) if res else None
        if t and (record is None or t < record[0]):
            record = (t, res["FastestLap"]["Time"]["time"], full_name(res["Driver"]), r["season"])

    path, path_season = track_outline(cid)
    loc = circuit["Location"]

    lines = [
        "---",
        f"title: {yaml_str(circuit['circuitName'])}",
        f"circuit_id: {cid}",
        f"locality: {yaml_str(loc['locality'])}",
        f"country: {yaml_str(loc['country'])}",
        f"wiki: {yaml_str(circuit['url'])}",
        f"lat: {loc['lat']}",
        f"lng: {loc['long']}",
        f"races_held: {len(wins)}",
        f"first_race: {wins[0]['season'] if wins else 'null'}",
    ]
    if path:
        lines += [f"track_path: {yaml_str(path)}", f"track_viewbox: {yaml_str(path_viewbox(path))}", f"track_source: {path_season}"]
    if record:
        lines += ["lap_record:", f"  time: {yaml_str(record[1])}", f"  driver: {yaml_str(record[2])}", f"  season: {record[3]}"]
    if most:
        lines += ["most_wins:", f"  count: {top}", "  drivers:"] + [f"    - {yaml_str(n)}" for n in most]
    lines.append("recent_winners:")
    for r in recent:
        res = r["Results"][0]
        lines.append(f"  - {{ season: {r['season']}, driver: {yaml_str(full_name(res['Driver']))}, team: {yaml_str(res['Constructor']['name'])} }}")
    lines.append("---")
    CIRCUITS.mkdir(exist_ok=True)
    (CIRCUITS / f"{cid}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote _circuits/{cid}.md" + ("" if path else " (no track outline)"), flush=True)


def circuits(only=None, skip_existing=False):
    season = jolpica("current/races")["RaceTable"]["season"]
    calendar = jolpica(f"{season}/races")["RaceTable"]["Races"]
    for r in calendar:
        c = r["Circuit"]
        if only and c["circuitId"] != only:
            continue
        if skip_existing and (CIRCUITS / f"{c['circuitId']}.md").exists():
            continue
        circuit_page(c)


# ---------- drivers ----------

def team_history(lists):
    """Collapse per-season constructors into spans like 2016–2025 Red Bull."""
    spans = []
    for sl in sorted(lists, key=lambda s: int(s["season"])):
        season = int(sl["season"])
        for c in sl["DriverStandings"][0]["Constructors"]:
            if spans and spans[-1]["team"] == c["name"] and spans[-1]["to"] in (season, season - 1):
                spans[-1]["to"] = season
            else:
                spans.append({"team": c["name"], "from": season, "to": season})
    return spans


def driver_page(d, current_season):
    did = d["driverId"]
    # Standings need a season, so list the driver's seasons first.
    seasons = [s["season"] for s in jolpica(f"drivers/{did}/seasons")["SeasonTable"]["Seasons"]]
    lists = []
    for season in seasons:
        lists += jolpica(f"{season}/drivers/{did}/driverStandings")["StandingsTable"]["StandingsLists"]
    lists = sorted(lists, key=lambda s: int(s["season"]))
    current = next((s for s in lists if s["season"] == current_season), None)
    past = [s for s in lists if s["season"] != current_season][-3:]
    team = current["DriverStandings"][0]["Constructors"][-1] if current else None

    lines = [
        "---",
        f"title: {yaml_str(full_name(d))}",
        f"driver_id: {did}",
        f"number: {d.get('permanentNumber', 'null')}",
        f"code: {yaml_str(d.get('code', ''))}",
        f"nationality: {yaml_str(d['nationality'])}",
        f"wiki: {yaml_str(d['url'])}",
        f"debut: {lists[0]['season'] if lists else current_season}",
    ]
    if team:
        lines += [f"team: {yaml_str(team['name'])}", f"team_id: {team['constructorId']}"]
    if current:
        st = current["DriverStandings"][0]
        lines += ["current:", f"  season: {current['season']}", f"  round: {current['round']}",
                  f"  position: {st.get('position', 'null')}", f"  points: {st['points']}", f"  wins: {st['wins']}"]
    lines.append("recent_seasons:")
    for s in reversed(past):
        st = s["DriverStandings"][0]
        teams = " / ".join(c["name"] for c in st["Constructors"])
        lines.append(f"  - {{ season: {s['season']}, position: {yaml_str(st.get('positionText', st.get('position', '-')))}, points: {st['points']}, wins: {st['wins']}, team: {yaml_str(teams)} }}")
    lines.append("teams:")
    for sp in team_history(lists):
        years = str(sp["from"]) if sp["from"] == sp["to"] else f"{sp['from']}–{sp['to']}"
        lines.append(f"  - {{ years: {yaml_str(years)}, team: {yaml_str(sp['team'])} }}")
    lines.append("---")
    DRIVERS.mkdir(exist_ok=True)
    (DRIVERS / f"{did}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote _drivers/{did}.md", flush=True)


def drivers(only=None, skip_existing=False):
    season = jolpica("current/races")["RaceTable"]["season"]
    for d in jolpica(f"{season}/drivers")["DriverTable"]["Drivers"]:
        if only and d["driverId"] != only:
            continue
        if skip_existing and (DRIVERS / f"{d['driverId']}.md").exists():
            continue
        driver_page(d, season)


def patch_circuits():
    """Add coordinates and a fitted viewBox to existing circuit pages without refetching everything."""
    coords = {c["circuitId"]: c["Location"] for c in jolpica("circuits", limit=100)["CircuitTable"]["Circuits"]}
    for f in sorted(CIRCUITS.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        head = text.split("\n")
        cid = f.stem
        extra = []
        if not any(l.startswith("lat:") for l in head) and cid in coords:
            extra += [f"lat: {coords[cid]['lat']}", f"lng: {coords[cid]['long']}"]
        path_line = next((l for l in head if l.startswith("track_path:")), None)
        if path_line and not any(l.startswith("track_viewbox:") for l in head):
            extra.append(f"track_viewbox: {yaml_str(path_viewbox(json.loads(path_line[len('track_path: '):])))}")
        if extra:
            f.write_text(text.replace("\n---\n", "\n" + "\n".join(extra) + "\n---\n", 1), encoding="utf-8")
            print(f"patched _circuits/{f.name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuits", action="store_true")
    ap.add_argument("--drivers", action="store_true")
    ap.add_argument("--circuit", help="regenerate one circuit by circuitId")
    ap.add_argument("--driver", help="regenerate one driver by driverId")
    ap.add_argument("--skip-existing", action="store_true", help="only create pages that do not exist yet")
    ap.add_argument("--patch", action="store_true", help="add coordinates and viewBox to existing circuit pages")
    a = ap.parse_args()
    if a.circuits or a.circuit:
        circuits(a.circuit, a.skip_existing)
    if a.drivers or a.driver:
        drivers(a.driver, a.skip_existing)
    if a.patch:
        patch_circuits()
    if not (a.circuits or a.circuit or a.drivers or a.driver or a.patch):
        ap.error("choose --circuits, --drivers, --circuit ID, or --driver ID")


if __name__ == "__main__":
    main()
