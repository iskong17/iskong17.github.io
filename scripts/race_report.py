#!/usr/bin/env python3
"""Generate race report posts (English) for the blog's Race > Seasons section.

Data comes from the Jolpica F1 API. Each report focuses on Max Verstappen:
grid vs finish, a lap-by-lap position chart, pit stops, pace, teammate,
the top 10, and championship standings after the round.

Usage:
  python3 scripts/race_report.py --latest            # last finished race, if no post yet
  python3 scripts/race_report.py --season 2026 --round 17
  python3 scripts/race_report.py --season 2024 --backfill
  python3 scripts/race_report.py --from-season 2021 --backfill
  add --force to overwrite existing posts
"""

import argparse
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from f1api import jolpica

DRIVER = "max_verstappen"
KST = timezone(timedelta(hours=9))
POSTS = Path(__file__).resolve().parent.parent / "_posts"

STATUS_CODES = {"R": "DNF", "D": "DSQ", "W": "DNS", "N": "NC", "F": "DNQ", "E": "EX"}


# ---------- data ----------

def race_results(season, rnd):
    rs = jolpica(f"{season}/{rnd}/results")["RaceTable"]["Races"]
    return rs[0] if rs else None


def driver_standings(season, rnd):
    lists = jolpica(f"{season}/{rnd}/driverStandings")["StandingsTable"]["StandingsLists"]
    return lists[0]["DriverStandings"] if lists else []


def constructor_standings(season, rnd):
    lists = jolpica(f"{season}/{rnd}/constructorStandings")["StandingsTable"]["StandingsLists"]
    return lists[0]["ConstructorStandings"] if lists else []


def pitstops(season, rnd, driver):
    rs = jolpica(f"{season}/{rnd}/drivers/{driver}/pitstops")["RaceTable"]["Races"]
    return rs[0].get("PitStops", []) if rs else []


def laps(season, rnd, driver):
    rs = jolpica(f"{season}/{rnd}/drivers/{driver}/laps")["RaceTable"]["Races"]
    out = []
    for lap in rs[0]["Laps"] if rs else []:
        t = lap["Timings"][0]
        out.append({"lap": int(lap["number"]), "pos": int(t["position"]), "time": to_sec(t.get("time"))})
    return out


# ---------- helpers ----------

def to_sec(s):
    if not s:
        return None
    try:
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(s.split(":"))))
    except ValueError:
        return None


def fmt_lap(sec):
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:06.3f}"


def name(res):
    d = res["Driver"]
    return f"{d['givenName']} {d['familyName']}"


def finished(res):
    """Classified finish. Retired cars can still carry a numeric position, so check the status."""
    s = res["status"]
    return res["positionText"].isdigit() and (s in ("Finished", "Lapped") or s.startswith("+") or "Lap" in s)


def result_label(res):
    if finished(res):
        return f"P{res['positionText']}"
    if res["positionText"] in STATUS_CODES:
        return STATUS_CODES[res["positionText"]]
    s = res["status"]
    if s == "Disqualified":
        return "DSQ"
    if s in ("Did not start", "Withdrew"):
        return "DNS"
    return "DNF"


def short_gp(race_name):
    return race_name.replace("Grand Prix", "GP")


def race_time_kst(race):
    t = race.get("time", "12:00:00Z").replace("Z", "+00:00")
    return datetime.fromisoformat(f"{race['date']}T{t}").astimezone(KST)


def md_escape(s):
    return s.replace("|", "\\|")


def ordinal(n):
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def plural(n, word):
    return f"{n:g} {word}" + ("" if n == 1 else "s")


# ---------- chart ----------

def position_chart(lap_data, grid, pit_laps):
    """Inline SVG of Max's position per lap. Colors come from the site's CSS variables."""
    if not lap_data:
        return ""
    w, h = 720, 260
    left, right, top, bottom = 44, 16, 16, 34
    n_laps = lap_data[-1]["lap"]
    worst = max([p["pos"] for p in lap_data] + [grid or 1])
    worst = 20 if worst > 10 else 10

    def x(lap):
        return left + (w - left - right) * lap / max(n_laps, 1)

    def y(pos):
        return top + (h - top - bottom) * (min(pos, worst) - 1) / (worst - 1)

    pts = [(x(0), y(grid))] if grid else []
    pts += [(x(p["lap"]), y(p["pos"])) for p in lap_data]
    line = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)

    g = []
    for pos in sorted({1, 5, 10, 15, 20} & set(range(1, worst + 1))):
        g.append(f'<line x1="{left}" x2="{w - right}" y1="{y(pos):.1f}" y2="{y(pos):.1f}" style="stroke: var(--line-strong)" stroke-width="1"/>')
        g.append(f'<text x="{left - 8}" y="{y(pos) + 4:.1f}" text-anchor="end" style="fill: var(--muted)" font-size="11">P{pos}</text>')
    step = 10 if n_laps > 30 else 5
    for lap in range(0, n_laps + 1, step):
        g.append(f'<text x="{x(lap):.1f}" y="{h - 12}" text-anchor="middle" style="fill: var(--muted)" font-size="11">{lap}</text>')
    for lap in pit_laps:
        g.append(f'<line x1="{x(lap):.1f}" x2="{x(lap):.1f}" y1="{top}" y2="{h - bottom}" style="stroke: var(--yellow)" stroke-width="1.5" stroke-dasharray="4 4"/>')
        g.append(f'<text x="{x(lap):.1f}" y="{top - 4}" text-anchor="middle" style="fill: var(--yellow-ink)" font-size="10" font-weight="700">PIT</text>')

    return (
        '<figure class="race-chart">\n'
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Lap-by-lap position chart">\n'
        + "\n".join(g)
        + f'\n<polyline points="{line}" fill="none" style="stroke: var(--red)" stroke-width="2.5" stroke-linejoin="round"/>\n'
        "</svg>\n"
        "<figcaption>Position by lap (lap 0 = grid). Dashed lines mark pit stops.</figcaption>\n"
        "</figure>"
    )


# ---------- report ----------

def build(season, rnd):
    race = race_results(season, rnd)
    if not race or not race.get("Results"):
        return None
    results = race["Results"]
    max_res = next((r for r in results if r["Driver"]["driverId"] == DRIVER), None)
    team_id = max_res["Constructor"]["constructorId"] if max_res else "red_bull"
    mate = next((r for r in results if r["Constructor"]["constructorId"] == team_id and r["Driver"]["driverId"] != DRIVER), None)
    winner = results[0]

    standings = driver_standings(season, rnd)
    prev = driver_standings(season, int(rnd) - 1) if int(rnd) > 1 else []
    cons = constructor_standings(season, rnd)
    lap_data = laps(season, rnd, DRIVER) if max_res else []
    stops = pitstops(season, rnd, DRIVER) if max_res else []

    when = race_time_kst(race)
    gp = race["raceName"]
    circuit_id = race["Circuit"]["circuitId"]
    loc = race["Circuit"]["Location"]

    out = []
    # --- summary ---
    lead = []
    if max_res:
        grid = int(max_res["grid"])
        grid_txt = "the pit lane" if grid == 0 else f"P{grid}"
        if finished(max_res):
            fin = int(max_res["position"])
            if fin == 1:
                lead.append(f"Max Verstappen won from {'pole' if grid == 1 else grid_txt}.")
            else:
                delta = (grid or 20) - fin
                move = f"up {plural(delta, 'place')}" if delta > 0 else (f"down {plural(-delta, 'place')}" if delta < 0 else "no change from the grid")
                lead.append(f"Max Verstappen started from {grid_txt} and finished **P{fin}** ({move}).")
            if lap_data and all(p["pos"] == 1 for p in lap_data):
                lead.append("He led every lap.")
        else:
            done = int(max_res["laps"])
            when_out = "on the opening lap" if done == 0 else f"after {plural(done, 'lap')}"
            reason = "" if max_res["status"] == "Retired" else f" ({max_res['status'].lower()})"
            lead.append(f"Max Verstappen started from {grid_txt} but retired {when_out}{reason}.")
        pts = float(max_res["points"])
        if pts:
            lead.append(f"He scored {plural(pts, 'point')}.")
    else:
        lead.append("Max Verstappen did not take part in this race.")
    if winner["Driver"]["driverId"] != DRIVER:
        lead.append(f"Winner: {name(winner)} ({winner['Constructor']['name']}).")
    out.append(" ".join(lead))

    if max_res:
        led = sum(1 for p in lap_data if p["pos"] == 1)
        rows = [("Result", result_label(max_res)), ("Grid", "Pit lane" if max_res["grid"] == "0" else f"P{max_res['grid']}"),
                ("Points", f"{float(max_res['points']):g}"), ("Laps completed", max_res["laps"])]
        if led:
            rows.append(("Laps led", str(led)))
        fl = max_res.get("FastestLap")
        if fl and fl.get("Time"):
            rows.append(("Fastest lap", f"{fl['Time']['time']} ({ordinal(fl['rank'])} overall, lap {fl['lap']})"))
        out.append("## Max's race\n\n| Stat | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows))

        pit_laps = [int(s["lap"]) for s in stops]
        chart = position_chart(lap_data, int(max_res["grid"]), pit_laps)
        if chart:
            out.append(chart)

        if stops:
            items = ", ".join(f"lap {s['lap']} ({s['duration']} s)" for s in stops)
            out.append(f"### Pit stops\n\n{plural(len(stops), 'stop')}: {items}. Times are total pit lane time, entry to exit.")

        excluded = {1} | set(pit_laps) | {l + 1 for l in pit_laps}
        clean = [p["time"] for p in lap_data if p["time"] and p["lap"] not in excluded]
        if len(clean) >= 5:
            med = statistics.median(clean)
            clean = [t for t in clean if t < med * 1.07]  # drop safety car and traffic laps
            if len(clean) >= 5:
                med = statistics.median(clean)
                out.append(
                    "### Pace\n\n"
                    f"Over {len(clean)} clean laps (excluding lap 1, in and out laps, and safety car laps), "
                    f"his median lap was **{fmt_lap(med)}** with a spread of {statistics.pstdev(clean):.3f} s."
                )

        if mate:
            mate_txt = f"P{mate['positionText']}" if finished(mate) else f"{result_label(mate)} ({mate['status'].lower()})"
            out.append(f"### Teammate\n\n{name(mate)}: P{mate['grid']} on the grid → {mate_txt}, {plural(float(mate['points']), 'point')}.")

    table = ["| Pos | Driver | Team | Grid | Time / Status | Pts |", "|---|---|---|---|---|---|"]
    shown = results[:10]
    if max_res and max_res not in shown:
        shown = shown + [max_res]
    for r in shown:
        rec = r.get("Time", {}).get("time") or r["status"]
        nm = f"**{name(r)}**" if r["Driver"]["driverId"] == DRIVER else name(r)
        table.append(f"| {result_label(r)} | {nm} | {md_escape(r['Constructor']['name'])} | {r['grid']} | {md_escape(rec)} | {float(r['points']):g} |")
    out.append("## Race result (top 10)\n\n" + "\n".join(table))

    me = next((s for s in standings if s["Driver"]["driverId"] == DRIVER), None)
    if me and standings:
        leader = standings[0]
        first = f"- Drivers' championship: **P{me['position']}**, {plural(float(me['points']), 'point')}, {plural(int(me['wins']), 'win')}"
        before = next((s for s in prev if s["Driver"]["driverId"] == DRIVER), None)
        if before and before["position"] != me["position"]:
            first += f" ({'up' if int(before['position']) > int(me['position']) else 'down'} from P{before['position']})"
        lines = [first]
        if leader["Driver"]["driverId"] != DRIVER:
            gap = float(leader["points"]) - float(me["points"])
            lines.append(f"- Gap to the leader, {leader['Driver']['givenName']} {leader['Driver']['familyName']}: {plural(gap, 'point')}")
        elif len(standings) > 1:
            second = standings[1]
            lines.append(f"- Lead over {second['Driver']['givenName']} {second['Driver']['familyName']} in P2: {plural(float(me['points']) - float(second['points']), 'point')}")
        team = next((c for c in cons if c["Constructor"]["constructorId"] == team_id), None)
        if team:
            lines.append(f"- Constructors' championship: {team['Constructor']['name']} P{team['position']}, {plural(float(team['points']), 'point')}")
        out.append(f"## Championship after round {rnd}\n\n" + "\n".join(lines))

    result = result_label(max_res) if max_res else "–"
    if max_res and result == "P1":
        desc = f"{short_gp(gp)}: Max wins"
    else:
        desc = f"{short_gp(gp)}: Max {result}, won by {name(winner)}"

    slug = f"{season}-r{int(rnd):02d}-{circuit_id.replace('_', '-')}"
    front = [
        "---",
        f'title: "{season} {short_gp(gp)} Race Report"',
        f"slug: {slug}",
        f"date: {when.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "category: race",
        "lang: en",
        f"season: {season}",
        f'description: "{desc}"',
        f'gp: "{gp}"',
        f"round: {int(rnd)}",
        f'circuit: "{race["Circuit"]["circuitName"]}, {loc["country"]}"',
        f"circuit_id: {circuit_id}",
        f'result: "{result}"',
        "auto: true",
        "---",
    ]
    filename = f"{when.strftime('%Y-%m-%d')}-{slug}.md"
    return filename, "\n".join(front) + "\n\n" + "\n\n".join(out) + "\n"


def write(season, rnd, force=False):
    existing = list(POSTS.glob(f"*-{season}-r{int(rnd):02d}-*.md"))
    if existing and not force:
        print(f"skip {season} R{rnd}: {existing[0].name} exists")
        return None
    built = build(season, rnd)
    if not built:
        print(f"skip {season} R{rnd}: no results yet")
        return None
    filename, text = built
    for old in existing:
        old.unlink()
    POSTS.mkdir(exist_ok=True)
    (POSTS / filename).write_text(text, encoding="utf-8")
    print(f"wrote _posts/{filename}", flush=True)
    return filename


def backfill(season, force):
    if season == "current":
        season = jolpica("current/races")["RaceTable"]["season"]
    last = jolpica(f"{season}/last/results")["RaceTable"]["Races"]
    if not last:
        return
    last_round = int(last[0]["round"])
    for r in jolpica(f"{season}/races")["RaceTable"]["Races"]:
        if int(r["round"]) <= last_round:
            write(season, r["round"], force)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="current")
    ap.add_argument("--from-season", type=int, help="with --backfill: every season from this year to now")
    ap.add_argument("--round")
    ap.add_argument("--latest", action="store_true", help="report the most recent race")
    ap.add_argument("--backfill", action="store_true", help="report every finished race of the season(s)")
    ap.add_argument("--force", action="store_true", help="overwrite existing posts")
    a = ap.parse_args()

    if a.latest:
        last = jolpica("current/last/results")["RaceTable"]["Races"]
        if last:
            write(last[0]["season"], last[0]["round"], a.force)
        else:
            print("no finished race yet")
    elif a.backfill:
        if a.from_season:
            current = int(jolpica("current/races")["RaceTable"]["season"])
            for season in range(a.from_season, current + 1):
                backfill(str(season), a.force)
        else:
            backfill(a.season, a.force)
    elif a.round:
        write(a.season, a.round, a.force)
    else:
        ap.error("use --latest, --backfill, or --season/--round")


if __name__ == "__main__":
    main()
