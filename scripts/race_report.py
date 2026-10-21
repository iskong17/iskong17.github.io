#!/usr/bin/env python3
"""Generate Korean race report posts for the blog's Race category.

Data comes from the Jolpica F1 API (Ergast-compatible, no key needed).
Each report covers the race result with a focus on Max Verstappen:
grid vs finish, lap-by-lap position chart, pit stops, lap pace,
teammate comparison, and championship standings after the round.

Usage:
  python3 scripts/race_report.py --latest            # last finished race, if no post yet
  python3 scripts/race_report.py --season 2026 --round 17
  python3 scripts/race_report.py --season 2026 --backfill
  add --force to overwrite an existing post
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1"
DRIVER = "max_verstappen"
KST = timezone(timedelta(hours=9))
POSTS = Path(__file__).resolve().parent.parent / "_posts"

STATUS_KO = {"R": "DNF", "D": "DSQ", "W": "DNS", "N": "NC", "F": "DNQ", "E": "EX"}

REASON_KO = {
    "Retired": "리타이어", "Accident": "사고", "Collision": "충돌", "Collision damage": "충돌 손상",
    "Spun off": "스핀", "Engine": "엔진 문제", "Power Unit": "파워 유닛 문제", "Gearbox": "기어박스 문제",
    "Hydraulics": "유압 계통 문제", "Brakes": "브레이크 문제", "Suspension": "서스펜션 문제",
    "Electrical": "전기 계통 문제", "Puncture": "펑처", "Disqualified": "실격", "Did not start": "출발하지 못함",
    "Withdrew": "기권", "Overheating": "과열", "Fuel pressure": "연료 압력 문제", "Water leak": "냉각수 누수",
}


def reason_ko(status):
    return REASON_KO.get(status, status)


# ---------- API ----------

_cache = {}


def get(path, retries=8):
    """GET an API path with an in-memory cache and backoff on rate limits (HTTP 429)."""
    if path in _cache:
        return _cache[path]
    url = f"{API}/{path}?format=json&limit=100"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "iskong17.github.io race report"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)["MRData"]
            time.sleep(0.5)  # stay under the burst limit
            _cache[path] = data
            return data
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = 15 * (attempt + 1) if getattr(e, "code", None) == 429 else 3 * (attempt + 1)
            print(f"retry in {wait}s ({e}): {path}", file=sys.stderr)
            time.sleep(wait)


def races(season):
    return get(f"{season}/races")["RaceTable"]["Races"]


def race_results(season, rnd):
    rs = get(f"{season}/{rnd}/results")["RaceTable"]["Races"]
    return rs[0] if rs else None


def driver_standings(season, rnd):
    lists = get(f"{season}/{rnd}/driverStandings")["StandingsTable"]["StandingsLists"]
    return lists[0]["DriverStandings"] if lists else []


def constructor_standings(season, rnd):
    lists = get(f"{season}/{rnd}/constructorStandings")["StandingsTable"]["StandingsLists"]
    return lists[0]["ConstructorStandings"] if lists else []


def pitstops(season, rnd, driver):
    rs = get(f"{season}/{rnd}/drivers/{driver}/pitstops")["RaceTable"]["Races"]
    return rs[0].get("PitStops", []) if rs else []


def laps(season, rnd, driver):
    rs = get(f"{season}/{rnd}/drivers/{driver}/laps")["RaceTable"]["Races"]
    out = []
    for lap in rs[0]["Laps"] if rs else []:
        t = lap["Timings"][0]
        out.append({"lap": int(lap["number"]), "pos": int(t["position"]), "time": to_sec(t.get("time"))})
    return out


# ---------- helpers ----------

def to_sec(s):
    if not s:
        return None
    parts = s.split(":")
    try:
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
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
    return res["positionText"].isdigit() and (s in ("Finished", "Lapped") or s.startswith("+"))


def result_label(res):
    if finished(res):
        return f"P{res['positionText']}"
    if res["positionText"] in STATUS_KO:
        return STATUS_KO[res["positionText"]]
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


# ---------- chart ----------

def position_chart(lap_data, grid, pit_laps):
    """Inline SVG of Max's position per lap. Colors come from the site's CSS variables."""
    if not lap_data:
        return ""
    w, h = 720, 260
    left, right, top, bottom = 44, 16, 16, 34
    n_laps = lap_data[-1]["lap"]
    worst = max([p["pos"] for p in lap_data] + [grid or 1, 10])
    worst = 20 if worst > 10 else 10

    def x(lap):
        return left + (w - left - right) * lap / max(n_laps, 1)

    def y(pos):
        return top + (h - top - bottom) * (pos - 1) / (worst - 1)

    pts = []
    if grid:
        pts.append((x(0), y(grid)))
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
        f'<figure class="race-chart">\n'
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Lap-by-lap position chart">\n'
        + "\n".join(g)
        + f'\n<polyline points="{line}" fill="none" style="stroke: var(--red)" stroke-width="2.5" stroke-linejoin="round"/>\n'
        f'</svg>\n'
        f'<figcaption>랩별 순위 (0랩 = 그리드). 노란 점선은 피트스톱.</figcaption>\n'
        f'</figure>'
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
    circuit = race["Circuit"]["circuitName"]
    loc = race["Circuit"]["Location"]

    out = []
    # --- summary ---
    lead = []
    if max_res:
        grid = int(max_res["grid"])
        grid_txt = "피트레인" if grid == 0 else f"{grid}번 그리드"
        if finished(max_res):
            fin = int(max_res["position"])
            delta = (grid or 20) - fin
            move = f"{delta}계단 상승" if delta > 0 else (f"{-delta}계단 하락" if delta < 0 else "순위 변동 없음")
            if fin == 1:
                pole = "폴 투 윈으로 " if grid == 1 else ""
                lead.append(f"막스 베르스타펜이 {grid_txt}에서 출발해 {pole}**우승**했습니다.")
            else:
                lead.append(f"막스 베르스타펜은 {grid_txt}에서 출발해 **{fin}위**로 레이스를 마쳤습니다 ({move}).")
            led_all = lap_data and all(p["pos"] == 1 for p in lap_data)
            if led_all:
                lead.append("모든 랩을 선두로 달렸습니다.")
        else:
            done = int(max_res["laps"])
            when_out = "첫 랩에서" if done == 0 else f"{done}랩을 달린 뒤"
            reason = "" if max_res["status"] == "Retired" else f" ({reason_ko(max_res['status'])})"
            lead.append(f"막스 베르스타펜은 {grid_txt}에서 출발했지만 {when_out} 리타이어했습니다{reason}.")
        pts = float(max_res["points"])
        if pts:
            lead.append(f"이번 레이스에서 {pts:g}점을 얻었습니다.")
    if winner["Driver"]["driverId"] != DRIVER:
        lead.append(f"우승: {name(winner)} ({winner['Constructor']['name']}).")
    out.append(" ".join(lead))

    # --- key numbers ---
    if max_res:
        led = sum(1 for p in lap_data if p["pos"] == 1)
        rows = [("결과", result_label(max_res)), ("그리드", "Pit lane" if max_res["grid"] == "0" else f"P{max_res['grid']}"),
                ("포인트", f"{float(max_res['points']):g}"), ("완주 랩", max_res["laps"])]
        if led:
            rows.append(("선두 주행", f"{led}랩"))
        fl = max_res.get("FastestLap")
        if fl:
            rows.append(("최고 랩", f"{fl['Time']['time']} (전체 {fl['rank']}위, {fl['lap']}랩)"))
        out.append("## 막스의 레이스\n\n| 항목 | 기록 |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows))

        pit_laps = [int(s["lap"]) for s in stops]
        chart = position_chart(lap_data, int(max_res["grid"]), pit_laps)
        if chart:
            out.append(chart)

        # pit stops
        if stops:
            items = ", ".join(f"{s['lap']}랩 ({s['duration']}초)" for s in stops)
            out.append(f"### 피트스톱\n\n{len(stops)}회 정차: {items}. 시간은 피트레인 진입부터 출구까지 걸린 시간입니다.")

        # pace
        excluded = {1} | set(pit_laps) | {l + 1 for l in pit_laps}
        clean = [p["time"] for p in lap_data if p["time"] and p["lap"] not in excluded]
        if len(clean) >= 5:
            med = statistics.median(clean)
            clean = [t for t in clean if t < med * 1.07]  # drop safety car / traffic laps
            if len(clean) >= 5:
                med = statistics.median(clean)
                spread = statistics.pstdev(clean)
                out.append(
                    "### 페이스\n\n"
                    f"1랩, 피트 인·아웃 랩, 세이프티카 구간을 뺀 {len(clean)}랩 기준으로 "
                    f"중간값 랩타임은 **{fmt_lap(med)}**, 표준편차는 {spread:.3f}초입니다."
                )

        # teammate
        if mate:
            mate_txt = f"P{mate['positionText']}" if finished(mate) else f"{result_label(mate)} ({reason_ko(mate['status'])})"
            out.append(f"### 팀메이트\n\n{name(mate)}: 그리드 P{mate['grid']} → {mate_txt}, {float(mate['points']):g}점.")

    # --- results table ---
    table = ["| 순위 | 드라이버 | 팀 | 그리드 | 기록 | 포인트 |", "|---|---|---|---|---|---|"]
    shown = results[:10]
    if max_res and max_res not in shown:
        shown = shown + [max_res]
    for r in shown:
        rec = r.get("Time", {}).get("time") or r["status"]
        nm = f"**{name(r)}**" if r["Driver"]["driverId"] == DRIVER else name(r)
        table.append(f"| {result_label(r)} | {nm} | {md_escape(r['Constructor']['name'])} | {r['grid']} | {md_escape(rec)} | {float(r['points']):g} |")
    out.append("## 레이스 결과 (Top 10)\n\n" + "\n".join(table))

    # --- championship ---
    me = next((s for s in standings if s["Driver"]["driverId"] == DRIVER), None)
    if me and standings:
        leader = standings[0]
        lines = [f"- 드라이버 챔피언십: **P{me['position']}**, {float(me['points']):g}점, {me['wins']}승"]
        before = next((s for s in prev if s["Driver"]["driverId"] == DRIVER), None)
        if before and before["position"] != me["position"]:
            d = int(before["position"]) - int(me["position"])
            lines[0] += f" (지난 라운드 P{before['position']}에서 {'상승' if d > 0 else '하락'})"
        if leader["Driver"]["driverId"] != DRIVER:
            gap = float(leader["points"]) - float(me["points"])
            lines.append(f"- 선두와의 격차: {gap:g}점 (선두 {leader['Driver']['givenName']} {leader['Driver']['familyName']})")
        else:
            second = standings[1]
            lines.append(f"- 2위와의 격차: +{float(me['points']) - float(second['points']):g}점 (2위 {second['Driver']['givenName']} {second['Driver']['familyName']})")
        rb = next((c for c in cons if c["Constructor"]["constructorId"] == team_id), None)
        if rb:
            lines.append(f"- 컨스트럭터 챔피언십: {rb['Constructor']['name']} P{rb['position']}, {float(rb['points']):g}점")
        out.append(f"## 라운드 {rnd} 이후 챔피언십\n\n" + "\n".join(lines))

    body = "\n\n".join(out)
    result = result_label(max_res) if max_res else "–"
    desc = f"{short_gp(gp)}: 막스 {result}" + (f", 우승 {name(winner)}" if winner["Driver"]["driverId"] != DRIVER else " 우승")

    front = [
        "---",
        f'title: "{season} {short_gp(gp)} 레이스 리포트"',
        f"slug: {season}-r{int(rnd):02d}-{race['Circuit']['circuitId'].replace('_', '-')}",
        f"date: {when.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "category: race",
        f'description: "{desc}"',
        f'gp: "{gp}"',
        f"round: {int(rnd)}",
        f'circuit: "{circuit}, {loc["country"]}"',
        f'result: "{result}"',
        "auto: true",
        "---",
    ]
    filename = f"{when.strftime('%Y-%m-%d')}-{season}-r{int(rnd):02d}-{race['Circuit']['circuitId'].replace('_', '-')}.md"
    return filename, "\n".join(front) + "\n\n" + body + "\n"


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
    print(f"wrote _posts/{filename}")
    return filename


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="current")
    ap.add_argument("--round")
    ap.add_argument("--latest", action="store_true", help="report the most recent race")
    ap.add_argument("--backfill", action="store_true", help="report every finished race of the season")
    ap.add_argument("--force", action="store_true", help="overwrite existing posts")
    a = ap.parse_args()

    if a.latest:
        last = get("current/last/results")["RaceTable"]["Races"]
        if not last:
            print("no finished race yet")
            return
        write(last[0]["season"], last[0]["round"], a.force)
    elif a.backfill:
        season = a.season
        if season == "current":
            season = get("current/races")["RaceTable"]["season"]
        last_round = int(get(f"{season}/last/results")["RaceTable"]["Races"][0]["round"])
        for r in races(season):
            if int(r["round"]) <= last_round:
                write(season, r["round"], a.force)
    elif a.round:
        write(a.season, a.round, a.force)
    else:
        ap.error("use --latest, --backfill, or --season/--round")


if __name__ == "__main__":
    main()
