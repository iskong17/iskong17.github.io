#!/usr/bin/env python3
"""Collect driver profiles for the Formula 1 driver pages into _data/driver_profiles.yml.

For each driver page in _drivers/ (which links to the driver's Wikipedia article):
  photo        - the article's lead image from Wikimedia Commons, only if it is
                 freely licensed (CC BY, CC BY-SA, CC0, public domain); saved
                 resized to images/drivers/<id>.jpg with author and license
  born, birthplace, height
               - from Wikidata
  style        - the "Driving style" (or similar) section of the Wikipedia
                 article, falling back to the article summary; quoted with
                 attribution under CC BY-SA 4.0

Usage:
  python3 scripts/driver_profiles.py                 # all drivers
  python3 scripts/driver_profiles.py max_verstappen  # some drivers
"""

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVERS = ROOT / "_drivers"
PHOTOS = ROOT / "images" / "drivers"
DATA = ROOT / "_data" / "driver_profiles.yml"
UA = {"User-Agent": "iskong17.github.io driver pages (https://iskong17.github.io; kong7963@uos.ac.kr)"}
FREE = ("cc by", "cc-by", "cc0", "public domain", "pd")
STYLE_HEADINGS = re.compile(r"^(driving style|racing style|driver profile|style|personality and driving style)$", re.I)


def get(url, as_json=True):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            time.sleep(0.5)
            return json.loads(data) if as_json else data
        except Exception as e:  # network hiccups and rate limits
            if attempt == 4:
                raise
            print(f"retry ({e}): {url}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def front_matter(path):
    out = {}
    for line in path.read_text(encoding="utf-8").split("\n")[1:]:
        if line == "---":
            break
        m = re.match(r"^(\w+): (.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def first_paragraphs(text, limit=900):
    paras = [p.strip() for p in text.split("\n") if p.strip() and not p.startswith("=")]
    out = ""
    for p in paras:
        if len(out) + len(p) > limit and out:
            break
        out = (out + "\n\n" + p).strip()
    if len(out) > limit:  # cut a single long paragraph at a sentence end
        cut = out[:limit].rsplit(". ", 1)[0]
        out = cut + "."
    return out


def style_section(title):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                                "format": "json", "formatversion": 2, "redirects": 1})
    pages = get(f"https://en.wikipedia.org/w/api.php?{q}")["query"]["pages"]
    text = pages[0].get("extract", "") if pages else ""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(=+)\s*(.+?)\s*=+$", line)
        if m and STYLE_HEADINGS.match(m.group(2)):
            level = len(m.group(1))
            body = []
            for nxt in lines[i + 1:]:
                h = re.match(r"^(=+)\s*.+?\s*=+$", nxt)
                if h and len(h.group(1)) <= level:
                    break
                body.append(nxt)
            section = first_paragraphs("\n".join(body))
            if len(section) > 80:
                return section, m.group(2)
    return None, None


def wikidata(qid):
    ent = get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")["entities"][qid]
    claims = ent.get("claims", {})

    def value(prop):
        c = claims.get(prop)
        return c[0]["mainsnak"].get("datavalue", {}).get("value") if c else None

    out = {}
    born = value("P569")
    if born:
        out["born"] = born["time"][1:11]
    place = value("P19")
    if place:
        pid = place["id"]
        labels = get(f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={pid}&props=labels|claims&languages=en&format=json")["entities"][pid]
        name = labels.get("labels", {}).get("en", {}).get("value")
        country = None
        cc = labels.get("claims", {}).get("P17")
        if cc:
            cid = cc[0]["mainsnak"]["datavalue"]["value"]["id"]
            country = get(f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={cid}&props=labels&languages=en&format=json")["entities"][cid]["labels"].get("en", {}).get("value")
        if name:
            out["birthplace"] = f"{name}, {country}" if country and country not in name else name
    height = value("P2048")
    if height:
        amount = float(height["amount"])
        unit = height.get("unit", "")
        if unit.endswith("Q11573"):  # metre
            amount *= 100
        if 140 < amount < 220:
            out["height_cm"] = round(amount)
    return out


def photo(title, driver_id):
    s = get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
    out = {"summary": first_paragraphs(s.get("extract", ""), 700), "qid": s.get("wikibase_item")}
    src = (s.get("originalimage") or s.get("thumbnail") or {}).get("source")
    if not src:
        return out
    src = src.split("?")[0]
    fname = urllib.parse.unquote(src.split("/")[-1])
    if "/thumb/" in src:
        fname = urllib.parse.unquote(src.split("/thumb/")[1].split("/")[2])
    q = urllib.parse.urlencode({"action": "query", "titles": f"File:{fname}", "prop": "imageinfo",
                                "iiprop": "url|extmetadata", "iiurlwidth": 480, "format": "json"})
    pages = get(f"https://commons.wikimedia.org/w/api.php?{q}")["query"]["pages"]
    info = next(iter(pages.values())).get("imageinfo")
    if not info:
        return out
    meta = info[0].get("extmetadata", {})
    license_name = strip_tags(meta.get("LicenseShortName", {}).get("value", ""))
    if not license_name.lower().startswith(FREE):
        print(f"  skip photo for {driver_id}: license {license_name!r}")
        return out
    PHOTOS.mkdir(parents=True, exist_ok=True)
    (PHOTOS / f"{driver_id}.jpg").write_bytes(get(info[0]["thumburl"], as_json=False))
    out["photo"] = {
        "src": f"/images/drivers/{driver_id}.jpg",
        "author": strip_tags(meta.get("Artist", {}).get("value", "")) or "Unknown",
        "license": license_name,
        "source": info[0].get("descriptionurl", ""),
    }
    return out


def yaml_str(s):
    return json.dumps(str(s), ensure_ascii=False)


def load_existing():
    if not DATA.exists():
        return {}
    blocks, current = {}, None
    for line in DATA.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^(\w+):$", line)
        if m:
            current = m.group(1)
            blocks[current] = []
        elif current and line.startswith("  "):
            blocks[current].append(line)
    return blocks


def main():
    only = set(sys.argv[1:])
    blocks = load_existing()
    for path in sorted(DRIVERS.glob("*.md")):
        did = path.stem
        if only and did not in only:
            continue
        if "\ncurrent:\n" not in path.read_text(encoding="utf-8"):
            continue  # practice-only reserve drivers are not listed on the site
        fm = front_matter(path)
        wiki = fm.get("wiki", "")
        if "wikipedia.org/wiki/" not in wiki:
            continue
        title = urllib.parse.unquote(wiki.split("/wiki/")[1]).replace("_", " ")
        print(f"{did}: {title}", flush=True)
        p = photo(title, did)
        facts = wikidata(p["qid"]) if p.get("qid") else {}
        style, heading = style_section(title)

        lines = [f"  wiki_title: {yaml_str(title)}"]
        for k in ("born", "birthplace"):
            if facts.get(k):
                lines.append(f"  {k}: {yaml_str(facts[k])}")
        if facts.get("height_cm"):
            lines.append(f"  height_cm: {facts['height_cm']}")
        if p.get("photo"):
            ph = p["photo"]
            lines += ["  photo:", f"    src: {yaml_str(ph['src'])}", f"    author: {yaml_str(ph['author'])}",
                      f"    license: {yaml_str(ph['license'])}", f"    source: {yaml_str(ph['source'])}"]
        text = style or p.get("summary", "")
        if text:
            lines.append(f"  style_heading: {yaml_str(heading or 'Summary')}")
            lines.append(f"  style: {yaml_str(text)}")
        blocks[did] = lines

    out = ["# Generated by scripts/driver_profiles.py from Wikipedia, Wikidata, and Wikimedia Commons.",
           "# Text is quoted from Wikipedia under CC BY-SA 4.0; photos keep their own free licenses."]
    for did in sorted(blocks):
        out.append(f"{did}:")
        out += blocks[did]
    DATA.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
