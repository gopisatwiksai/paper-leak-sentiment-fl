"""
Generate the Paper Leak Sentiment dataset (1 lakh+ labeled comments).

Backbone : data/raw/paper_leaks.csv  (110 real, sourced paper-leak incidents)
Output   : data/processed/paper_leak_comments.csv

Each comment is grounded in one incident: it only mentions facts the incident
actually has (an arrest count only if arrests were reported, a retest only if a
retest happened, etc.).

Run:  python src/generate_dataset.py
"""
import math
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import comment_bank as cb
import grammar as gr

SEED = 42
PER_CLASS = 33_334                      # final rows per label: 3 x 33,334 = 100,002 (exactly balanced)
GEN_PER_CLASS = 38_000                  # generate a few extra, then re-balance after annotation noise
LABELS = ["negative", "neutral", "positive"]
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "paper_leaks.csv"
OUT = ROOT / "data" / "processed" / "paper_leak_comments.csv"
NOISE_LOG = ROOT / "reports" / "label_noise_log.csv"
CUTOFF = datetime(2026, 11, 6)                  # dataset snapshot date

rng = random.Random(SEED)
nrng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. How people actually refer to each exam (written by hand)
# ---------------------------------------------------------------------------
ALIASES = {
    "PL-0001": ["AIPMT 2004", "CBSE PMT 2004"], "PL-0002": ["HP CPMT 2006", "HP-CPMT"],
    "PL-0003": ["Vyapam Patwari exam", "MP Patwari 2008"], "PL-0004": ["MP PMT 2009", "Vyapam PMT"],
    "PL-0005": ["RRB Mumbai exam", "RRB ALP/ASM 2010"], "PL-0006": ["Vyapam Samvida exam", "MP contract teacher exam"],
    "PL-0007": ["CBSE board exam", "Andaman CBSE paper"], "PL-0008": ["AIEEE 2011", "AIEEE"],
    "PL-0009": ["CG PMT 2011", "CGPMT"], "PL-0010": ["UPTET 2011", "UP TET"],
    "PL-0011": ["MP food inspector exam", "Vyapam food inspector exam"], "PL-0012": ["MP Pre-PG 2012", "Vyapam Pre PG"],
    "PL-0013": ["MP transport constable exam", "Vyapam transport constable exam"], "PL-0014": ["AIIMS PG 2012", "AIIMS PG entrance"],
    "PL-0015": ["Karnataka 2nd PU exam", "2nd PUC paper"], "PL-0016": ["JKCET 2012", "JK CET"],
    "PL-0017": ["MP PMT 2012", "Vyapam PMT 2012"], "PL-0018": ["Vyapam constable exam 2012", "MP police constable exam"],
    "PL-0019": ["Assam ADO exam", "APSC ADO recruitment"], "PL-0020": ["Vyapam constable exam 2013", "MP police constable 2013"],
    "PL-0021": ["SSC CGL 2013", "CGL 2013"], "PL-0022": ["MP PMT 2013", "Vyapam PMT 2013"],
    "PL-0023": ["RAS prelims 2013", "RAS 2013"], "PL-0024": ["CBSE physics paper", "Class 12 physics board exam"],
    "PL-0025": ["DSSSB head clerk exam", "DSSSB 90/09"], "PL-0026": ["UPPCS prelims 2015", "UPPCS 2015"],
    "PL-0027": ["AIPMT 2015", "AIPMT"], "PL-0028": ["HTET 2015", "HTET level 3"],
    "PL-0029": ["WB SLST 2016", "SLST"], "PL-0030": ["Uttarakhand VPDO exam", "VDO exam"],
    "PL-0031": ["NEET 2016", "NEET phase 2"], "PL-0032": ["BSSC inter level exam", "BSSC 2017"],
    "PL-0033": ["NEET 2017", "NEET"], "PL-0034": ["HCS judicial prelims", "Haryana judicial exam"],
    "PL-0035": ["UP SI exam 2017", "UP police inspector exam"], "PL-0036": ["DSSSB PRT exam", "DSSSB primary teacher exam"],
    "PL-0037": ["SSC CGL 2017 Tier 2", "SSC CGL 2017"], "PL-0038": ["CBSE Class 12 economics paper", "CBSE eco paper"],
    "PL-0039": ["CBSE Class 10 maths paper", "CBSE maths paper 2018"], "PL-0040": ["UPPCL JE exam", "UPPCL recruitment"],
    "PL-0041": ["UPSSSC tubewell operator exam", "Nalkoop Chalak exam"], "PL-0042": ["Gujarat non secretariat clerk exam", "Bin Sachivalay exam"],
    "PL-0043": ["Rajasthan librarian exam", "Librarian Grade 3 exam"], "PL-0044": ["Maharashtra TET 2020", "MAHA TET"],
    "PL-0045": ["Assam SI exam", "Assam police SI exam"], "PL-0046": ["Punjab dental MO exam", "Medical Officer Dental exam"],
    "PL-0047": ["Rajasthan JEN exam", "RSMSSB JEN 2020"], "PL-0048": ["Railway GDCE 2021", "RRC Mumbai GDCE"],
    "PL-0049": ["HSSC Gram Sachiv exam", "Gram Sachiv exam"], "PL-0050": ["Army GD CEE 2021", "army soldier GD exam"],
    "PL-0051": ["UKSSSC forest guard exam", "Uttarakhand forest guard exam"], "PL-0052": ["HSSC constable exam", "Haryana police constable exam"],
    "PL-0053": ["NEET 2021", "NEET"], "PL-0054": ["Rajasthan SI exam 2021", "RPSC SI 2021"],
    "PL-0055": ["REET 2021", "REET"], "PL-0056": ["Karnataka PSI exam", "PSI recruitment"],
    "PL-0057": ["Rajasthan Patwari exam", "Patwari exam 2021"], "PL-0058": ["UPTET 2021", "UP TET 2021"],
    "PL-0059": ["UKSSSC graduate level exam", "Uttarakhand graduate level exam"], "PL-0060": ["Gujarat head clerk exam", "GSSSB head clerk exam"],
    "PL-0061": ["MHADA exam", "MHADA recruitment"], "PL-0062": ["TN revision exam", "TN class 10 revision paper"],
    "PL-0063": ["Rajasthan LDC exam", "LDC exam Jodhpur"], "PL-0064": ["Gujarat forest guard exam", "Van Rakshak exam Gujarat"],
    "PL-0065": ["HP police constable exam", "HP constable paper 2022"], "PL-0066": ["AP SSC exams", "AP 10th class exams"],
    "PL-0067": ["BPSC 67th prelims", "BPSC 67th"], "PL-0068": ["Rajasthan constable exam", "Rajasthan police constable exam"],
    "PL-0069": ["Punjab Naib Tehsildar exam", "PPSC Naib Tehsildar exam"], "PL-0070": ["Uttarakhand AE/JE exam", "UKPSC AE JE exam"],
    "PL-0071": ["UPSSSC Lekhpal mains", "Lekhpal exam"], "PL-0072": ["Arunachal AE exam", "APPSC AE civil exam"],
    "PL-0073": ["Rajasthan forest guard exam", "Vanrakshak exam Rajasthan"], "PL-0074": ["BSSC CGL 2022", "BSSC graduate level exam"],
    "PL-0075": ["RPSC 2nd grade teacher exam", "2nd grade teacher GK paper"], "PL-0076": ["HP JOA IT exam", "HPSSC JOA paper"],
    "PL-0077": ["UKPSC Patwari Lekhpal exam", "Uttarakhand Patwari exam"], "PL-0078": ["GPSSB junior clerk exam", "Gujarat junior clerk exam"],
    "PL-0079": ["REET 2023", "REET mains"], "PL-0080": ["TSPSC AE exam", "TSPSC Group 1", "TSPSC exam"],
    "PL-0081": ["MP Patwari 2023", "MP Patwari exam"], "PL-0082": ["CGPSC 2021", "CGPSC exam"],
    "PL-0083": ["MP forest guard exam", "MP Vanrakshak 2023"], "PL-0084": ["AIIMS NORCET 4", "NORCET"],
    "PL-0085": ["Odisha JE civil exam", "OSSC JE exam"], "PL-0086": ["Maharashtra Talathi exam", "Talathi bharti"],
    "PL-0087": ["Bihar constable exam", "CSBC constable exam", "Bihar police constable exam"], "PL-0088": ["JSSC CGL", "JSSC CGL 2024"],
    "PL-0089": ["UPPSC RO/ARO", "RO ARO exam", "RO/ARO prelims"], "PL-0090": ["UP Police constable exam", "UP constable paper", "UP police bharti"],
    "PL-0091": ["BPSC TRE 3", "TRE 3.0", "BPSC teacher exam"], "PL-0092": ["JPSC prelims", "JPSC CSE"],
    "PL-0093": ["NEET 2024", "NEET UG 2024", "NEET"], "PL-0094": ["UGC NET 2024", "UGC NET June 2024", "UGC NET"],
    "PL-0095": ["MPPSC prelims 2024", "MPPSC 2024"], "PL-0096": ["NEET PG 2024", "NEET PG"],
    "PL-0097": ["JSSC CGL re-exam", "JSSC CGL"], "PL-0098": ["Bihar CHO exam", "NHM CHO exam"],
    "PL-0099": ["BPSC 70th", "BPSC 70th prelims"], "PL-0100": ["ECR loco pilot exam", "railway promotion exam"],
    "PL-0101": ["UPESSC assistant professor exam", "UP assistant professor exam"], "PL-0102": ["UKSSSC graduate level exam 2025", "Uttarakhand graduate level exam"],
    "PL-0103": ["Maharashtra TET 2025", "MAHA TET"], "PL-0104": ["Punjab Group B exam", "PSSSB Group B exam"],
    "PL-0105": ["Maharashtra HSC chemistry paper", "HSC chem paper"], "PL-0106": ["CGBSE Hindi paper", "CG board class 12 Hindi exam"],
    "PL-0107": ["JECCE", "Jharkhand excise constable exam"], "PL-0108": ["NEET 2026", "NEET UG 2026", "NEET", "Re-NEET"],
    "PL-0109": ["HTET 2026", "HTET PGT"], "PL-0110": ["Punjab pharmacist officer exam", "pharmacist officer exam"],
}
BODY_OVERRIDE = {
    "PL-0022": "Vyapam",
    "PL-0017": "Vyapam",
    "PL-0013": "Vyapam",
    "PL-0012": "Vyapam",
    "PL-0011": "Vyapam",
    "PL-0004": "Vyapam",
    "PL-0002": "HP University", "PL-0010": "UP board", "PL-0015": "the PU department", "PL-0034": "the High Court",
    "PL-0050": "the Army", "PL-0056": "Karnataka police", "PL-0062": "the education department",
    "PL-0063": "the High Court", "PL-0064": "the forest department", "PL-0065": "HP police",
    "PL-0066": "the AP board", "PL-0068": "the police recruitment board", "PL-0086": "the revenue department",
    "PL-0098": "NHM Bihar", "PL-0100": "the railway board", "PL-0105": "the Maharashtra board",
    "PL-0096": "NBEMS", "PL-0053": "NTA", "PL-0093": "NTA", "PL-0094": "NTA", "PL-0108": "NTA",
    "PL-0084": "AIIMS", "PL-0045": "SLPRB", "PL-0005": "RRB", "PL-0048": "the railways",
}
HINDI_BELT = ("Uttar Pradesh", "Bihar", "Madhya Pradesh", "Rajasthan", "Haryana", "Jharkhand",
              "Chhattisgarh", "Uttarakhand", "Delhi", "Himachal", "All India")
TELUGU = ("Telangana", "Andhra Pradesh")

# ---------------------------------------------------------------------------
# 2. Incident features
# ---------------------------------------------------------------------------
def short_body(iid, text):
    if iid in BODY_OVERRIDE:
        return BODY_OVERRIDE[iid]
    m = re.findall(r"\(([^)]+)\)", text)
    if m:
        return re.split(r"[/,]", m[-1])[0].strip()
    return text.split(",")[0].strip()

def clean_state(area):
    s = re.sub(r"\s*\(.*?\)", "", area).split("/")[0].strip()
    return s.replace("NCT of Delhi", "Delhi")

def category(name):
    n = name.lower()
    if any(k in n for k in ["neet", "pmt", "aipmt", "pre-pg", "aiims post", "jkcet", "aieee"]):
        return "entrance"
    if any(k in n for k in ["board exam", "class 10", "class 12", "pu (class", "second pu", "revision exam", "ssc (class"]):
        return "board"
    return "recruit"

def fmt_asp(x):
    if x >= 100_000:
        v = x / 100_000
        return f"{v:.0f} lakh" if v >= 10 else f"{v:.1f} lakh".replace(".0 ", " ")
    return f"{int(round(x, -2)):,}"

def agency_of(action, central):
    a = action.upper()
    if "CBI" in a and central:
        return "CBI"
    for key, name in [("STF", "STF"), ("SOG", "SOG"), ("ED", "ED"), ("VB", "Vigilance Bureau"), ("SIT", "SIT"), ("CBI", "CBI")]:
        if re.search(rf"\b{key}\b", a):
            return name
    return "police"

def load_incidents():
    df = pd.read_csv(RAW)
    rows = []
    for r in df.itertuples():
        act = r.action_taken
        date = datetime.strptime(r.date, "%Y-%m-%d")
        central = r.body_type == "Central"
        cat = category(r.exam_name)
        n = int(r.arrests) if not pd.isna(r.arrests) and r.arrests > 0 else None
        asp = float(r.aspirants_affected) if not pd.isna(r.aspirants_affected) else None
        tags = {cat}
        if re.search(r"cancel", act, re.I): tags.add("cancel")
        if re.search(r"retest", act, re.I): tags.add("retest")
        if re.search(r"arrest", act, re.I) or n: tags.add("arrests")
        if n: tags.add("n")
        if re.search(r"probe", act, re.I): tags.add("probe")
        if act.strip().lower() == "none reported": tags.add("noaction")
        if r.leak_status == "Confirmed": tags.add("confirmed")
        if r.leak_status in ("Alleged", "Suspected"): tags.add("alleged")
        if r.leak_status == "Denied": tags.add("denied")
        if asp: tags.add("asp")
        if not r.area.startswith("All India"): tags.add("state")
        if date >= datetime(2024, 6, 21): tags.add("post2024")
        day_known = not (date.day == 1 and ("unknown" in r.note.lower() or "1st of month" in r.note.lower()))
        when = f"on {date.day} {date:%b %Y}" if day_known and date.day != 1 else f"in {date:%B %Y}"
        rows.append(dict(
            incident_id=r.incident_id, exam_name=r.exam_name, date=date, year=date.year,
            state=clean_state(r.area), area=r.area, body=short_body(r.incident_id, r.conducting_body),
            body_type=r.body_type, leak_status=r.leak_status, action=act, category=cat,
            n=n, asp=asp, agency=agency_of(act, central) if "probe" in tags else None,
            tags=tags, when=when, aliases=ALIASES[r.incident_id],
        ))
    return rows

# ---------------------------------------------------------------------------
# 3. Comment bank parsing
# ---------------------------------------------------------------------------
def parse(block):
    out = []
    for line in block.strip().splitlines():
        req, who, text = [p.strip() for p in line.split(";;")]
        out.append((set() if req == "-" else set(req.split(",")), who, text))
    return out

BANK = {
    "en": {"negative": parse(cb.EN_NEG), "positive": parse(cb.EN_POS), "neutral": parse(cb.EN_NEU)},
    "hinglish": {"negative": parse(cb.HI_NEG), "positive": parse(cb.HI_POS), "neutral": parse(cb.HI_NEU)},
    "hi": {"negative": parse(cb.DV_NEG), "positive": parse(cb.DV_POS), "neutral": parse(cb.DV_NEU)},
    "te-en": {"negative": parse(cb.TE_NEG), "positive": parse(cb.TE_POS), "neutral": parse(cb.TE_NEU)},
}
GRAMMAR = {
    "en": {k: parse(v) for k, v in gr.EN.items()},
    "hinglish": {k: parse(v) for k, v in gr.HI.items()},
}
P_GRAMMAR = 0.72          # share of comments built from the phrase grammar (rest: full hand-written sentences)

SYM = re.compile(r"(?<!<)<([A-Z_]+)(?::([ul]))?>(?!>)")
ALT = re.compile(r"\[([^\[\]]*)\]")

def alternations(s):
    while True:
        m = ALT.search(s)
        if not m:
            return s
        s = s[:m.start()] + rng.choice(m.group(1).split("|")) + s[m.end():]

def expand(lang, sym, inc, persona):
    ok = [t for t in GRAMMAR[lang][sym] if t[0] <= inc["tags"]]
    if not ok:                                   # nothing fits this incident -> fall back to a generic fact
        ok = [t for t in GRAMMAR[lang]["FACT"] if t[0] <= inc["tags"]]
    req, who, text = rng.choice(ok)
    if who != "-" and persona[0] == "general":
        persona[0] = who
    def sub(m):
        out = expand(lang, m.group(1), inc, persona)
        if m.group(2) == "u": out = out[:1].upper() + out[1:]
        if m.group(2) == "l": out = out[:1].lower() + out[1:]
        return out
    return SYM.sub(sub, alternations(text))

# ---------------------------------------------------------------------------
# 4. Platforms = future federated clients (label mix differs per platform: non-IID)
# ---------------------------------------------------------------------------
PLATFORMS = ["YouTube", "X", "Instagram", "Reddit", "NewsPortal"]
P_PLATFORM = {   # P(platform | label)
    "negative": [0.30, 0.25, 0.20, 0.15, 0.10],
    "neutral":  [0.15, 0.20, 0.10, 0.25, 0.30],
    "positive": [0.25, 0.20, 0.25, 0.15, 0.15],
}
LAUNCH = {"YouTube": 2006, "X": 2007, "Instagram": 2011, "Reddit": 2009, "NewsPortal": 2003}
STYLE = {   # how people type on each platform
    "YouTube":    dict(hinglish=0.60, lower=0.55, emoji=0.45, hashtag=0.05, second=0.08, txt=0.35, typo=0.14),
    "X":          dict(hinglish=0.30, lower=0.35, emoji=0.30, hashtag=0.55, second=0.10, txt=0.25, typo=0.10),
    "Instagram":  dict(hinglish=0.45, lower=0.50, emoji=0.65, hashtag=0.30, second=0.06, txt=0.30, typo=0.12),
    "Reddit":     dict(hinglish=0.05, lower=0.15, emoji=0.06, hashtag=0.00, second=0.40, txt=0.08, typo=0.04),
    "NewsPortal": dict(hinglish=0.03, lower=0.10, emoji=0.04, hashtag=0.00, second=0.30, txt=0.05, typo=0.05),
}
TXT = [(r"\byou\b", "u"), (r"\byour\b", "ur"), (r"\bare\b", "r"), (r"\bplease\b", "pls"), (r"\bpeople\b", "ppl"),
       (r"\bbecause\b", "coz"), (r"\breally\b", "rly"), (r"\bthough\b", "tho"), (r"\bthanks\b", "thx"),
       (r"\bbefore\b", "b4"), (r"\bsomething\b", "smth"), (r"\bwhat\b", "wat"), (r"\bok\b", "k")]
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")

# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------
def fill(text, inc):
    s = alternations(text)
    s = s.replace("{exam}", rng.choice(inc["aliases"]))
    voc = inc["body"][4:] if inc["body"].startswith("the ") else inc["body"]   # "Wake up NTA", not "Wake up the NTA"
    s = re.sub(r"(Do better|Wake up|Thanks|Thank you|Congrats|job|wah|Wah|Shame on) \{body\}", lambda m: f"{m.group(1)} {voc}", s)
    s = s.replace("{body}", inc["body"])
    s = s.replace("{state}", inc["state"])
    s = s.replace("{n}", str(inc["n"]) if inc["n"] else "")
    s = s.replace("{agency}", inc["agency"] or "police")
    s = s.replace("{asp}", fmt_asp(inc["asp"]) if inc["asp"] else "")
    s = s.replace("{yr}", str(inc["year"]))
    s = s.replace("{when}", inc["when"])
    s = re.sub(r"\bpaper (question )?paper\b", lambda m: "question paper" if m.group(1) else "paper", s, flags=re.I)
    return s

def typo(s):
    words = s.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not idx:
        return s
    i = rng.choice(idx); w = words[i]; j = rng.randrange(1, len(w) - 2)
    op = rng.random()
    if op < 0.5:   w = w[:j] + w[j + 1] + w[j] + w[j + 2:]      # swap
    elif op < 0.8: w = w[:j] + w[j + 1:]                          # drop
    else:          w = w[:j] + w[j] + w[j:]                       # double
    words[i] = w
    return " ".join(words)

def stylise(s, platform, lang, label, keep_emoji=False):
    st = STYLE[platform]
    if lang in ("en", "hinglish"):
        if rng.random() < st["txt"]:
            for pat, rep in rng.sample(TXT, k=4):
                s = re.sub(pat, rep, s, flags=re.I)
        if rng.random() < st["typo"]:
            s = typo(s)
        if rng.random() < st["lower"]:
            s = s.lower().replace("'", "")
        if label == "negative" and rng.random() < 0.06:
            words = s.split(" "); k = rng.randrange(len(words))
            words[k] = words[k].upper(); s = " ".join(words)        # one shouted word
        if rng.random() < 0.15:
            s = s.rstrip(".")                                       # people skip full stops
        if label == "negative" and rng.random() < 0.12 and s[-1:].isalnum() or (label == "negative" and s.endswith(".") and rng.random() < 0.12):
            s = s.rstrip(".!") + rng.choice(["!!", "!!!", "..."])
        if label == "neutral" and s.endswith("?") and rng.random() < 0.15:
            s += "?"
    if not keep_emoji and rng.random() > st["emoji"]:
        stripped = EMOJI.sub("", s).replace("‍", "").replace("️", "")
        if len(stripped.strip()) >= 3:
            s = stripped
    return re.sub(r"\s{2,}", " ", s).strip()

def pick(pool, inc):
    ok = [t for t in pool if t[0] <= inc["tags"]]
    return rng.choice(ok)

def join(a, b):
    a = a.rstrip()
    if a and a[-1] not in ".!?" and not EMOJI.search(a[-1]):
        a += "."
    return f"{a} {b}"

def capitalise(s):
    s = s[:1].upper() + s[1:]
    return re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)

def build(inc, label, platform, lang):
    st = STYLE[platform]
    persona = ["general"]
    if lang in GRAMMAR and rng.random() < P_GRAMMAR:
        s = fill(expand(lang, label.upper(), inc, persona), inc)
        if lang == "en":
            s = capitalise(s)
        who = persona[0]
    else:
        req, who, text = pick(BANK[lang][label], inc)
        s = fill(text, inc)
        # optional second sentence (generic, no exam name), more common on Reddit / news sites
        if rng.random() < st["second"]:
            generic = [t for t in BANK[lang][label] if t[0] <= inc["tags"] and "{exam}" not in t[2] and t[2] != text]
            if generic:
                s = join(s, fill(rng.choice(generic)[2], inc))
    keep = "<<KEEP>>" in s
    s = s.replace("<<KEEP>>", "")
    s = re.sub(r"\s+([.,!?])", r"\1", s).replace("..", ".").replace(".?", "?")
    s = re.sub(r"([!?]),", r"\1", s)
    if lang in ("en", "hinglish"):
        if rng.random() < 0.30:
            op = rng.choice(cb.OPENERS["shared"] + cb.OPENERS[label])
            if op:
                ends_sentence = op[-1] in ".!?:"
                keep_case = (s[:2] in ("I ", "I'") or sum(c.isupper() for c in s.split(" ")[0]) >= 2
                             or any(s.startswith(a) for a in inc["aliases"]) or s.startswith(inc["body"]))
                s = f"{op} {s if ends_sentence or keep_case else s[0].lower() + s[1:]}"
        if rng.random() < 0.35:
            cl = rng.choice(cb.CLOSERS["shared"] + cb.CLOSERS[label])
            if cl in ("!!", "..."):
                s = s.rstrip(".!") + cl
            elif cl and cl[0].isalpha() and cl[0].isupper():
                s = join(s, cl)
            elif cl:
                s = f"{s} {cl}"
    s = paraphrase(s, lang)
    s = stylise(s, platform, lang, label, keep_emoji=keep)
    if rng.random() < st["hashtag"]:
        tags = cb.HASHTAGS["shared"] + cb.HASHTAGS[label]
        if "arrests" not in inc["tags"]:
            tags = [t for t in tags if t != "#JusticeServed"]
        if "retest" not in inc["tags"]:
            tags = [t for t in tags if t != "#RetestReady"]
        alias_tag = "#" + re.sub(r"[^A-Za-z0-9]", "", rng.choice(inc["aliases"]))
        chosen = rng.sample(tags, k=rng.choice([1, 1, 2])) + ([alias_tag] if rng.random() < 0.5 else [])
        if rng.random() < 0.4:
            chosen = [c.lower() for c in chosen]
        s = f"{s} {' '.join(chosen)}"
    if platform == "X" and len(s) > 280:
        s = s[:277].rsplit(" ", 1)[0] + "..."
    return s, who

def choose_lang(inc, platform):
    st = STYLE[platform]
    if inc["state"].startswith(TELUGU) and rng.random() < 0.35:
        return "te-en"
    if inc["area"].startswith(HINDI_BELT):
        if rng.random() < (0.10 if platform == "NewsPortal" else 0.04):
            return "hi"
        if rng.random() < st["hinglish"]:
            return "hinglish"
    elif rng.random() < st["hinglish"] * 0.5:   # outside the Hindi belt Hinglish is less common
        return "hinglish"
    return "en"

def posted_at(inc):
    r = rng.random()
    if r < 0.70:   d = nrng.exponential(4)
    elif r < 0.95: d = rng.uniform(7, 60)
    else:          d = rng.uniform(60, 400)
    room = (CUTOFF - inc["date"]).days                # no comment can be dated in the future
    if d > room:
        d = rng.uniform(0, max(room, 0))
    hour = int(nrng.choice(range(24), p=HOUR_P))
    return inc["date"] + timedelta(days=int(d), hours=hour, minutes=rng.randrange(60))

_h = np.array([1, .6, .4, .3, .3, .5, 1, 2, 3, 3.5, 3.5, 3.5, 4, 4, 3.5, 3.5, 4, 4.5, 5, 6, 6.5, 6, 4.5, 2.5])
HOUR_P = _h / _h.sum()


# ---------------------------------------------------------------------------
# 5b. Word-level variety: real people say the same thing in many ways
# ---------------------------------------------------------------------------
SYNONYMS_EN = {
    "students": ["aspirants", "candidates", "kids", "youngsters"], "aspirants": ["students", "candidates"],
    "candidates": ["students", "aspirants"], "angry": ["furious", "upset", "mad"],
    "shameful": ["disgraceful", "embarrassing", "shameless"], "pathetic": ["useless", "hopeless", "terrible"],
    "good": ["nice", "great", "solid"], "great": ["awesome", "excellent", "brilliant"], "happy": ["glad", "pleased"],
    "decision": ["call", "move", "step"], "really": ["truly", "genuinely", "seriously"], "very": ["really", "super", "so"],
    "system": ["setup", "machinery", "process"], "action": ["steps", "measures"], "quickly": ["fast", "swiftly"],
    "honestly": ["tbh", "frankly"], "hard work": ["effort", "mehnat"], "thank you": ["thanks", "thank u", "thanku"],
    "update": ["news", "info"], "announced": ["declared", "released"], "everyone": ["everybody", "all of you"],
    "paper leak case": ["paper leakage case", "leak case"], "another paper leak": ["another paper leakage", "one more leak"], "government": ["govt", "sarkar"], "disappointed": ["let down", "upset"],
    "relieved": ["glad", "so relieved"], "hope": ["hope", "wish"], "trust": ["faith in", "belief in"] ,
    "officials": ["authorities", "officers"], "years": ["years", "yrs"], "please": ["pls", "plz"],
}
SYNONYMS_HI = {
    "bhai": ["bro", "yaar", "bhaiya"], "accha": ["acha", "achha"], "nahi": ["nhi", "nahin"], "bahut": ["bohot", "bhot", "bht"],
    "sharam": ["sharm"], "kya": ["kya", "kyaa"], "hai": ["h", "hai"], "mehnat": ["mehnat", "mehenat"], "kab": ["kb"],
    "paper": ["paper", "pepar"], "kuch": ["kuchh", "kch"], "abhi": ["abi", "abhi"], "log": ["log", "logo"],
}
SYN_RE = {lang: re.compile(r"\b(" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b", re.I)
          for lang, d in (("en", SYNONYMS_EN), ("hinglish", SYNONYMS_HI))}

def paraphrase(s, lang, p=0.35):
    if lang not in SYN_RE:
        return s
    table = SYNONYMS_EN if lang == "en" else SYNONYMS_HI
    def swap(m):
        word = m.group(0)
        if rng.random() > p:
            return word
        new = rng.choice(table[word.lower()])
        return new[:1].upper() + new[1:] if word[:1].isupper() else new
    return SYN_RE[lang].sub(swap, s)

# ---------------------------------------------------------------------------
# 6. Which incidents get how many comments
# ---------------------------------------------------------------------------
def incident_weights(incs, label):
    w = []
    for i in incs:
        size = math.log10(i["asp"]) if i["asp"] else 4.3          # bigger exams -> more comments
        recency = 1 + (i["year"] - 2004) / 8                       # social media grew over time
        reach = 1.8 if i["area"].startswith("All India") else 1.0
        base = (size - 2.5) ** 2 * recency * reach
        t = i["tags"]
        if label == "positive":
            aff = 1 + 0.6 * ("arrests" in t) + 0.5 * ("cancel" in t) + 0.4 * ("retest" in t) + 0.4 * ("denied" in t) - 0.4 * ("noaction" in t)
        elif label == "negative":
            aff = 1 + 0.3 * ("confirmed" in t) + 0.5 * ("noaction" in t) + 0.3 * ("alleged" in t)
        else:
            aff = 1.0
        w.append(base * aff)
    w = np.array(w)
    w = 0.85 * w / w.sum() + 0.15 / len(w)                         # every incident gets a fair floor
    return w / w.sum()


# ---------------------------------------------------------------------------
# 6b. Simulated annotator disagreement (documented in reports/DATASET_CARD.md)
#     Real sentiment labels are not perfectly consistent: people disagree most on
#     short, mixed, sarcastic or question-style comments. A share of labels is
#     moved to a plausible neighbouring class to reflect that.
# ---------------------------------------------------------------------------
NOISE_BASE = 0.05
SARCASM = re.compile(r"👏|🙂|/s\b|\b(great|fantastic|superb|amazing|brilliant|wow|congrats|thank you|thanks|proud|wah|best)\b", re.I)
MIXED = re.compile(r"\b(but|par|though|still)\b", re.I)
FEEL_NEG = re.compile(r"\b(angry|upset|unfair|protesting|worried|gusse|criticised|questioned|oppose)\b", re.I)
FEEL_POS = re.compile(r"\b(relieved|happy|support|khush|defended)\b", re.I)
STRONG = re.compile(r"😡|🤬|💔|💪|🙌|\b(shame|shameful|disgusting|pathetic|disgraceful|well done|respect|hats off|kudos|bakwas|sharam|badhiya|zabardast)\b", re.I)
FLIP = {"negative": ["neutral", "positive"], "positive": ["neutral", "negative"], "neutral": ["negative", "positive"]}

def noise_prob(text, label):
    """Chance that a human annotator would label this comment differently."""
    sarcastic = label == "negative" and SARCASM.search(text)
    p = 0.015 if (STRONG.search(text) and not sarcastic) else NOISE_BASE   # clear emotion -> annotators agree
    if label == "neutral" and not (FEEL_NEG.search(text) or FEEL_POS.search(text)):
        p = 0.01 if "?" in text else 0.02                                # plain info / question -> clearly neutral
    if len(text.split()) <= 5:                                  p += 0.15   # too short to be sure
    if MIXED.search(text):                                      p += 0.15   # mixed feelings
    if "?" in text and label != "neutral":                      p += 0.12   # a question that carries an opinion
    if label == "neutral" and (FEEL_NEG.search(text) or FEEL_POS.search(text)): p += 0.15   # reports others' feelings
    if sarcastic:                                               p += 0.20   # sarcasm is easy to miss
    return min(p, 0.45)

def annotate(label, text):
    if rng.random() >= noise_prob(text, label):
        return label
    if label == "negative" and SARCASM.search(text):
        return "positive" if rng.random() < 0.7 else "neutral"   # missed sarcasm -> read as praise
    if label == "neutral":
        if FEEL_NEG.search(text) and not FEEL_POS.search(text): return "negative"
        if FEEL_POS.search(text) and not FEEL_NEG.search(text): return "positive"
        return rng.choice(["negative", "positive"])
    a, b = FLIP[label]
    return a if rng.random() < 0.8 else b

# ---------------------------------------------------------------------------
# 7. Generate
# ---------------------------------------------------------------------------
def main():
    incs = load_incidents()
    rows, seen = [], set()
    for label in LABELS:
        w = incident_weights(incs, label)
        made, tries = 0, 0
        while made < GEN_PER_CLASS:
            tries += 1
            inc = incs[nrng.choice(len(incs), p=w)]
            ts = posted_at(inc)
            p = P_PLATFORM[label]
            allowed = [pl for pl in PLATFORMS if ts.year >= LAUNCH[pl]]
            probs = np.array([p[PLATFORMS.index(pl)] for pl in allowed]); probs /= probs.sum()
            platform = allowed[nrng.choice(len(allowed), p=probs)]
            lang = choose_lang(inc, platform)
            text, who = build(inc, label, platform, lang)
            key = re.sub(r"[^\w]", "", text.lower())
            if key in seen or len(text) < 8:
                continue
            seen.add(key)
            likes = int(nrng.lognormal({"negative": 2.3, "positive": 1.9, "neutral": 1.2}[label], 1.3))
            rows.append(dict(
                incident_id=inc["incident_id"], exam_name=inc["exam_name"], exam_category=inc["category"],
                state=inc["state"], incident_year=inc["year"], platform=platform,
                posted_at=ts.strftime("%Y-%m-%d %H:%M"), user_type=who, language=lang,
                text=text, generated_label=label, likes=likes,
            ))
            made += 1
        print(f"{label:9s}: {made:,} comments ({tries:,} attempts, {tries - made:,} duplicates rejected)")
    df = pd.DataFrame(rows)
    df["label"] = [annotate(l, t) for l, t in zip(df.generated_label, df.text)]
    # exact balance on the FINAL label
    df = (df.groupby("label", group_keys=False).sample(n=PER_CLASS, random_state=SEED)
            .sample(frac=1, random_state=SEED).reset_index(drop=True))
    df.insert(0, "comment_id", [f"C{i:06d}" for i in range(1, len(df) + 1)])
    flipped = df[df.label != df.generated_label]
    NOISE_LOG.parent.mkdir(parents=True, exist_ok=True)
    flipped[["comment_id", "generated_label", "label", "text"]].to_csv(NOISE_LOG, index=False, encoding="utf-8-sig")
    print(f"Simulated annotator disagreement: {len(flipped):,} labels ({len(flipped)/len(df):.1%}) -> {NOISE_LOG.name}")
    print(pd.crosstab(df.generated_label, df.label, rownames=["generated"], colnames=["final"]))
    df = df.drop(columns="generated_label")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")   # BOM so Excel shows Hindi / emojis correctly
    print(f"\nSaved {len(df):,} rows -> {OUT}")

if __name__ == "__main__":
    main()
