"""
AI201 Project 3: TakeMeter — Apify Reddit CSV Labeling Script
Reads raw Apify scraper CSV, cleans, labels, and writes data/takemeter_dataset.csv
"""

import re
import pandas as pd
from pathlib import Path

INPUT_PATH = "data/raw_reddit.csv"
OUTPUT_PATH = "data/takemeter_dataset.csv"

# ─── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows | columns: {list(df.columns)}")

# ─── 2. Build combined text field ─────────────────────────────────────────────
def build_text(row):
    title = str(row.get("title", "") or "").strip()
    body  = str(row.get("body",  "") or "").strip()
    # Comment rows: title is "/u/username on <post title>" — drop it
    if title.startswith("/u/"):
        return body
    # Post rows: title + body
    if body and body.lower() not in ("nan", ""):
        return (title + " " + body).strip() if title else body
    return title

df["raw_text"] = df.apply(build_text, axis=1)

# ─── 3. Clean text ─────────────────────────────────────────────────────────────
SCRAPER_PATS = [
    r"submitted by\s+/u/\S+",
    r"\[link\]",
    r"\[comments\]",
    r"&#\d+;",
    r"&amp;|&lt;|&gt;|&quot;|&#39;",
    r"https?://\S+",
    r"/u/\S+",
    r"/r/\S+",
]

def clean_text(text):
    t = str(text)
    for p in SCRAPER_PATS:
        t = re.sub(p, " ", t, flags=re.IGNORECASE)
    t = t.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s{2,}", " ", t).strip()

df["text"] = df["raw_text"].apply(clean_text)

# ─── 4. Filter rows ────────────────────────────────────────────────────────────
REMOVED_PATTERNS = {"[ removed by reddit ]","[removed]","[deleted]","removed","deleted","[ removed ]"}

skip_log = []

def is_bad(text):
    return text.lower().strip() in REMOVED_PATTERNS or len(text.strip()) < 40

df["skip_reason"] = ""
df.loc[df["text"].apply(is_bad), "skip_reason"] = "removed_or_too_short"
df.loc[df.duplicated(subset="text", keep="first") & (df["skip_reason"]==""), "skip_reason"] = "duplicate"

skipped = df[df["skip_reason"] != ""]
usable  = df[df["skip_reason"] == ""].copy()
print(f"Skipped (removed/short/dup): {len(skipped)}  |  Usable: {len(usable)}")

# ─── 5. Rule-based labeling ───────────────────────────────────────────────────

# --- Employee signals (worker perspective) ---
EMP_PATS = [
    r"\b(my (store|manager|dm|rsm|ram|rsa|coach|territory manager|district manager))\b",
    r"\b(commission|quota|kpi|attainment|scorecard|coaching session|performance plan)\b",
    r"\b(we (have to|need to|get penalized|get dinged|are told|were told|can't))\b",
    r"\b(our (store|team|customers|manager|reps))\b",
    r"\b(frontline|floor staff|retail employee|store employee|tmo employee)\b",
    r"\b(clocking|scheduling|shift|paycheck|wage|hourly pay)\b",
    r"\b(my reps?|my team|my district|my coach|my manager)\b",
    r"\b(cage connect|t-life (check.?in|scan))\b",
    r"\bping.?pong(ing)?\b",
    r"\b(store experience|in-store|walk.?in customer)\b",
    r"\b(customers (coming in|come in|walk in)|walk.?in(s)?)\b",
    r"\b(i (work|worked|am working|was working) (at|for|in) (t.?mobile|tmobile|the store))\b",
    r"\b(as (an|a) (employee|rep|rsa|expert|manager))\b",
    r"\b(net negative|negative to (an|the) employee)\b",
    r"\b(they preach|corporate (says|tells|wants|pushes|policy))\b",
    r"\b(sell (tablets|accessories|lines|protection)|selling points?)\b",
    r"\bkpis?\b",
    r"\b(accountability|write.?up|counseled|written up)\b",
    r"\b(customers? (waste|wasting) (my|our) time)\b",
    r"\b(hit (my|our|the) (quota|goal|numbers|kpi|metrics))\b",
]

# --- Help request signals ---
HELP_PATS = [
    r"\b(should i|can i|is it worth|how do i|how can i|what (is|are) the best)\b",
    r"\b(advice|recommend(ation)?|suggestion|help me|confused|not sure|wondering)\b",
    r"\b(looking for (advice|help|info|information|recommendations))\b",
    r"\b(need (help|advice|clarification|guidance)|need to know)\b",
    r"\b(does anyone|has anyone|any ideas|any advice|anyone know)\b",
    r"\b(troubleshoot|fix (this|my|the)|get this (working|resolved))\b",
    r"\b(what (should|would|do) (i|you)|help\b)",
    r"\b(can someone|someone help|please help)\b",
    r"\b(question about|questions? (about|on|regarding))\b",
    r"\b(not sure (what|how|if|whether)|unsure (about|how|if))\b",
    r"\bhelp\b",
    r"\?",   # any question mark
    r"\b(trying to (figure|understand|find out|determine|fix|troubleshoot))\b",
    r"\b(first time|new (to|customer)|just (joined|switched|got))\b",
    r"\b(at the end of my rope|i don'?t know (what|how))\b",
    r"\b(a few question|couple of question|have (a|some) question)\b",
]

# --- Deal / policy signals ---
DEAL_PATS = [
    r"\b(trade.?in|bill credit|promo(tion)?|promotional|discount|deal|offer|rebate)\b",
    r"\b(plan (includes?|has|costs?|details?|is)|essentials|magenta|go5g|simple choice|sero|business plan)\b",
    r"\b(upgrade|device payment|ipp|eip|installment plan|financing)\b",
    r"\b(eligible|eligibility|qualify|qualifies|terms (and conditions)?|fine print)\b",
    r"\b(per (month|line)|monthly (cost|fee|charge)|per month)\b",
    r"\b(perk|perks|hulu|netflix|disney|apple tv|streaming include)\b",
    r"\b(network priority|deprioritize|throttle|data cap|unlimited (data|plan))\b",
    r"\b(policy (change|update|is|rule)|account (rule|terms|hold|restriction))\b",
    r"\b(bogo|buy one get one|free line|4.?for.?100|3 free|on us)\b",
    r"\b(promo code|promo (id|expires?|is valid|ends))\b",
    r"\b(credit (check|score)|postpaid|prepaid|mvno)\b",
    r"\b(port(ing)? (over|in|out)|keep and switch|family freedom)\b",
    r"\b(unlock(ed)?|sim.?lock|imei)\b",
    r"\b(autopay|auto.?pay discount)\b",
    r"\b(taxes (and fees|included)|tax inclusive)\b",
    r"\b(magenta max|go5g plus|essentials (plan|is))\b",
    r"\b(third line free|3rd line free|free (line|device|phone))\b",
    r"\b(12.?month|24.?month|36.?month|installment)\b",
    r"\b(\$\d+\s*(off|credit|discount|per month|a month))\b",
]

# --- Customer complaint signals ---
COMP_PATS = [
    r"\b(worst|terrible|horrible|awful|garbage|trash|fraud|lied|lies|dishonest)\b",
    r"\b(never again|cancel(led)?|port.?out|leaving t.?mobile|switching (away|to (verizon|att|cricket)))\b",
    r"\b(ripped off|overcharged|charged me (wrong|extra|twice|incorrectly))\b",
    r"\b(billing (error|issue|problem|mess|nightmare)|wrong (charge|amount|bill))\b",
    r"\b(coverage (is|was|sucks)|dead zone|dropped calls|no signal|poor (service|coverage|reception))\b",
    r"\b(customer (service|support|care) (is|was|sucks|terrible|awful|horrible|unhelpful))\b",
    r"\b(waited (forever|hours|too long)|long (wait|hold|line))\b",
    r"\b(they (lied|cheated|stole|took|screwed|played me|misled)|t.?mobile (lied|screwed|misled))\b",
    r"\b(mistake to (switch|join)|regret (switching|joining)|wish i (had|never))\b",
    r"\b(done with t.?mobile|frustrated with t.?mobile|fed up with)\b",
    r"\b(loyal (customer|for \d+ years)|20\+ years|15 years|decade)\b",
    r"\b(not quoted|wasn't told|never disclosed|hidden (fee|charge|cost))\b",
    r"\b(scam|deceptive|misleading|false (advertising|promise))\b",
    r"\b(store rep (lied|misled|told me wrong)|rep (lied|told me)|was told incorrectly)\b",
    r"\b(going to (switch|cancel|leave|file)|considering (switching|leaving))\b",
    r"\b(lost (my|all) (photos?|data|contacts|messages))\b",
    r"\b(t.?mobile doesn't care|t.?mobile (won't|refuses to|can't) help)\b",
]

def score(text, patterns):
    t = text.lower()
    return sum(1 for p in patterns if re.search(p, t))

def label_row(text):
    t = text.lower()

    emp  = score(t, EMP_PATS)
    hlp  = score(t, HELP_PATS)
    deal = score(t, DEAL_PATS)
    comp = score(t, COMP_PATS)

    # Rule 3: strong employee perspective
    if emp >= 2:
        return "employee_complaint", emp, hlp, deal, comp

    # Rule 1: help request (asking for advice/fix)
    # Require question mark OR two explicit help signals
    has_q = "?" in text
    if (has_q and hlp >= 1) or hlp >= 3:
        return "help_request", emp, hlp, deal, comp

    # Rule 4: deal/policy info dominates
    if deal >= 2 and deal > max(comp, emp):
        return "deal_or_policy_info", emp, hlp, deal, comp

    # Rule 2: customer complaint
    if comp >= 2:
        return "customer_complaint", emp, hlp, deal, comp

    # Soft tie-break: any single signal
    scores = {"employee_complaint": emp, "help_request": hlp,
              "deal_or_policy_info": deal, "customer_complaint": comp}
    winner = max(scores, key=scores.get)
    if scores[winner] >= 1:
        return winner, emp, hlp, deal, comp

    return None, emp, hlp, deal, comp

results = usable["text"].apply(label_row)
usable["label"]      = results.apply(lambda x: x[0])
usable["_emp_score"] = results.apply(lambda x: x[1])
usable["_hlp_score"] = results.apply(lambda x: x[2])
usable["_deal_score"]= results.apply(lambda x: x[3])
usable["_comp_score"]= results.apply(lambda x: x[4])

no_signal = usable[usable["label"].isna()]
usable = usable[usable["label"].notna()].copy()
print(f"No-signal (unlabelable): {len(no_signal)}")

# ─── 6. Balance: cap any label at 70% ─────────────────────────────────────────
def cap_label(df_in, label, max_frac):
    cap = max(1, int(len(df_in) * max_frac))
    mask = df_in["label"] == label
    if mask.sum() > cap:
        over_idx = df_in[mask].index[cap:]
        return df_in.drop(index=over_idx)
    return df_in

for lbl in ["help_request","deal_or_policy_info","customer_complaint","employee_complaint"]:
    usable = cap_label(usable, lbl, 0.70)

print(f"After balance cap: {len(usable)} rows")
print("Distribution after cap:", usable["label"].value_counts().to_dict())

# ─── 7. Add notes for borderline cases ────────────────────────────────────────
def make_note(row):
    scores = {
        "employee_complaint":  row["_emp_score"],
        "help_request":        row["_hlp_score"],
        "deal_or_policy_info": row["_deal_score"],
        "customer_complaint":  row["_comp_score"],
    }
    label = row["label"]
    sorted_vals = sorted(scores.values(), reverse=True)
    if len(sorted_vals) >= 2 and sorted_vals[0] - sorted_vals[1] <= 1 and sorted_vals[0] > 0:
        runner_up = [k for k, v in scores.items() if v == sorted_vals[1] and k != label]
        if runner_up:
            return f"borderline: also considered {runner_up[0]}"
    return ""

usable["notes"] = usable.apply(make_note, axis=1)

# ─── 8. Final output ──────────────────────────────────────────────────────────
Path("data").mkdir(exist_ok=True)
final = usable[["text", "label", "notes"]].reset_index(drop=True)
final.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅  Saved {len(final)} rows → {OUTPUT_PATH}")

# ─── 9. Report ────────────────────────────────────────────────────────────────
print("\n=== LABEL DISTRIBUTION ===")
dist = final["label"].value_counts()
for lbl, cnt in dist.items():
    pct = cnt / len(final) * 100
    print(f"  {lbl:<25} {cnt:>4}  ({pct:.1f}%)")

print(f"\n=== TOTAL USABLE EXAMPLES: {len(final)} ===")

print("\n=== SKIPPED ROWS SUMMARY ===")
sr = pd.concat([
    skipped[["text","skip_reason"]],
    no_signal[["text"]].assign(skip_reason="no_signal")
])
print(sr["skip_reason"].value_counts().to_string())
print(f"Total skipped: {len(sr)}")

print("\n=== 3 DIFFICULT / BORDERLINE EXAMPLES ===")
bdf = final[final["notes"].str.startswith("borderline")].head(3)
for i, (_, row) in enumerate(bdf.iterrows(), 1):
    print(f"\n[{i}] FINAL LABEL: {row['label']}")
    print(f"    Note: {row['notes']}")
    print(f"    Text: {row['text'][:250]}")