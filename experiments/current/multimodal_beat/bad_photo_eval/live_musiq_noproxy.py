"""Live MUSIQ test — NO PROXY. curl_cffi chrome impersonation, paced, backs off on block.

Scrapes newest pages of griffati uomo+donna via a plain browser-TLS session (no
brightdata / datacenter proxy), scores each first image with MUSIQ, flags bad photos.
Stops a search on soft-block instead of hammering. Run from the scripts/ dir on sys.path.

  cd scripts && python .../live_musiq_noproxy.py     (or it inserts SCRIPTS itself)
"""
import sys, time, io, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

SCRIPTS = "/home/ale/Desktop/vinted/Vinted_New_Version/scripts"
BPE = "/home/ale/Desktop/vinted/Vinted_New_Version/experiments/current/multimodal_beat/bad_photo_eval"
sys.path.insert(0, SCRIPTS)

from curl_cffi import requests as cffi
import requests_html
import pandas as pd, numpy as np
from PIL import Image
import pyiqa
from sklearn.metrics import roc_curve

from config.project_config import settings
from config.search_loader import load_searches
from simple_scraper import Simple_scraper

SEARCHES = ["griffati_uomo_all", "griffati_donna_all"]
PAGES = 10
PRODUCT_SEL = ".new-item-box__container"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.vinted.it/", "Connection": "keep-alive",
}
GAP = 60.0
BACKOFF = 90.0
IMGDIR = Path(BPE) / "data" / "live_imgs"
IMGDIR.mkdir(parents=True, exist_ok=True)


def preflight():
    """Return True if the plain IP can reach Vinted (not DataDome-blocked)."""
    try:
        r = cffi.Session(impersonate="chrome").get("https://www.vinted.it/", timeout=20)
        return r.status_code == 200
    except Exception:
        return False


def musiq_threshold():
    t = pd.read_csv(f"{BPE}/data/scored_with_musiq.csv", low_memory=False)
    t["item_id"] = t.item_id.astype(str)
    lab = []
    for d, pv, lb in [("holdout", "holdout_candidates_private.csv", "holdout_label_sheet_human.csv"),
                      ("musiq_batch", "musiq_candidates_private.csv", "musiq_label_sheet_human.csv")]:
        p = pd.read_csv(f"{BPE}/data/{d}/{pv}"); p["item_id"] = p.item_id.astype(str)
        m = p.merge(pd.read_csv(f"{BPE}/data/{d}/{lb}"), on="blind_id")
        m["tq"] = m.technical_quality.astype(str).str.lower().str.strip()
        lab.append(m[["item_id", "tq"]])
    ar = pd.read_csv(f"{BPE}/data/evaluation/chatgpt_results/human_audit_queue_reviewed.csv")
    ar["item_id"] = ar.item_id.astype(str)
    lab.append(ar.rename(columns={"human_technical_quality": "tq"})[["item_id", "tq"]].assign(
        tq=lambda x: x.tq.str.lower().str.strip()))
    L = pd.concat(lab).drop_duplicates("item_id")
    L = L[L.tq.isin(["bad", "good"])].merge(t[["item_id", "musiq"]], on="item_id").dropna()
    y = (L.tq == "bad").astype(int).values
    fpr, tpr, thr = roc_curve(y, -L.musiq.values)
    return float(-thr[int(np.argmax(tpr - fpr))]), len(L), int(y.sum())


def scrape():
    scraper = Simple_scraper()
    searches = load_searches(str(settings.paths.searches_yaml))
    session = cffi.Session(impersonate="chrome")
    rows = []
    for name in SEARCHES:
        s = searches[name]
        try: s.sort = "newest_first"
        except Exception: pass
        base = scraper.create_webpage(s)
        got = 0
        for pg in range(1, PAGES + 1):
            url = f"{base}&page={pg}"
            r = session.get(url, headers=HEADERS, timeout=30)
            prods = requests_html.HTML(html=r.text).find(PRODUCT_SEL)
            if r.status_code in (403, 429) or len(prods) == 0:
                print(f"  [{name}] page {pg}: SOFT-BLOCK (status={r.status_code}, "
                      f"items={len(prods)}) -> backoff {BACKOFF:.0f}s + 1 retry", flush=True)
                time.sleep(BACKOFF)
                r = session.get(url, headers=HEADERS, timeout=30)
                prods = requests_html.HTML(html=r.text).find(PRODUCT_SEL)
                if r.status_code in (403, 429) or len(prods) == 0:
                    print(f"  [{name}] still blocked at page {pg}; stopping ({got} listings).",
                          flush=True)
                    break
            n = 0
            for p in prods:
                row = scraper.extract_catalog_item_meta(p, {}, pg - 1, 0, get_images=True)
                if row:
                    row["SearchName"] = name; rows.append(row); n += 1
            got += n
            print(f"  [{name}] page {pg}: {n} listings (total {got})", flush=True)
            time.sleep(GAP)
        print(f"[{name}] done: {got} listings", flush=True)
    return pd.DataFrame(rows)


def download(session, url, iid):
    try:
        b = session.get(url, headers=HEADERS, timeout=20).content
        im = Image.open(io.BytesIO(b)).convert("RGB")
        p = IMGDIR / f"{iid}.jpg"; im.save(p, "JPEG", quality=90)
        return str(p)
    except Exception:
        return None


def main():
    if not preflight():
        print("BLOCKED: Vinted homepage != 200 on this IP (DataDome). No proxy allowed; "
              "wait for the IP window to clear and re-run. Nothing scraped.")
        return
    cut, n, nb = musiq_threshold()
    print(f"[threshold] MUSIQ < {cut:.1f} = bad (from {n} labels, {nb} bad)\n")
    df = scrape()
    if df.empty:
        print("NO LISTINGS scraped."); return
    df = df.drop_duplicates("Dataid")
    df = df[df["Images"].notna() & (df["Images"].astype(str) != "")].copy()
    print(f"\n[images] downloading {len(df)} first images (no proxy)...", flush=True)
    session = cffi.Session(impersonate="chrome")
    df["local"] = [download(session, u, i) for u, i in zip(df["Images"], df["Dataid"])]
    df = df[df["local"].notna()].copy()
    print(f"[musiq] scoring {len(df)} images...", flush=True)
    metric = pyiqa.create_metric("musiq", device="cpu")
    df["musiq"] = [float(metric(p)) if p else np.nan for p in df["local"]]
    df = df.dropna(subset=["musiq"])
    df["musiq_bad"] = df["musiq"] < cut
    df = df.sort_values("musiq")
    df[["SearchName", "Dataid", "Title", "Link", "musiq", "musiq_bad"]].to_csv(
        f"{BPE}/data/live_musiq_test.csv", index=False)
    bad = df[df.musiq_bad]
    print(f"\n=== scored {len(df)} | flagged BAD {len(bad)} ({len(bad)/max(len(df),1)*100:.0f}%) "
          f"| threshold MUSIQ<{cut:.1f} ===\n")
    for _, r in df.head(25).iterrows():
        print(f"  {'BAD ' if r.musiq_bad else '    '} {r.musiq:5.1f}  {r.SearchName[:15]:15s} "
              f"{str(r.Title)[:38]:38s} https://www.vinted.it{r.Link}")
    print(f"\nfull -> {BPE}/data/live_musiq_test.csv")


if __name__ == "__main__":
    main()
