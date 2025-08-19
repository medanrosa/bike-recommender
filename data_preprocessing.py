import pandas as pd
import numpy as np

# ---- Canonical mapping (codes are consistent across the app) ----
STYLE_TO_CODE = {
    "supersport": 0,
    "supermoto":  1,
    "naked":      2,
    "touring":    3,
    "dirtbike":   4,
    "autocycle":  5,
    "commuter":   6,
    "cruiser":    7,
    "adventure":  8,
}

def load_data(path: str) -> pd.DataFrame:
    """Load the bike dataset from Excel."""
    return pd.read_excel(path)

def _normalize_style_value(s: str) -> str:
    """
    Normalize dataset labels into the canonical set used by the app.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().lower()

    # synonyms → canonical
    if s in {"street", "standard", "roadster"}:
        return "naked"
    if s in {"dirt"}:
        return "dirtbike"
    if s in {"adv", "adventure touring"}:
        return "adventure"
    if s in {"sport", "sportbike"}:
        return "supersport"

    return s

def _infer_style_from_name(name: str) -> str | None:
    """
    Lightweight heuristic if a row is missing/unknown.
    """
    m = (name or "").lower()

    if "supermoto" in m:
        return "supermoto"
    if "adventure" in m or " adv" in m:
        return "adventure"
    if any(k in m for k in ["rebel","vulcan","intruder","virago","shadow","bolt","sportster","fat boy","heritage","meteor","classic 350","thunderbird"]):
        return "cruiser"
    if any(k in m for k in ["tour","gold wing","spyder rt","road king"," rt"]):
        return "touring"
    if any(k in m for k in ["crf","wr","rm","exc","husqvarna","sx ","kx "," dirt"," enduro"]):
        return "dirtbike"
    if any(k in m for k in ["r1","r6","cbr","gsx-r","ninja","yzf","panigale","rc","rs","rr","zx-","s1000rr"]):
        return "supersport"
    if any(k in m for k in ["mt-","street","fz","sv ","monster","scr","duke","z650","z900","cb300","cb500","cb650","hornet"]):
        return "naked"
    return None

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normalize riding_style to the canonical set.
    - Overwrite riding_style_code consistently (ignore any existing codes).
    - Ensure engine_size and price are ints.
    """
    df = df.copy()

    # Normalize style values
    df["riding_style"] = df["riding_style"].map(_normalize_style_value)

    # Infer if still missing/empty
    df["riding_style"] = df.apply(
        lambda r: r["riding_style"] or _infer_style_from_name(r.get("bike_name")),
        axis=1
    )

    # Drop rows missing essentials
    df = df.dropna(subset=["engine_size", "price", "riding_style"]).copy()

    # Types
    df["engine_size"] = df["engine_size"].astype(int)
    df["price"]       = df["price"].astype(int)

    # Canonical codes
    df["riding_style"] = df["riding_style"].str.lower().str.strip()
    df["riding_style_code"] = df["riding_style"].map(STYLE_TO_CODE)

    # Drop rows with unknown style
    df = df.dropna(subset=["riding_style_code"]).copy()
    df["riding_style_code"] = df["riding_style_code"].astype(int)

    return df