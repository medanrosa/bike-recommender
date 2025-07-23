import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load the bike dataset from Excel."""
    return pd.read_excel(path)

def impute_missing_riding_style(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Manual lookup for any known missing entries.
    2) Fallback heuristic based on bike_name keywords.
    """
    df = df.copy()
    lookup = {
        "2021 Slingshot SL Autodrive": "autocycle",
        "2019 Slingshot SL":              "autocycle",
        # …add any others you’ve researched…
    }
    def infer_style(name):
        m = str(name).lower()
        if any(k in m for k in ["r1","r6","cbr","gsx-r","ninja","yzf","panigale","rc","rs","rr"]):
            return "supersport"
        if "supermoto" in m:
            return "supermoto"
        if any(k in m for k in ["tour","wing","spyder","rt","road king","gold"]):
            return "touring"
        if any(k in m for k in ["crf","wr","rm","ktm","exc","husqvarna","dirt","sx","kx"]):
            return "dirtbike"
        if any(k in m for k in ["mt","street","fz","sv","monster","scr","duke","gt"]):
            return "naked"
        return "naked"

    # 1) apply manual lookup where riding_style is missing or empty
    df["riding_style"] = df.apply(
        lambda r: lookup.get(r["bike_name"], r["riding_style"])
        if pd.isna(r["riding_style"]) or r["riding_style"] == "" else r["riding_style"],
        axis=1
    )
    # 2) heuristic fill for any still-missing
    df["riding_style"] = df.apply(
        lambda r: infer_style(r["bike_name"])
        if pd.isna(r["riding_style"]) or r["riding_style"] == "" else r["riding_style"],
        axis=1
    )
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1) fill riding_style
    df = impute_missing_riding_style(df)
    # 2) drop any rows missing core specs
    df = df.dropna(subset=["engine_size","price","riding_style"]).copy()
    # 3) enforce int types
    df["engine_size"] = df["engine_size"].astype(int)
    df["price"]       = df["price"].astype(int)
    # 4) encode riding_style
    style_map = {
        "supersport": 0, "supermoto": 1, "naked": 2,
        "touring":    3, "dirtbike": 4,   "autocycle": 5
    }
    df["riding_style_code"] = df["riding_style"].map(style_map)
    return df