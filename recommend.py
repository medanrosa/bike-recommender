import pandas as pd

def filter_by_rules(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    """
    Hard filters before model scoring:
      - Beginner safety cap (<=500cc)
      - Geography -> allowed riding styles (no user-chosen style)
      - Budget cap
    """
    df = df.copy()

    # Beginner → ≤500cc
    if user["experience_level"] == 0:
        df = df[df.engine_size <= 500]

    # Geography -> allowed style sets
    # geo codes: 0=mountain, 1=coastal, 2=city, 3=small_town, 4=highway, 5=back_roads
    allowed_styles_by_geo = {
        4: ["supersport", "touring", "naked"],                 # highway
        0: ["dirtbike", "supermoto", "touring", "naked"],      # mountain
        5: ["dirtbike", "supermoto", "touring"],               # back roads / forest
        2: ["naked", "supermoto", "commuter"],                 # city
        3: ["naked", "supermoto", "dirtbike", "commuter"],     # small town / village
        1: ["touring", "naked", "supersport"],                 # coastal
    }
    geo = user["geography"]
    allowed = allowed_styles_by_geo.get(geo, None)
    if allowed:
        df = df[df.riding_style.isin(allowed)]

    # Budget cap
    df = df[df.price <= user["budget"]]
    return df