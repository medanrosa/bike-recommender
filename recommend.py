import pandas as pd

def filter_by_rules(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    """
    Apply hard rules before model scoring:
      - Experience caps:
          beginner  (0) → ≤ 500cc
          intermediate (1) → ≤ 800cc
          expert (2) → no cap
      - Geography → allowed riding styles:
          0 mountain   → supermoto, dirtbike
          1 coastal    → supermoto, dirtbike
          2 city       → naked
          3 small_town → naked
          4 highway    → supersport, cruiser
          5 back_roads → supermoto, dirtbike, adventure
      - Budget cap
    """
    df = df.copy()

    # --- Engine caps by experience ---
    level = int(user["experience_level"])
    if level == 0:        # beginner
        df = df[df.engine_size <= 500]
    elif level == 1:      # intermediate
        df = df[df.engine_size <= 800]
    # expert => no cap

    # --- Geography style gates ---
    # 0=mountain, 1=coastal, 2=city, 3=small_town, 4=highway, 5=back_roads
    geo = int(user["geography"])
    allowed_styles_by_geo = {
        4: ["supersport", "cruiser"],                 # highway
        2: ["naked"],                                 # city
        3: ["naked"],                                 # small town / village
        5: ["supermoto", "dirtbike", "adventure"],    # back roads / forest / dirt
        0: ["supermoto", "dirtbike"],                 # mountain
        1: ["supermoto", "dirtbike"],                 # coastal
    }
    allowed = allowed_styles_by_geo.get(geo)
    if allowed:
        df = df[df.riding_style.isin(allowed)]

    # --- Budget cap (final hard filter) ---
    df = df[df.price <= int(user["budget"])]

    return df