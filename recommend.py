import pandas as pd

def filter_by_rules(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    """
    Hard rules:
      - Experience caps: beginner ≤500cc, intermediate ≤800cc, expert = no cap
      - Geography → allowed riding styles
      - Budget cap
    """
    df = df.copy()

    # Experience caps
    level = int(user["experience_level"])
    if level == 0:
        df = df[df.engine_size <= 500]
    elif level == 1:
        df = df[df.engine_size <= 800]

    # Geo codes: 0=mountain,1=coastal,2=city,3=small_town,4=highway,5=back_roads
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

    # Budget
    df = df[df.price <= int(user["budget"])]
    return df