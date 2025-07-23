def filter_by_rules(df, user):
    # beginner → ≤500cc
    if user["experience_level"] == 0:
        df = df[df.engine_size <= 500]

    # geography rules
    geo = user["geography"]
    if   geo == 4:              # highway
        df = df[~df.riding_style.isin(["dirtbike","supermoto"])]
    elif geo in [0,5]:          # mountain or back_roads
        df = df[df.riding_style.isin(["dirtbike","touring","supermoto"])]
    elif geo == 2:              # city
        df = df[df.riding_style.isin(["naked","supermoto"])]
    # preferred style
    df = df[df.riding_style_code == user["preferred_style"]]

    # budget cap
    df = df[df.price <= user["budget"]]
    return df