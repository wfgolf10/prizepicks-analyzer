def run_pipeline(hist_df, stats=["passing_yards","rushing_yards","receiving_yards"]):
    base, frames = build_features(hist_df, stats)
    models = {}
    for stat in stats:
        if stat not in frames:
            continue
        train_df = frames[stat].merge(base, on=["player", "date"], how="left")
        model, resid_std = train_model_per_stat(train_df, stat)
        models[stat] = (model, resid_std)

    pp = fetch_prizepicks_projections()

    latest_by_player = {}
    for stat in stats:
        if stat not in frames:
            continue
        g = frames[stat]
        latest = g.sort_values(["player", "date"]).groupby("player").tail(1)
        latest_by_player[stat] = latest

    rows = []
    for _, row in pp.iterrows():
        stat_key = row["stat_key"]
        if stat_key not in models:
            continue
        model, resid_std = models[stat_key]
        latest = latest_by_player[stat_key]
        pl = latest[latest["player"] == row["player"]]
        if pl.empty:
            continue
        feature_cols = [c for c in pl.columns if c.startswith(stat_key + "_") or c == "games_played"]
        X_inf = pl[feature_cols].fillna(0.0)
        pred = float(model.predict(X_inf)[0])
        prob_over, edge = compute_edge(pred, float(row["line"]), resid_std, "more")
        rows.append({
            "player": row["player"],
            "label": row["label"],
            "line": float(row["line"]),
            "prediction": pred,
            "edge": edge,
            "prob_more": prob_over,
        })

    return pd.DataFrame(rows).sort_values("edge", ascending=False)
