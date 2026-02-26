def process_df(trans: pl.DataFrame, batch_size: int, agg_col: str) -> dict:
    application_ids = trans["unique_app_id"].unique().to_list()
    num_batches = math.ceil(len(application_ids) / batch_size)
    objects = {}

    unique_mcc_set = set(UNIQUE_MCC_GROUPS)
    zero_cache = {}  # cache zero arrays by length

    for i in tqdm(range(num_batches), desc="processing batches"):
        app_id_batch = application_ids[i * batch_size : (i + 1) * batch_size]
        batch_set = set(app_id_batch)

        feats = get_seq_features(
            trans.filter(pl.col("unique_app_id").is_in(app_id_batch)), agg_col
        )

        # Single partition_by replaces N individual filters
        grouped = feats.sort(agg_col, descending=True).partition_by(
            "unique_app_id", as_dict=True, include_key=False
        )

        for app_id, app_feats in grouped.items():
            # Pivot once per app — no redundant filter/sort
            app_data = (
                app_feats.pivot(on=agg_col, index="mcc_group_name", values="sum")
                .fill_null(0)
            )

            mcc_groups = app_data["mcc_group_name"].to_list()
            value_cols = app_data.columns[1:]
            history_len = len(value_cols)
            days_before_app_arr = np.array(value_cols, dtype=int)

            # Extract numeric matrix in one shot instead of transpose gymnastics
            mat = app_data.select(value_cols).to_numpy().astype(int)  # (n_mcc, n_days)

            result = {
                mcc_groups[j]: mat[j] for j in range(len(mcc_groups))
            }
            result[agg_col] = days_before_app_arr

            # Add missing MCC groups with shared zero arrays
            missing = unique_mcc_set - set(mcc_groups)
            if missing:
                if history_len not in zero_cache:
                    zero_cache[history_len] = np.zeros(history_len, dtype=int)
                zeros = zero_cache[history_len]
                for mcc_group in missing:
                    result[mcc_group] = zeros

            parts = app_id.split("_", 1)
            objects[app_id] = {
                "feature_arrays": result,
                "APPLICATION_ID": parts[1],
                "APPLICATION_DATE": parts[0],
                "SEQ_LENGTH": history_len,
            }

    return objects
