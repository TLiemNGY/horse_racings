def feature_engineering(df):
    # External conditions
    df['market_expectation_odds'] = df['win_odds'] + df['place_odds']
    df['track_conditions'] = df['going'] + df['surface'] + df['config']
    df['track_depth'] = df['distance'] + df['going'] + df['surface']
    df['weather'] = df['mean_temp'] + df['going']
    df['enjeu'] = df['race_class'] + df['prize']

    # Horse characteristics
    df['age_rating'] = df['horse_rating'] + df['horse_age']
    df['horse_type_per_country'] = df['horse_country'] + df['horse_type']
    df['country_powered'] = df['horse_country'] + df['horse_type'] + df['horse_rating']
    df['global_weight'] = df['actual_weight'] - df['declared_weight']

    # Jockey characteristics : réintégrer dans les modèles qui acceptent les catégories larges
    #df['worker_synergy'] = df['win_odds'] + df['trainer_id'] + df['jockey_id']
    #df['specialized_trainer'] = df['trainer_id'] + df['jockey_id'] + df['horse_rating']

    # Combined
    df['distance_vs_race_class'] = df['distance'] + df['race_class']
    df['distance_vs_age'] = df['distance'] + df['horse_age']
    df['distance_vs_weight'] = df['distance'] + df['actual_weight']

    return df

