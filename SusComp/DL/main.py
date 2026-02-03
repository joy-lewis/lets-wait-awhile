from dataclasses import dataclass
import train
import models
import torch
import pandas as pd
from xai import run_xai_permutation


def sanity_check_timeseries_df(df: pd.DataFrame,feature_cols: list[str], target_col: str, expected_freq: str = "H",):
    # Index checks
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not sorted (not monotonic increasing). Call df.sort_index().")

    dup_count = int(df.index.duplicated().sum())
    if dup_count > 0:
        examples = df.index[df.index.duplicated()].unique()[:5]
        raise ValueError(f"Found {dup_count} duplicate timestamps in index. Examples: {list(examples)}")

    # Frequency / gaps check (optional but useful for hourly)
    inferred = pd.infer_freq(df.index)
    if inferred is None:
        # fall back to median step
        med = df.index.to_series().diff().dropna().median()
        raise ValueError(
            f"Could not infer a fixed frequency (gaps/irregular steps). "
            f"Median step is {med}. Expected {expected_freq}."
        )
    if expected_freq and inferred != expected_freq:
        raise ValueError(f"Unexpected inferred frequency: {inferred}. Expected {expected_freq}.")

    # Column checks
    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # NaN checks
    na_counts = df[required].isna().sum()
    bad = na_counts[na_counts > 0]
    if len(bad) > 0:
        # show a few columns with NaNs
        top = bad.sort_values(ascending=False).head(10)
        print(f"NaNs found in required columns:\n{top.to_string()}")

        # Drop NaNs for simplicity
        df.dropna(subset=required, inplace=True)
        print(f"Dropped rows with NaNs in required columns. New row count: {len(df)}")

    # Quick info
    print(
        f"✅ sanity_check passed | rows={len(df)} | "
        f"range={df.index.min()} → {df.index.max()} | freq={inferred}"
    )
    return df



@dataclass
class LSTMForecastConfig:
    # Data
    hist_input_size: int # regular historical features
    fut_input_size: int  # forecast weather features
    horizon: int = 24               # forecast length (24 hours)
    # LSTM
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1            # effective only if num_layers > 1
    bidirectional: bool = False
    # Head
    head_hidden_size: int = 256
    head_dropout: float = 0.1
    # Misc
    use_layernorm: bool = False     # optional stabilization


def test():
    """
    Model test with random data
    """
    cfg = LSTMForecastConfig(
            hist_input_size = 0,
            fut_input_size  = 0,  # if add_time_features_to_future=True,  # e.g., 10 features per hour
            horizon=24,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            head_hidden_size=128,
            head_dropout=0.1,
            use_layernorm=True,
    )

    model = models.LSTMMultiHorizon(cfg)
    x = torch.randn(32, 168, 10)  # batch=32, lookback=168 hours, features=10
    y = model(x)
    print(y.shape)  # torch.Size([32, 24])

def main():
    # Setting features
    # FEATURE_COLS = [ # all these features should be also represented in the SusComp/compute_carbon_intensity.py mapping
    #     'Biomass',
    #     #'Energy storage',
    #     'Fossil Brown coal/Lignite',
    #     'Fossil Coal-derived gas',
    #     'Fossil Gas',
    #     'Fossil Hard coal',
    #     'Fossil Oil',
    #     #'Fossil Oil shale',
    #     #'Fossil Peat',
    #     'Geothermal',
    #     #'Hydro Pumped Storage',
    #     'Hydro Run-of-river and pondage',
    #     'Hydro Water Reservoir',
    #     #'Marine',
    #     'Nuclear',
    #     'Other',
    #     'Other renewable',
    #     'Solar',
    #     'Waste',
    #     'Wind Offshore',
    #     'Wind Onshore',
    # ]

    HIST_FEATURE_COLS = [
        # Historical Energy Generation
        'Biomass', 'Fossil Brown coal/Lignite',
        'Fossil Coal-derived gas', 'Fossil Gas', 'Fossil Hard coal',
        'Fossil Oil', 'Geothermal',
       'Hydro Run-of-river and pondage',
        'Hydro Water Reservoir', 'Nuclear', 'Other',
        'Other renewable', 'Solar', 'Waste', 'Wind Offshore', 'Wind Onshore',
        #'total_power_mw', 'emissions_weighted',

        # Historical weather
        'wx_mean__temperature_2m (°C)', 'wx_mean__relative_humidity_2m (%)',
        'wx_mean__cloud_cover (%)', 'wx_mean__wind_speed_100m (km/h)',
        'wx_mean__wind_direction_100m (°)', 'wx_mean__precipitation (mm)',
        'wx_mean__soil_moisture_0_to_7cm (m³/m³)',
        'wx_mean__shortwave_radiation (W/m²)', 'wx_std__temperature_2m (°C)',
        'wx_std__relative_humidity_2m (%)', 'wx_std__cloud_cover (%)',
        'wx_std__wind_speed_100m (km/h)', 'wx_std__wind_direction_100m (°)',
        'wx_std__precipitation (mm)', 'wx_std__soil_moisture_0_to_7cm (m³/m³)',
        'wx_std__shortwave_radiation (W/m²)',
    ]

    FUT_FEATURE_COLS = [
        # Forecasted weather
        'wxf__wx_mean__temperature_2m (°C)',
        'wxf__wx_mean__relative_humidity_2m (%)',
        'wxf__wx_mean__cloud_cover (%)', 'wxf__wx_mean__wind_speed_120m (km/h)',
        'wxf__wx_mean__wind_direction_120m (°)',
        'wxf__wx_mean__precipitation (mm)',
        'wxf__wx_mean__soil_moisture_1_to_3cm (m³/m³)',
        'wxf__wx_mean__shortwave_radiation (W/m²)',
        'wxf__wx_std__temperature_2m (°C)',
        'wxf__wx_std__relative_humidity_2m (%)', 'wxf__wx_std__cloud_cover (%)',
        'wxf__wx_std__wind_speed_120m (km/h)',
        'wxf__wx_std__wind_direction_120m (°)',
        'wxf__wx_std__precipitation (mm)',
        'wxf__wx_std__soil_moisture_1_to_3cm (m³/m³)',
        'wxf__wx_std__shortwave_radiation (W/m²)'
    ]

    TARGET_COL = "carbon_intensity"

    # Loading data
    df = pd.read_csv(
        "../new_data/germany_energy_with_weather_with_forecast_CI.csv",
        parse_dates=["time"],
        index_col="time",
    )
    df = df.sort_index()

    # Sanity check
    df = sanity_check_timeseries_df(df, feature_cols=HIST_FEATURE_COLS+FUT_FEATURE_COLS,
                                    target_col=TARGET_COL, expected_freq="h")

    # Model configuration
    cfg = LSTMForecastConfig(
        hist_input_size=len(HIST_FEATURE_COLS)+6,  # +6 for time encodings
        fut_input_size=len(FUT_FEATURE_COLS)+6,
        horizon=24,
        hidden_size=1024,
        num_layers=2,
        dropout=0.15,
        bidirectional=False,
        head_hidden_size=1024,
        head_dropout=0.15,
        use_layernorm=True,
    )

    # Running training pipeline
    train.run_training(
        df=df,
        hist_feature_cols=HIST_FEATURE_COLS,
        fut_feature_cols=FUT_FEATURE_COLS,
        target_col=TARGET_COL,
        lookback_steps=30*24,
        horizon_steps=24,      # set None to infer; use "H" if your data is hourly
        batch_size=16,
        lr=0.0012,
        epochs=150,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cfg=cfg,
    )

def xai():
    # Setting features
    # FEATURE_COLS = [ # all these features should be also represented in the SusComp/compute_carbon_intensity.py mapping
    #     'Biomass',
    #     #'Energy storage',
    #     'Fossil Brown coal/Lignite',
    #     'Fossil Coal-derived gas',
    #     'Fossil Gas',
    #     'Fossil Hard coal',
    #     'Fossil Oil',
    #     #'Fossil Oil shale',
    #     #'Fossil Peat',
    #     'Geothermal',
    #     #'Hydro Pumped Storage',
    #     'Hydro Run-of-river and pondage',
    #     'Hydro Water Reservoir',
    #     #'Marine',
    #     'Nuclear',
    #     'Other',
    #     'Other renewable',
    #     'Solar',
    #     'Waste',
    #     'Wind Offshore',
    #     'Wind Onshore',
    # ]

    HIST_FEATURE_COLS = [
        # Historical Energy Generation
        'Biomass', 'Fossil Brown coal/Lignite',
        'Fossil Coal-derived gas', 'Fossil Gas', 'Fossil Hard coal',
        'Fossil Oil', 'Geothermal',
        'Hydro Run-of-river and pondage',
        'Hydro Water Reservoir', 'Nuclear', 'Other',
        'Other renewable', 'Solar', 'Waste', 'Wind Offshore', 'Wind Onshore',
        'carbon_intensity'
        #'total_power_mw', 'emissions_weighted',

        # Historical weather
        'wx_mean__temperature_2m (°C)', 'wx_mean__relative_humidity_2m (%)',
        'wx_mean__cloud_cover (%)', 'wx_mean__wind_speed_100m (km/h)',
        'wx_mean__wind_direction_100m (°)', 'wx_mean__precipitation (mm)',
        'wx_mean__soil_moisture_0_to_7cm (m³/m³)',
        'wx_mean__shortwave_radiation (W/m²)', 'wx_std__temperature_2m (°C)',
        'wx_std__relative_humidity_2m (%)', 'wx_std__cloud_cover (%)',
        'wx_std__wind_speed_100m (km/h)', 'wx_std__wind_direction_100m (°)',
        'wx_std__precipitation (mm)', 'wx_std__soil_moisture_0_to_7cm (m³/m³)',
        'wx_std__shortwave_radiation (W/m²)',
    ]

    FUT_FEATURE_COLS = [
        # Forecasted weather
        'wxf__wx_mean__temperature_2m (°C)',
        'wxf__wx_mean__relative_humidity_2m (%)',
        'wxf__wx_mean__cloud_cover (%)', 'wxf__wx_mean__wind_speed_120m (km/h)',
        'wxf__wx_mean__wind_direction_120m (°)',
        'wxf__wx_mean__precipitation (mm)',
        'wxf__wx_mean__soil_moisture_1_to_3cm (m³/m³)',
        'wxf__wx_mean__shortwave_radiation (W/m²)',
        'wxf__wx_std__temperature_2m (°C)',
        'wxf__wx_std__relative_humidity_2m (%)', 'wxf__wx_std__cloud_cover (%)',
        'wxf__wx_std__wind_speed_120m (km/h)',
        'wxf__wx_std__wind_direction_120m (°)',
        'wxf__wx_std__precipitation (mm)',
        'wxf__wx_std__soil_moisture_1_to_3cm (m³/m³)',
        'wxf__wx_std__shortwave_radiation (W/m²)'
    ]

    TARGET_COL = "carbon_intensity"

    # Loading data
    df = pd.read_csv(
        "../new_data/germany_energy_with_weather_with_forecast_CI.csv",
        parse_dates=["time"],
        index_col="time",
    )
    df = df.sort_index()

    # Sanity check
    df = sanity_check_timeseries_df(df, feature_cols=HIST_FEATURE_COLS+FUT_FEATURE_COLS, target_col=TARGET_COL, expected_freq="h")

    # Model configuration
    cfg = LSTMForecastConfig(
        hist_input_size=len(HIST_FEATURE_COLS)+6,  # +6 for time encodings
        fut_input_size=len(FUT_FEATURE_COLS)+6,
        horizon=24,
        hidden_size=1024,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
        head_hidden_size=1024,
        head_dropout=0.3,
        use_layernorm=True,
    )

    res = run_xai_permutation(
        df=df,
        hist_feature_cols=HIST_FEATURE_COLS,
        fut_feature_cols=FUT_FEATURE_COLS,
        target_col=TARGET_COL,
        cfg=cfg,
        model_pth_path="best_model.pth",
        lookback_steps=30*24,
        horizon_steps=24,
        batch_size=16,
        n_repeats=5,
        max_batches=100,
        out_dir="xai",
    )

if __name__ == "__main__":
    #test()
    main()
    #xai()