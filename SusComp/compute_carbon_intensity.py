import numpy as np
import pandas as pd


# To stay consistnet with the paper (Lets wait awhile),
# we also used the source http://www.ipcc-wg3.de/report/IPCC_SRREN_Annex_II.pdf for mapping energy sources to an emissions factor

EMISSION_MAP = {  # all these features should be also represented in the SusComp/compute_carbon_intensity.py mapping
    'Biomass' : 18,
    # 'Energy storage',
    'Fossil Brown coal/Lignite':1001,
    'Fossil Coal-derived gas':1001,
    'Fossil Gas':469,
    'Fossil Hard coal':1001,
    'Fossil Oil':840,
    # 'Fossil Oil shale',
    # 'Fossil Peat',
    'Geothermal':45,
    #'Hydro Pumped Storage',
    'Hydro Run-of-river and pondage':4,
    'Hydro Water Reservoir':4,
    # 'Marine',
    'Nuclear':16,
    'Other':770,  # Avg of fossil
    'Other renewable':22,  # Avg. of renewables
    'Solar':46,
    'Waste':770, # Avg of fossil
    'Wind Offshore':12,
    'Wind Onshore':12,
    'total_carbon_import':1, # the emissions imported from other countries
}


def add_carbon_intensity(df):
    """
    Compute power-weighted carbon intensity (gCO2/kWh)
    for each timestamp from ENTSO-E generation data.
    """

    power_cols = list(EMISSION_MAP.keys())

    #  adding energy column for total energy computation
    power_cols.remove('total_carbon_import')
    power_cols.append('total_energy_import')

    # Sanity check
    missing_cols = [col for col in power_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    # Total power (MW)
    df['total_power_mw'] = df[power_cols].sum(axis=1)

    # adding carbon column for emissions computation
    power_cols.remove('total_energy_import')
    power_cols.append('total_carbon_import')

    # Power-weighted emissions (MW * gCO2/kWh)
    df['emissions_weighted'] = sum(
        df[col] * EMISSION_MAP[col]
        for col in power_cols
    )

    # Carbon intensity (gCO2/kWh)
    df['carbon_intensity'] = np.where(
        df['total_power_mw'] > 0,
        df['emissions_weighted'] / df['total_power_mw'],
        0
    )

    return df


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("full_datasets/den_2225_generation_carbon.csv")
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time").sort_index()

    df_ci = add_carbon_intensity(df)
    print(df_ci[['carbon_intensity']].head())

    df_ci.to_csv("full_datasets/den_2225_generation_carbon_wTarget.csv")

