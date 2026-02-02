"""Converts the generation CSVs to carbon intensity CSVs"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# A literature review of numerous total life cycle energy sources CO2 emissions per unit of electricity generated,
# conducted by the Intergovernmental Panel on Climate Change in 2011, found that the CO2 emission value,
# that fell within the 50th percentile of all total life cycle emissions studies were as follows.[6]
# http://www.ipcc-wg3.de/report/IPCC_SRREN_Annex_II.pdf
EMISSIONS_IPCC = dict(
    # Renewable
    biopower=18,
    solar_pv=46,
    solar_csp=22,
    geothermal=45,
    hydro=4,
    ocean=8,
    wind=12,
    # Fossil
    nuclear=16,
    gas=469,
    oil=840,
    coal=1001,
    # Other?
)

# Mapping from ENTSOE to IPCC labels
ENTSOE_MAP = {
    'Biomass': EMISSIONS_IPCC["biopower"],
    'Fossil Brown coal/Lignite': EMISSIONS_IPCC["coal"],
    'Fossil Gas': EMISSIONS_IPCC["gas"],
    'Fossil Coal-derived gas': EMISSIONS_IPCC["gas"],
    'Fossil Hard coal': EMISSIONS_IPCC["coal"],
    'Fossil Oil': EMISSIONS_IPCC["oil"],
    'Geothermal': EMISSIONS_IPCC["geothermal"],
    # 'Hydro Pumped Storage': 43,
    'Hydro Run-of-river and poundage': EMISSIONS_IPCC["hydro"],
    'Hydro Water Reservoir': EMISSIONS_IPCC["hydro"],
    'Nuclear': EMISSIONS_IPCC["nuclear"],
    'Other': 770,  # Avg of fossil
    'Other renewable': 22,  # Avg. of renewables
    'Solar': EMISSIONS_IPCC["solar_pv"],
    'Wind Offshore': EMISSIONS_IPCC["wind"],
    'Wind Onshore': EMISSIONS_IPCC["wind"],
    'Fossil Oil shale': EMISSIONS_IPCC["oil"],
    'Fossil Peat': EMISSIONS_IPCC["coal"],
    'Energy storage': None,  # TODO
    'Marine': EMISSIONS_IPCC["ocean"],
    'Waste': EMISSIONS_IPCC["biopower"],
    'Hydro Run-of-river and pondage': EMISSIONS_IPCC["hydro"],
    # generation Mix (https://www.carbonfootprint.com/docs/2020_09_emissions_factors_sources_for_2020_electricity_v14.pdf)
    "AT": 114,
    'BE': 150,
    "CZ": 401,
    "DK": 114,
    'FR': 42,
    "DE": 332,
    'IE': 256,
    "IT": 285,
    "LU": 124,
    'NL': 253,
    "NO": 29,
    "PL": 592,
    "ES": 153,
    "SE": 35,
    "CH": 33,
    "GB": 217,

}

# Mapping from CAISO to IPCC labels
CA_MAP = {  # WNA_map
    "Solar": EMISSIONS_IPCC["solar_pv"],
    "Wind": EMISSIONS_IPCC["wind"],
    "Geothermal": EMISSIONS_IPCC["geothermal"],
    "Biomass": EMISSIONS_IPCC["biopower"],
    "Biogas": EMISSIONS_IPCC["biopower"],
    "Small hydro": EMISSIONS_IPCC["hydro"],
    "Large hydro": EMISSIONS_IPCC["hydro"],
    "Small Hydro": EMISSIONS_IPCC["hydro"],
    "Large Hydro": EMISSIONS_IPCC["hydro"],

    "Coal": EMISSIONS_IPCC["coal"],
    "Nuclear": EMISSIONS_IPCC["nuclear"],
    "Natural gas": EMISSIONS_IPCC["gas"],
    "Natural Gas": EMISSIONS_IPCC["gas"],
    "Batteries": None,  # TODO
    "Imports": 453,  # https://www.carbonfootprint.com/docs/2020_09_emissions_factors_sources_for_2020_electricity_v14.pdf
    "Other": None,
}


# Mapping from GB to IPCC labels
GB_MAPS = {  # WNA_map
    "Biomass": EMISSIONS_IPCC["biopower"],
    'Fossil Gas': EMISSIONS_IPCC["gas"],
    'Fossil Hard coal': EMISSIONS_IPCC["coal"],
    'Fossil Oil': EMISSIONS_IPCC["oil"],
    'Hydro Pumped Storage': EMISSIONS_IPCC["hydro"],
    'Hydro Water Reservoir': EMISSIONS_IPCC["hydro"],
    'Nuclear': EMISSIONS_IPCC["nuclear"],
    'Wind Onshore': EMISSIONS_IPCC["wind"],
    'Other': 770,  # Avg of fossil

    "AT": 114,
    'BE': 150,
    "CZ": 401,
    "DK": 114,
    'FR': 42,
    "DE": 332,
    'IE': 256,
    "IT": 285,
    "LU": 124,
    'NL': 253,
    "NO": 29,
    "PL": 592,
    "ES": 153,
    "SE": 35,
    "CH": 33,
    "GB": 217,

}

def convert(generation_csv: str, mapping: Dict, out_csv: str, interpolation_factor: float = None):
    with open(generation_csv, "r") as csvfile:
        result = pd.read_csv(csvfile, index_col=0, parse_dates=True)

    # Drop unwanted columns
    for col in ["Hydro Pumped Storage", "Batteries"]:
        if col in result.columns:
            result = result.drop(col, axis=1)

    # --- DIAGNOSTIC 1: Check for missing mapping keys ---
    missing_keys = [col for col in result.columns if col not in mapping]
    if missing_keys:
        print("⚠️ WARNING: These columns have no mapping values:", missing_keys)

    # --- DIAGNOSTIC 2: Check for NaNs in the input data ---
    nan_input = result.isna().any(axis=1)
    if nan_input.any():
        print("⚠️ WARNING: Input contains NaNs at timestamps:")
        print(result[nan_input])

    # Compute totals
    total_energy = result.sum(axis=1)
    total_carbon = (result * pd.Series(mapping)).sum(axis=1)

    # --- DIAGNOSTIC 3: Division-by-zero detection ---
    zero_energy = total_energy == 0
    if zero_energy.any():
        print("❌ ERROR: total_energy is ZERO at timestamps (division by zero):")
        print(total_energy[zero_energy])

    # Compute carbon intensity
    carbon_intensity = total_carbon / total_energy

    # --- DIAGNOSTIC 4: Identify NaNs in carbon_intensity ---
    nan_ci = carbon_intensity.isna()
    if nan_ci.any():
        print("\n❌ NaN detected in carbon_intensity at timestamps:")
        for ts in carbon_intensity[nan_ci].index:
            print(f"--- {ts} ---")
            print("Row values:")
            print(result.loc[ts])
            print("total_energy:", total_energy.loc[ts])
            print("total_carbon:", total_carbon.loc[ts])
            print()

    # Continue with your interpolation logic
    if interpolation_factor:
        y = carbon_intensity.to_numpy()
        x = range(len(y))
        xnew = np.arange(0, len(y) - 1, interpolation_factor)
        interpolate = interp1d(x, y, kind=3)

        new_index = pd.date_range(
            result.index[0],
            result.index[-1],
            freq=pd.DateOffset(minutes=30)
        )
        _write_csv(new_index, interpolate(xnew), out_csv=out_csv)

    else:
        _write_csv(carbon_intensity.index, carbon_intensity.values, out_csv=out_csv)



def _write_csv(x, y, out_csv):
    csv_content = "Time,Carbon Intensity\n"
    for i, value in zip(x, y):
        csv_content += f"{i},{value}\n"
    with open(out_csv, "w") as csvfile:
        csvfile.write(csv_content)


if __name__ == '__main__':
    gb = dict(
        generation_csv="data/2025/gb_production_2025.csv",
        mapping=GB_MAPS,
        out_csv="data/gb_2025_ci.csv",
        interpolation_factor=None,
    )
    ger = dict(
        generation_csv="data/2025/ger_production_2025.csv",
        mapping=ENTSOE_MAP,
        out_csv="data/ger_2025_ci.csv",
        interpolation_factor=None,
    )
    cal = dict(
        generation_csv="data/2025/cal_production_2025.csv",
        mapping=CA_MAP,
        out_csv="data/cal_2025_ci.csv",
        interpolation_factor=None,
    )
    fr = dict(
        generation_csv="data/2025/fr_production_2025_fixed.csv",
        mapping=ENTSOE_MAP,
        out_csv="data/fr_2025_ci.csv",
        interpolation_factor=None,  
    )

    for country in [fr, gb, ger, cal]:
        convert(**country)
