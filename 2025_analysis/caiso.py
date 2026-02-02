import pandas as pd
import gridstatus

caiso = gridstatus.CAISO()

start = pd.Timestamp("2026-01-01")
end = pd.Timestamp("2026-01-09")  # end is exclusive

mix = caiso.get_fuel_mix(start=start, end=end, verbose=True)

mix.to_csv("caiso_fuel_mix_2025.csv", index=False)
print(mix.head())
print("Saved caiso_fuel_mix_2025.csv")
