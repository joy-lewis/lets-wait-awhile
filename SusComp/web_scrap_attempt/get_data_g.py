import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_KEY = "852cc563-92ea-4a53-94f6-5323b71e6bcd"  # Your API Key

# EIC CODES
ZONE_GEN = '10Y1001A1001A82H'  # DE-LU (Bidding Zone)
ZONE_FLOW = '10Y1001A1001A83F'  # DE (Control Area)

# FORCE REAL DATES (Ignore System Clock)
# We fetch exactly year 2024 to ensure data exists
START_DATE = pd.Timestamp('2024-10-01', tz='UTC')
END_DATE = pd.Timestamp('2025-01-01', tz='UTC')

print(f"--- ENTSO-E BARE METAL DOWNLOADER ---")
print(f"Status:        Bypassing library bugs.")
print(f"Fetching Data: {START_DATE.date()} -> {END_DATE.date()} (2024 Data)")
print("-" * 60)


# ==========================================
# 2. API REQUESTER
# ==========================================
def fetch_entsoe(params):
    base_url = "https://web-api.tp.entsoe.eu/api"
    params['securityToken'] = API_KEY

    for attempt in range(3):
        try:
            # Request without client-side timestamp headers to avoid 2026 conflicts
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 401:
                print("   [!] Error 401: Key Invalid / Not Active")
                return None
            elif response.status_code == 429:
                print("   [!] Error 429: Rate Limit. Sleeping 30s...")
                time.sleep(30)
            else:
                if response.status_code != 400:
                    print(f"   [!] HTTP {response.status_code}")
        except Exception as e:
            time.sleep(1)
    return None


# ==========================================
# 3. XML PARSERS
# ==========================================
PSR_CODES = {
    'B01': 'Biomass', 'B02': 'Fossil Brown coal/Lignite', 'B03': 'Fossil Coal-derived gas',
    'B04': 'Fossil Gas', 'B05': 'Fossil Hard coal', 'B06': 'Fossil Oil',
    'B09': 'Geothermal', 'B10': 'Hydro Pumped Storage', 'B11': 'Hydro Run-of-river and poundage',
    'B12': 'Hydro Water Reservoir', 'B14': 'Nuclear', 'B15': 'Other renewable',
    'B16': 'Solar', 'B17': 'Waste', 'B18': 'Wind Offshore', 'B19': 'Wind Onshore',
    'B20': 'Other'
}


def parse_gen(xml):
    if not xml: return pd.DataFrame()
    try:
        # Strip namespaces to prevent parsing errors
        xml = xml.replace('xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"', '')
        root = ET.fromstring(xml)
        data = []
        for ts in root.findall('.//TimeSeries'):
            psr = ts.find('.//MktPSRType/psrType')
            if psr is None: continue
            fuel = PSR_CODES.get(psr.text, psr.text)

            period = ts.find('.//Period')
            start = pd.Timestamp(period.find('.//timeInterval/start').text)
            res = period.find('.//resolution').text
            mins = 15 if '15M' in res else 60

            for p in period.findall('.//Point'):
                pos = int(p.find('position').text)
                qty = float(p.find('quantity').text)
                t = start + pd.Timedelta(minutes=mins * (pos - 1))
                data.append({'Time': t, 'Type': fuel, 'MW': qty})

        if not data: return pd.DataFrame()
        return pd.DataFrame(data).pivot_table(index='Time', columns='Type', values='MW', aggfunc='mean')
    except:
        return pd.DataFrame()


def parse_flow(xml, neighbor):
    if not xml: return pd.DataFrame()
    try:
        xml = xml.replace('xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"', '')
        root = ET.fromstring(xml)
        data = []
        for ts in root.findall('.//TimeSeries'):
            period = ts.find('.//Period')
            start = pd.Timestamp(period.find('.//timeInterval/start').text)
            for p in period.findall('.//Point'):
                pos = int(p.find('position').text)
                qty = float(p.find('quantity').text)
                t = start + pd.Timedelta(hours=(pos - 1))
                data.append({'Time': t, neighbor: qty})
        if not data: return pd.DataFrame()
        return pd.DataFrame(data).set_index('Time')
    except:
        return pd.DataFrame()


# ==========================================
# 4. MAIN LOOP
# ==========================================
all_gen = []
all_flows = []
current_end = END_DATE

# Loop backwards month by month
while current_end > START_DATE:
    current_start = current_end - pd.DateOffset(months=1)
    if current_start < START_DATE: current_start = START_DATE

    p_start = current_start.strftime('%Y%m%d%H%M')
    p_end = current_end.strftime('%Y%m%d%H%M')

    print(f"Processing: {current_start.date()} -> {current_end.date()}")

    # 1. GENERATION
    params = {'documentType': 'A75', 'processType': 'A16', 'in_Domain': ZONE_GEN, 'periodStart': p_start,
              'periodEnd': p_end}
    df = parse_gen(fetch_entsoe(params))
    if not df.empty: all_gen.append(df)

    # 2. FLOWS
    neighbors = {
        'DK': '10YDK-1--------W', 'NL': '10YNL----------L', 'CH': '10YCH-SWISSGRIDZ',
        'CZ': '10YCZ-CEPS-----N', 'PL': '10YPL-AREA-----S', 'AT': '10YAT-APG------L',
        'FR': '10YFR-RTE------C', 'SE': '10Y1001A1001A47J', 'BE': '10YBE----------2',
        'LU': '10YLU-CEGEDEL-NQ'
    }
    for name, code in neighbors.items():
        params = {'documentType': 'A11', 'out_Domain': ZONE_FLOW, 'in_Domain': code, 'periodStart': p_start,
                  'periodEnd': p_end}
        df = parse_flow(fetch_entsoe(params), name)
        if not df.empty: all_flows.append(df)

    current_end = current_start
    time.sleep(1)

# ==========================================
# 5. MERGE & SAVE
# ==========================================
print("\nMerging Data...")
full_gen = pd.concat(all_gen).sort_index().resample('60min').mean() if all_gen else pd.DataFrame()
full_flows = pd.concat(all_flows).sort_index().resample('60min').mean() if all_flows else pd.DataFrame()
full_flows = full_flows.groupby(full_flows.index).sum()

df_final = pd.concat([full_gen, full_flows], axis=1)

# Fill Missing Cols
expected = list(PSR_CODES.values()) + list(neighbors.keys())
for c in expected:
    if c not in df_final.columns: df_final[c] = 0.0

# Carbon Intensity
co2_factors = {
    'Biomass': 230, 'Fossil Brown coal/Lignite': 1150, 'Fossil Hard coal': 820,
    'Fossil Gas': 490, 'Fossil Oil': 650, 'Geothermal': 38, 'Waste': 300,
    'Other': 700, 'Hydro Pumped Storage': 0, 'Nuclear': 12, 'Solar': 45,
    'Wind Offshore': 11, 'Wind Onshore': 11, 'Other renewable': 11
}


def get_ci(row):
    ems = 0;
    tot = 0
    for k, v in co2_factors.items():
        val = row.get(k, 0)
        if pd.notna(val) and val > 0: ems += val * v; tot += val
    return (ems / tot) if tot > 0 else 0


if not df_final.empty:
    df_final['Carbon Intensity'] = df_final.apply(get_ci, axis=1)
    df_final.index.name = 'Time'

    # Save to CSV
    filename = 'germany_2024_data.csv'
    df_final.reset_index().to_csv(filename, index=False)
    print(f"SUCCESS: Saved {len(df_final)} rows to '{filename}'")
else:
    print("FAILED: No data found (Check if API key is active).")