import requests

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
API_KEY = "852cc563-92ea-4a53-94f6-5323b71e6bcd"  # Your Key
ZONE_GEN = '10Y1001A1001A82H'  # DE-LU

# Querying just 1 day of Generation Data
params = {
    'securityToken': API_KEY,
    'documentType': 'A75',  # Actual Generation
    'processType': 'A16',  # Realized
    'in_Domain': ZONE_GEN,
    'periodStart': "202401010000",
    'periodEnd': "202401020000"
}

print("--- GENERATION ACCESS CHECK ---")
try:
    r = requests.get("https://web-api.tp.entsoe.eu/api", params=params, timeout=10)
    print(f"HTTP Status: {r.status_code}")
    print("Server Response:")
    print(r.text[:500])  # Print first 500 characters

    if "No matching data found" in r.text and r.status_code == 200:
        print("\nVERDICT: [LOCKED]")
        print("Your key works, but ENTSO-E has hidden this data from you.")
        print("You MUST email 'transparency@entsoe.eu' to request API access.")
    elif "Authentication failed" in r.text:
        print("\nVERDICT: [INVALID KEY]")
    elif "<TimeSeries>" in r.text:
        print("\nVERDICT: [SUCCESS]")
        print("Data is actually there. The library bug is the only problem.")
except Exception as e:
    print(e)