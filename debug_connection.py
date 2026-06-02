"""
debug_connection.py
===================
Standalone diagnostic script — run this BEFORE starting the app
to verify your environment is set up correctly.

Run from Anaconda Prompt (inside mf_analytics folder):
    python debug_connection.py

This checks:
  1. All required libraries are installed
  2. mftool version
  3. AMFI connectivity
  4. mfapi.in connectivity
  5. A live NAV fetch for one known fund
"""

import sys
import importlib

print("=" * 60)
print("  MF Analytics Platform — Connection Diagnostics")
print("=" * 60)

# ── 1. Library checks ─────────────────────────────────────────────────────────
print("\n[1] Checking required libraries...")
REQUIRED = {
    "streamlit": "1.35.0",
    "mftool": "2.0.0",
    "pandas": "2.0.0",
    "numpy": "1.26.0",
    "scipy": "1.11.0",
    "plotly": "5.18.0",
    "requests": "2.28.0",
}

all_ok = True
for lib, min_ver in REQUIRED.items():
    try:
        mod = importlib.import_module(lib)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  ✓ {lib:<15} {ver}")
    except ImportError:
        print(f"  ✗ {lib:<15} NOT INSTALLED  →  pip install {lib}")
        all_ok = False

if not all_ok:
    print("\nInstall missing libraries then re-run this script.")
    sys.exit(1)

# ── 2. mftool version ────────────────────────────────────────────────────────
print("\n[2] Checking mftool API compatibility...")
import mftool as mf_module
mftool_ver = getattr(mf_module, "__version__", "unknown")
print(f"  mftool version: {mftool_ver}")

from mftool import Mftool
mf = Mftool()

# Check which methods exist
has_scheme_codes = hasattr(mf, 'get_scheme_codes')
has_avail        = hasattr(mf, 'get_available_schemes')
has_history      = hasattr(mf, 'get_scheme_historical_nav')
has_hist_new     = hasattr(mf, 'history')

print(f"  get_scheme_codes()           : {'✓ exists' if has_scheme_codes else '✗ missing'}")
print(f"  get_available_schemes()      : {'✓ exists (needs AMC name in v3+)' if has_avail else '✗ missing'}")
print(f"  get_scheme_historical_nav()  : {'✓ exists' if has_history else '✗ missing'}")
print(f"  history()                    : {'✓ exists' if has_hist_new else '✗ missing'}")

# ── 3. Network connectivity ───────────────────────────────────────────────────
print("\n[3] Checking network connectivity...")
import requests

URLS = {
    "AMFI NAV file (scheme list)":   "https://www.amfiindia.com/spages/NAVAll.txt",
    "mfapi.in (NAV details)":        "https://api.mfapi.in/mf/119551",
    "mfapi.in (all schemes)":        "https://api.mfapi.in/mf",
}

connectivity = {}
for name, url in URLS.items():
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"})
        ok = r.status_code == 200
        connectivity[url] = ok
        status = f"✓ HTTP {r.status_code} ({len(r.text):,} chars)" if ok else f"✗ HTTP {r.status_code}"
        print(f"  {status}  →  {name}")
    except requests.exceptions.ConnectionError as e:
        connectivity[url] = False
        print(f"  ✗ Connection refused  →  {name}")
        print(f"    Error: {e}")
    except requests.exceptions.Timeout:
        connectivity[url] = False
        print(f"  ✗ Timeout (15s)  →  {name}")
    except Exception as e:
        connectivity[url] = False
        print(f"  ✗ {type(e).__name__}: {e}  →  {name}")

# ── 4. mftool live test ──────────────────────────────────────────────────────
print("\n[4] Testing mftool scheme list fetch...")
try:
    codes = mf.get_scheme_codes(as_json=False)
    if codes and len(codes) > 0:
        print(f"  ✓ get_scheme_codes() returned {len(codes):,} schemes")
        # Show a few
        sample = list(codes.items())[:3]
        for code, name in sample:
            print(f"    {code}: {name}")
    else:
        print("  ✗ get_scheme_codes() returned empty dict")
        print("    → This means AMFI URL (https://www.amfiindia.com/spages/NAVAll.txt)")
        print("      returned no data. Check your internet/firewall.")
except Exception as e:
    print(f"  ✗ get_scheme_codes() failed: {type(e).__name__}: {e}")

# ── 5. Live NAV fetch test ───────────────────────────────────────────────────
print("\n[5] Testing live NAV fetch (Axis Bluechip Fund, code 120503)...")
TEST_CODE = "120503"   # Axis Bluechip Fund - Direct Growth

try:
    # Try mftool method first
    nav_data = mf.get_scheme_historical_nav(TEST_CODE, as_Dataframe=True)

    if nav_data is not None and not (hasattr(nav_data, 'empty') and nav_data.empty):
        import pandas as pd
        if isinstance(nav_data, pd.DataFrame):
            print(f"  ✓ get_scheme_historical_nav() returned DataFrame:")
            print(f"    Shape:   {nav_data.shape}")
            print(f"    Columns: {list(nav_data.columns)}")
            print(f"    Index name: {nav_data.index.name}")
            print(f"    Sample:\n{nav_data.head(3)}")
        else:
            print(f"  ✓ Returned: {type(nav_data)}")
    else:
        print(f"  ✗ get_scheme_historical_nav() returned None or empty")
        print(f"    → is_valid_code('{TEST_CODE}') = {mf.is_valid_code(TEST_CODE)}")
        print(f"    → This usually means scheme_codes list is empty (AMFI URL blocked)")
        
        # Try direct API fallback
        print("\n  Trying direct mfapi.in fallback...")
        r = requests.get(f"https://api.mfapi.in/mf/{TEST_CODE}", timeout=15)
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Direct API works! Scheme: {data['meta']['scheme_name']}")
            print(f"    NAV records: {len(data['data'])}")
            print(f"    Latest NAV: {data['data'][0]}")
        else:
            print(f"  ✗ Direct API also returned HTTP {r.status_code}")

except Exception as e:
    print(f"  ✗ NAV fetch failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

amfi_ok    = connectivity.get("https://www.amfiindia.com/spages/NAVAll.txt", False)
mfapi_ok   = connectivity.get("https://api.mfapi.in/mf/119551", False)

if amfi_ok and mfapi_ok:
    print("  ✅ All connections working — run: streamlit run app.py")
elif mfapi_ok and not amfi_ok:
    print("  ⚠️  AMFI blocked but mfapi.in works.")
    print("     The app will use the direct API fallback.")
    print("     Run: streamlit run app.py")
elif amfi_ok and not mfapi_ok:
    print("  ⚠️  mfapi.in blocked but AMFI works.")
    print("     Scheme list will load, but individual NAVs may fail.")
elif not amfi_ok and not mfapi_ok:
    print("  ❌ Both data sources are blocked.")
    print("  Possible causes:")
    print("    • Corporate/office network firewall blocking external sites")
    print("    • VPN is active and routing traffic through a restricted gateway")
    print("    • Antivirus / Windows Defender blocking Python's requests")
    print()
    print("  Try:")
    print("    1. Disable VPN temporarily")
    print("    2. Try on a personal WiFi/hotspot")
    print("    3. Add Python / requests to antivirus exceptions")
    print("    4. Run: pip install --upgrade certifi")
    print("       Then retry")

print()
