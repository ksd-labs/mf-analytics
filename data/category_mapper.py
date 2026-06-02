"""
category_mapper.py
==================
Maps mftool scheme names to our 12 standardized fund categories.

Design decisions:
  - Uses keyword matching on lowercase scheme names (no extra API calls)
  - Priority order matters — more specific categories are checked first
    to avoid false positives (e.g. "Focused" before "Large Cap")
  - Index Funds are identified first and ETFs are excluded explicitly
  - filter_preferred_plans() removes Dividend, IDCW, ETF, FoF variants
    so only Growth open-ended funds remain for analysis

To extend: add new keywords to CATEGORY_KEYWORDS in constants.py.
"""

import re
from typing import Optional, Dict, List
from utils.constants import (
    CATEGORY_KEYWORDS,
    EXCLUDED_PLAN_KEYWORDS,
    EXCLUDED_STRUCTURE_KEYWORDS,
    INDEX_EXCLUSIONS,
    PREFERRED_OPTIONS,
    CATEGORIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_category_for_scheme(scheme_name: str) -> Optional[str]:
    """
    Detect the category of a mutual fund from its scheme name.

    Algorithm:
      1. Lowercase the name for case-insensitive matching
      2. Check Index Funds first (with ETF exclusion)
      3. Check remaining 11 categories in priority order

    Priority order is important:
      - "Contra" before "Large Cap" (some contra funds say "Large Cap Contra")
      - "Focused" before "Large Cap" (Focused 25 funds may have "large cap" in name)
      - "ELSS" before "Multi Cap" etc.
      - "Balanced Advantage" before "Aggressive Hybrid"

    Args:
        scheme_name: Full scheme name string from mftool

    Returns:
        Category string (one of CATEGORIES) or None if unmatched
    """
    if not scheme_name or not isinstance(scheme_name, str):
        return None

    name_lower = scheme_name.lower()

    # ── Step 1: Index Funds (check first, then exclude ETFs) ─────────────────
    has_index_keyword = any(kw in name_lower for kw in CATEGORY_KEYWORDS["Index Funds"])
    is_etf = any(excl in name_lower for excl in INDEX_EXCLUSIONS)

    if has_index_keyword and not is_etf:
        return "Index Funds"

    # ── Step 2: All other categories in priority order ────────────────────────
    # DO NOT include "Index Funds" here — already handled above
    PRIORITY_ORDER: List[str] = [
        "Contra",               # Most specific — before Value/Large Cap
        "Focused",              # "Focused 25" before "Large Cap"
        "ELSS",                 # Tax saver — before Multi Cap
        "Balanced Advantage",   # Before Aggressive Hybrid
        "Aggressive Hybrid",    # Before Multi Cap
        "Multi Cap",            # Before Flexi Cap
        "Flexi Cap",            # Before Large Cap (some flexi names include "large")
        "Small Cap",
        "Mid Cap",
        "Large Cap",
        "Value",                # Last — "Value" keyword is short and can false-match
    ]

    for category in PRIORITY_ORDER:
        keywords = CATEGORY_KEYWORDS.get(category, [])
        for keyword in keywords:
            if keyword in name_lower:
                return category

    return None   # Unrecognized — will be excluded from all categories


# ─────────────────────────────────────────────────────────────────────────────
# PLAN FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def filter_preferred_plans(all_schemes: Dict[str, str]) -> Dict[str, str]:
    """
    Filter the full scheme dict to Growth option only, removing:
      - Dividend / IDCW / Bonus options
      - ETFs (exchange-traded funds)
      - Fund of Funds
      - Fixed Maturity Plans, Interval Funds, etc.

    A scheme MUST contain a PREFERRED_OPTIONS keyword ('growth') to pass.
    It MUST NOT contain any EXCLUDED_PLAN_KEYWORDS or EXCLUDED_STRUCTURE_KEYWORDS.

    Args:
        all_schemes: {scheme_code: scheme_name} from mftool

    Returns:
        Filtered {scheme_code: scheme_name} dict
    """
    filtered: Dict[str, str] = {}

    for code, name in all_schemes.items():
        if not name or not isinstance(name, str):
            continue

        name_lower = name.lower()

        # ── MUST be a growth option ───────────────────────────────────────────
        if not any(opt in name_lower for opt in PREFERRED_OPTIONS):
            continue

        # ── MUST NOT be a dividend/IDCW/bonus option ─────────────────────────
        if any(excl in name_lower for excl in EXCLUDED_PLAN_KEYWORDS):
            continue

        # ── MUST NOT be a structural exclusion (ETF, FoF, etc.) ──────────────
        if any(excl in name_lower for excl in EXCLUDED_STRUCTURE_KEYWORDS):
            continue

        filtered[code] = name

    return filtered


def filter_direct_plans(schemes: Dict[str, str]) -> Dict[str, str]:
    """
    From an already-filtered Growth scheme dict, keep only Direct plans.

    Args:
        schemes: Pre-filtered {code: name} dict (Growth only)

    Returns:
        {code: name} with only Direct Plan schemes
    """
    return {
        code: name
        for code, name in schemes.items()
        if "direct" in name.lower()
    }


def filter_regular_plans(schemes: Dict[str, str]) -> Dict[str, str]:
    """
    From an already-filtered Growth scheme dict, keep only Regular plans.

    Note: Some older schemes don't have "Regular" in their name — they were
    named before the Direct/Regular distinction existed. This function
    keeps schemes that either explicitly say "regular" OR have neither
    "direct" nor "regular" in their name.

    Args:
        schemes: Pre-filtered {code: name} dict (Growth only)

    Returns:
        {code: name} with Regular Plan schemes
    """
    result = {}
    for code, name in schemes.items():
        name_lower = name.lower()
        if "regular" in name_lower:
            result[code] = name
        elif "direct" not in name_lower:
            # Older scheme without plan label — include as regular
            result[code] = name
    return result


# ─────────────────────────────────────────────────────────────────────────────
# NAME CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_fund_name(raw_name: str) -> str:
    """
    Remove plan/option suffixes from a scheme name for display purposes.

    Examples:
      'Axis Bluechip Fund - Direct Plan - Growth'     → 'Axis Bluechip Fund'
      'HDFC Mid-Cap Opportunities Fund - Growth'       → 'HDFC Mid-Cap Opportunities Fund'
      'SBI Small Cap Fund - Regular Plan - Growth'     → 'SBI Small Cap Fund'

    Args:
        raw_name: Full scheme name from mftool

    Returns:
        Cleaned display name
    """
    if not raw_name:
        return raw_name

    patterns = [
        r"\s*-\s*direct\s*plan.*",
        r"\s*-\s*regular\s*plan.*",
        r"\s*-\s*plan\s*[a-z1-9].*",
        r"\s*-\s*growth.*",
        r"\s*-\s*option.*",
        r"\s*\(.*\)\s*$",
    ]

    name = raw_name.strip()
    for pattern in patterns:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

    return name


def get_fund_house(scheme_name: str) -> str:
    """
    Extract the AMC/fund house name from a scheme name.
    Uses the first word(s) before the first space-dash-space as the fund house.

    This is a heuristic — not 100% reliable for all names.

    Args:
        scheme_name: Full scheme name

    Returns:
        Likely fund house name string
    """
    # Most scheme names: "AMC Name Fund Type... - Plan - Option"
    # Fund house is typically the first 1-3 words
    known_houses = [
        "Axis", "HDFC", "SBI", "ICICI Prudential", "Nippon India",
        "Kotak", "Mirae Asset", "DSP", "Franklin Templeton", "UTI",
        "Aditya Birla Sun Life", "Tata", "Invesco India", "Edelweiss",
        "PGIM India", "Motilal Oswal", "Parag Parikh", "Quant",
        "Canara Robeco", "IDFC", "Sundaram", "L&T", "Baroda BNP Paribas",
        "Navi", "360 ONE", "WhiteOak Capital", "Bandhan", "Union",
        "JM Financial", "LIC MF", "BOI AXA", "Quantum",
    ]

    name_lower = scheme_name.lower()
    for house in known_houses:
        if name_lower.startswith(house.lower()):
            return house

    # Fallback: first word
    return scheme_name.split()[0] if scheme_name.split() else "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_category_fund_counts(all_schemes: Dict[str, str]) -> Dict[str, int]:
    """
    Count how many Growth-plan funds exist per category.
    Used on the Dashboard page for the overview table.

    Args:
        all_schemes: Full {code: name} dict from mftool

    Returns:
        Dict {category: count}
    """
    preferred = filter_preferred_plans(all_schemes)
    counts: Dict[str, int] = {cat: 0 for cat in CATEGORIES}

    for name in preferred.values():
        category = get_category_for_scheme(name)
        if category and category in counts:
            counts[category] += 1

    return counts
