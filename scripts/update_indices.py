"""
scripts/update_indices.py

Update all validated benchmark indices.

Run from the mf_analytics project root:
    python -m scripts.update_indices

CHANGE from indice_loader standalone:
    Import paths updated to indices.* namespace.
    VALIDATED_INDICES list and all main() logic unchanged.
"""

from indices.data_ingestion.nifty_tri_downloader import (   # ← updated
    update_index,
)
from indices.data_ingestion.cache_manager import (           # ← updated
    load_existing_data,
)


VALIDATED_INDICES = [
    "NIFTY 500",
    "NIFTY 50",
    "NIFTY 100",
    "NIFTY MIDCAP 150",
    "NIFTY SMALLCAP 250",
    "NIFTY100 QUALITY 30",
    "NIFTY200 QUALITY 30",
    "NIFTY200 MOMENTUM 30",
    "NIFTY500 VALUE 50",
    "NIFTY50 VALUE 20",
    "NIFTY100 LOW VOLATILITY 30",
]


def main():
    print("\n" + "=" * 60)
    print("UPDATING ALL BENCHMARK INDICES")
    print("=" * 60)

    success = []
    failed  = []

    for index_name in VALIDATED_INDICES:
        print(f"\nUpdating: {index_name}")

        try:
            before_rows = len(load_existing_data(index_name))
            df          = update_index(index_name)
            after_rows  = len(df)
            rows_added  = after_rows - before_rows
            status      = "UPDATED" if rows_added > 0 else "CURRENT"

            print(f"Status : {status}")
            if rows_added > 0:
                print(f"Added  : {rows_added:,}")
            print(f"Rows   : {after_rows:,}")

            success.append((index_name, after_rows))

        except Exception as exc:
            failed.append((index_name, str(exc)))
            print(f"FAILED | {exc}")

    print("\n" + "=" * 60)
    print("UPDATE SUMMARY")
    print("=" * 60)

    print(f"\nSuccessful: {len(success)}")
    for index_name, rows in success:
        print(f"  OK  {index_name:<30}{rows:>8,} rows")

    print(f"\nFailed: {len(failed)}")
    for index_name, error in failed:
        print(f"  XX  {index_name}")
        print(f"      {error}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
