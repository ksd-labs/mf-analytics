"""
data_ingestion/validators.py

Validation utilities for:

1. Supported indices
2. Date ranges
3. Download parameters
"""

from datetime import datetime

from indices.config.index_metadata import (
    INDEX_METADATA,
)


def validate_index_name(index_name: str) -> None:
    """
    Validate that the index exists in metadata.

    Parameters
    ----------
    index_name : str

    Raises
    ------
    ValueError
    """

    if not isinstance(index_name, str):
        raise TypeError(
            "index_name must be a string."
        )

    if index_name not in INDEX_METADATA:
        supported = ", ".join(
            sorted(INDEX_METADATA.keys())
        )

        raise ValueError(
            f"Unsupported index: {index_name}\n"
            f"Supported indices:\n{supported}"
        )


def validate_date_order(
    start_date: datetime,
    end_date: datetime
) -> None:
    """
    Ensure start date is before end date.

    Raises
    ------
    ValueError
    """

    if start_date > end_date:
        raise ValueError(
            "start_date must be earlier than end_date."
        )


def validate_chunk_size(
    chunk_days: int
) -> None:
    """
    Validate chunk size.

    Raises
    ------
    ValueError
    """

    if chunk_days <= 0:
        raise ValueError(
            "chunk_days must be positive."
        )


def validate_update_inputs(
    index_name: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """
    Master validation helper.
    """

    validate_index_name(index_name)
    validate_date_order(
        start_date,
        end_date
    )