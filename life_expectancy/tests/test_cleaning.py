"""Tests for the cleaning module"""
import pandas as pd
from pathlib import Path

from life_expectancy.cleaning import LifeExpectancyCleaner
from . import OUTPUT_DIR


def test_clean_data(pt_life_expectancy_expected):
    """Run the `clean_data` function and compare the output to the expected output"""

    input_path = Path("life_expectancy/data/eu_life_expectancy_raw.tsv")
    output_path = OUTPUT_DIR / "pt_life_expectancy.csv"
    country = "PT"

    cleaner = LifeExpectancyCleaner(input_path, output_path, country)
    cleaner.clean_data()

    pt_life_expectancy_actual = pd.read_csv(output_path)

    print(pt_life_expectancy_actual)
    print(pt_life_expectancy_expected)

    pd.testing.assert_frame_equal(
        pt_life_expectancy_actual, pt_life_expectancy_expected
    )