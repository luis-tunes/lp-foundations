import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Dict, Any
import logging
import argparse

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LifeExpectancyModel(BaseModel):
    unit: str
    sex: str
    age: str
    region: str
    year: int
    value: float

    @field_validator('sex')
    def check_sex(cls, v: str) -> str:
        """Validate and normalize sex field."""
        if v not in {'M', 'F', 'T'}:
            raise ValueError('Invalid sex, must be M, F, T')
        return v

    @field_validator('year')
    def check_year(cls, v: int) -> int:
        """Validate and normalize year field."""
        if not (1000 <= v <= 9999):
            raise ValueError('Invalid year, must be between 1000 and 9999')
        return v

    @field_validator('value')
    def check_value(cls, v: Any) -> float:
        """Validate and normalize value field."""
        if v is None or v == ':' or pd.isna(v):
            return float('nan')
        v = str(v).strip()
        if v == ':' or v == '':
            return float('nan')
        try:
            v = float(v)
        except ValueError:
            raise ValueError('Invalid value, must be a float')
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

class LifeExpectancyCleaner:
    def __init__(self, input_path: Path, output_path: Path, country: str = 'PT') -> None:
        """
        Initialize LifeExpectancyCleaner object.

        Args:
        -   input_path (Path): Path to the input TSV file.
        -   output_path (Path): Path to save the cleaned CSV file.
        -   country (str): Country code to filter data by. Default is 'PT'.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.country = country
        self.logger = logging.getLogger(__name__)  # Initialize logger for the class

    def clean_value(self, value: Any) -> float:
        """
        Converts input value to float or NaN.

        Args:
        -   value (Any): Input value to convert.

        Returns:
        -   float: Converted float value or NaN if conversion fails.
        """
        try:
            value = str(value).strip()
            if value in {':', ''}:
                return float('nan')  # Handle special cases
            return float(value)
        except ValueError:
            return float('nan')  # Handle conversion errors

    def validate_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Validates a single row of data using LifeExpectancyModel.

        Args:
        -   row (pd.Series): Pandas Series representing a single row of data.

        Returns:
        -   Dict[str, Any]: Validated and converted row as a dictionary.

        Raises:
        -   ValidationError: If validation of row fails.
        -   ValueError: If unexpected error occurs during processing.
        """
        try:
            year = int(row['year'])
            value = self.clean_value(row['value'])
            le = LifeExpectancyModel(unit=row['unit'], sex=row['sex'], age=row['age'], region=row['region'], year=year, value=value)
            return le.model_dump()
        except ValidationError as e:
            raise ValidationError(f"Validation error: {e}")  # Propagate validation errors
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error processing row: {e}")  # Handle unexpected errors

    def clean_data(self) -> None:
        """
        Cleans and validates data from input file.

        Raises:
        -   FileNotFoundError: If input file is not found.
        -   RuntimeError: If unexpected error occurs during data processing.
        """
        try:
            self.logger.info(f"Loading data from {self.input_path}")
            df = pd.read_csv(self.input_path, sep='\t')

            # Unpivot DataFrame to long format
            df_long = pd.melt(df, id_vars=['unit,sex,age,geo\\time'], var_name='year', value_name='value')
            df_long[['unit', 'sex', 'age', 'geo\\time']] = df_long['unit,sex,age,geo\\time'].str.split(',', expand=True)
            df_long.drop(columns=['unit,sex,age,geo\\time'], inplace=True)
            df_long.rename(columns={'geo\\time': 'region'}, inplace=True)

            # Filter for rows where region is the specified country
            df_long = df_long[df_long['region'] == self.country]

            self.logger.info("Starting data validation and conversion")
            validated_rows = []
            for _, row in df_long.iterrows():
                validated_rows.append(self.validate_row(row))

            validated_df = pd.DataFrame(validated_rows)

            # Remove rows with NaN values in 'value' column
            validated_df.dropna(subset=['value'], inplace=True)

            # Save the resulting DataFrame to CSV
            validated_df.to_csv(self.output_path, index=False)
            self.logger.info(f"Cleaned data saved to {self.output_path}")

        except FileNotFoundError as e:
            self.logger.error(f"File not found: {self.input_path}")
            raise e
        except Exception as e:
            self.logger.error(f"Unexpected error occurred: {e}")
            raise RuntimeError(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean life expectancy data")
    parser.add_argument('--input', type=str, default='data/eu_life_expectancy_raw.tsv', help='Path to the input TSV file')
    parser.add_argument('--output', type=str, default='data/pt_life_expectancy.csv', help='Path to save the cleaned CSV file')
    parser.add_argument('--country', type=str, default='PT', help='Country code to filter data by')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    cleaner = LifeExpectancyCleaner(input_path, output_path, args.country)
    cleaner.clean_data()
