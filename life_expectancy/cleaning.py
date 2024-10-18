import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Dict, Any
import logging
import argparse
import os

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
    def __init__(self, country: str = 'PT') -> None:
        """
        Initialize LifeExpectancyCleaner object.

        Args:
        -   country (str): Country code to filter data by. Default is 'PT'.
        """
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
            return le.dict()
        except ValidationError as e:
            raise ValidationError(f"Validation error: {e}")  # Propagate validation errors
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error processing row: {e}")  # Handle unexpected errors

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and validates data.

        Args:
        -   data (pd.DataFrame): Input DataFrame to clean.

        Returns:
        -   pd.DataFrame: Cleaned DataFrame.
        """
        try:
            # Unpivot DataFrame to long format
            df_long = pd.melt(data, id_vars=['unit,sex,age,geo\\time'], var_name='year', value_name='value')
            df_long[['unit', 'sex', 'age', 'geo\\time']] = df_long['unit,sex,age,geo\\time'].str.split(',', expand=True)
            df_long.drop(columns=['unit,sex,age,geo\\time'], inplace=True)
            df_long.rename(columns={'geo\\time': 'region'}, inplace=True)

            # Filter for rows where region is the specified country
            df_long = df_long[df_long['region'] == self.country]

            validated_rows = []
            for _, row in df_long.iterrows():
                validated_rows.append(self.validate_row(row))

            validated_df = pd.DataFrame(validated_rows)

            # Remove rows with NaN values in 'value' column
            validated_df.dropna(subset=['value'], inplace=True)

            return validated_df

        except Exception as e:
            self.logger.error(f"Unexpected error occurred during data cleaning: {e}")
            raise RuntimeError(f"Unexpected error occurred during data cleaning: {e}")

def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load data from input file.

    Args:
    -   input_path (Path): Path to the input TSV file.

    Returns:
    -   pd.DataFrame: Loaded DataFrame.
    """
    try:
        return pd.read_csv(input_path, sep='\t')
    except FileNotFoundError as e:
        logging.error(f"File not found: {input_path}")
        raise e
    except Exception as e:
        logging.error(f"Error loading data from {input_path}: {e}")
        raise RuntimeError(f"Error loading data from {input_path}: {e}")

def save_data(data: pd.DataFrame, output_path: Path) -> None:
    """
    Save data to CSV file.

    Args:
    -   data (pd.DataFrame): Data to save.
    -   output_path (Path): Path to save the cleaned CSV file.
    """
    try:
        data.to_csv(output_path, index=False)
        logging.info(f"Cleaned data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving data to {output_path}: {e}")
        raise RuntimeError(f"Error saving data to {output_path}: {e}")

def main(input_path: Path, output_path: Path, country: str = 'PT') -> None:
    """
    Main function to orchestrate data cleaning process.

    Args:
    -   input_path (Path): Path to the input TSV file.
    -   output_path (Path): Path to save the cleaned CSV file.
    -   country (str): Country code to filter data by. Default is 'PT'.
    """
    try:
        # Load data
        data = load_data(input_path)

        # Initialize cleaner
        cleaner = LifeExpectancyCleaner(country)

        # Clean data
        cleaned_data = cleaner.clean_data(data)

        # Save cleaned data
        save_data(cleaned_data, output_path)

    except Exception as e:
        logging.error(f"Unexpected error occurred during processing: {e}")
        raise RuntimeError(f"Unexpected error occurred during processing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean life expectancy data")

    default_input_path = os.path.join('life_expectancy', 'data', 'eu_life_expectancy_raw.tsv')
    default_output_path = os.path.join('life_expectancy', 'data', 'pt_life_expectancy.csv')

    parser.add_argument('--input', type=str, default=default_input_path, help='Path to the input TSV file')
    parser.add_argument('--output', type=str, default=default_output_path, help='Path to save the cleaned CSV file')
    parser.add_argument('--country', type=str, default='PT', help='Country code to filter data by')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    main(input_path, output_path, args.country)
