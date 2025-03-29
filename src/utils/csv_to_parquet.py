#!/usr/bin/env python3

import os
import argparse
import pyarrow.csv as pv
import pyarrow.parquet as pq

def csv_to_parquet_pyarrow(csv_file_path, parquet_file_path):
    """
    Converts a CSV file to a Parquet file using pyarrow.

    Parameters:
        csv_file_path (str): Path to the input CSV file.
        parquet_file_path (str): Path to the output Parquet file.

    Returns:
        None: Saves the Parquet file to the specified path.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(parquet_file_path), exist_ok=True)

        # Read CSV into a PyArrow Table
        table = pv.read_csv(csv_file_path)

        # Write the Table to a Parquet file
        pq.write_table(table, parquet_file_path)

        print(f"Successfully converted '{csv_file_path}' to '{parquet_file_path}'.")
    except Exception as e:
        print(f"An error occurred while converting CSV to Parquet: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CSV file to Parquet using PyArrow.")
    parser.add_argument("csv_file_path", help="Path to the input CSV file.")
    parser.add_argument("parquet_file_path", help="Path to the output Parquet file.")
    args = parser.parse_args()

    csv_to_parquet_pyarrow(args.csv_file_path, args.parquet_file_path)
