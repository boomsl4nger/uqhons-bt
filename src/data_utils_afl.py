"""
Match data is sourced from: https://github.com/akareen/AFL-Data-Analysis. This is an excellent 
data store with a great lot of information, much of which is currently out of scope in this project.
"""

import csv
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from typing import List, Optional

def check_csv_headers(path: str, expected_headers: Optional[List[str]] = None) -> bool:
    """
    Checks if all CSV files in a directory have a specific set of column names.

    Args:
        path (str): The path to the directory containing the CSV files.
        expected_headers (Optional[List[str]]): A list of column names to check against.
            If None, the headers from the first CSV file found will be used as the standard.

    Returns:
        bool: True if all files have matching headers, False otherwise.
    
    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If a non-list is provided for expected_headers.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    if expected_headers is not None and not isinstance(expected_headers, list):
        raise ValueError("expected_headers must be a list of strings or None.")

    csv_files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in '{path}'.")
        return True

    if expected_headers is None:
        first_file_path = os.path.join(path, csv_files[0])
        try:
            with open(first_file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                expected_headers = next(reader)
                print(f"Using '{first_file_path}' headers as the standard: {expected_headers}")
        except (csv.Error, IndexError):
            print(f"Could not read headers from the first file: {first_file_path}")
            return False
            
    # Check all files against the determined standard
    for filename in csv_files:
        filepath = os.path.join(path, filename)
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                current_headers = next(reader)
                if current_headers != expected_headers:
                    print(f"Mismatch found in '{filename}'.")
                    print(f"  Expected: {expected_headers}")
                    print(f"  Found:    {current_headers}")
                    return False
        except (csv.Error, IndexError) as e:
            print(f"Could not read headers from '{filename}': {e}")
            return False

    print("All CSV files have matching headers.")
    return True

def calculate_points_afl(goals: int, behinds: int) -> int:
    """Calculate the points for an AFL team based on the number of goals and behinds. Simply, goals
    are worth 6 points and behinds are worth 1 point.

    See: https://afl-explained.com.au/scoring/

    Args:
        goals (int): Number of goals scored.
        behinds (int): Number of behinds scored.

    Returns:
        int: Total number of points.
    """
    return goals * 6 + behinds

def _get_unique_from_column(path: str, col_name: str, start_year: int = None, end_year: int = None) -> set:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    # Determine the list of files to process
    # TODO refactor out
    if start_year is not None and end_year is not None:
        files_to_process = [f"matches_{year}.csv" for year in range(start_year, end_year + 1)]
    else:
        files_to_process = [f for f in os.listdir(directory) if f.startswith("matches_") and f.endswith(".csv")]

    unique_vals = set()
    for f in files_to_process:
        filepath = os.path.join(path, f)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}. Skipping...")
            continue

        df = pd.read_csv(filepath)
        if col_name not in df.columns:
            print(f"Column '{col_name}' not found in file: {filepath}. Skipping...")
            continue
        
        unique_vals.update(df[col_name].unique())

    return unique_vals

def get_unique_teams(path: str, start_year: int = None, end_year: int = None) -> set:
    return _get_unique_from_column(path, "team_1_team_name", start_year, end_year)

def get_unique_venues(path: str, start_year: int = None, end_year: int = None) -> set:
    return _get_unique_from_column(path, "venue", start_year, end_year)

def load_data_afl(
        path: str, start_year: int = None, end_year: int = None, 
        tenants: DataFrame = None, aliases: dict = None, include_venue: bool = False,
        nominal_home: bool = False, fix_swapped: bool = False, print_missing_data: bool = False
    ) -> DataFrame:
    # For each row in each csv, which represents a single match between two teams in a given season
    # (i) Determine which is the home team, or if the game is neutral
    # (ii) Determine scores of each team
    # (iii) Append the row: (home, away, hscore, ascore, is_neutral, time)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if start_year is not None and end_year is not None:
        files_to_process = [f"matches_{year}.csv" for year in range(start_year, end_year + 1)]
    else:
        files_to_process = [f for f in os.listdir(directory) if f.startswith("matches_") and f.endswith(".csv")]

    # Precompute team-specific venue DFs
    if tenants is not None:
        tenants_map = {team: df for team, df in tenants.groupby("team")}

        def is_home_venue(team, venue, year):
            if team not in tenants_map:
                return False
            
            df = tenants_map[team]
            return ((df["venue"] == venue) & (df["start_year"] <= year) & (df["end_year"] >= year)).any()

    # Check all files against the determined standard
    result = []
    for filename in files_to_process:
        filepath = os.path.join(path, filename)
        cur_file = pd.read_csv(filepath, )
        cur_year = cur_file.loc[:, "year"].unique()[0]

        for i, row in cur_file.iterrows():
            hteam = row.loc["team_1_team_name"]
            hscore = calculate_points_afl(row.loc["team_1_final_goals"], row.loc["team_1_final_behinds"])
            ateam = row.loc["team_2_team_name"]
            ascore = calculate_points_afl(row.loc["team_2_final_goals"], row.loc["team_2_final_behinds"])
            venue = row.loc["venue"]

            # Check aliases
            if aliases is not None:
                if hteam in aliases:
                    hteam = aliases[hteam]
                if ateam in aliases:
                    ateam = aliases[ateam]

            # Check missing
            if any(pd.isnull([hteam, ateam, hscore, ascore, venue])):
                if print_missing_data:
                    print(f"Missing data in file '{filename}' at line {i}")

                continue # Skip missing data

            # Determine if the game is neutral based on the venue and tenant info (if provided)
            if tenants is not None:
                i_home = is_home_venue(hteam, venue, cur_year)
                j_home = is_home_venue(ateam, venue, cur_year)
                if nominal_home:
                    is_neutral = int(not i_home and not j_home)
                else:
                    is_neutral = int(i_home == j_home)

                # Also check if the home and away teams are swapped
                if not i_home and j_home:
                    if print_missing_data: print(f"Swapped (?) : {[row["round_num"], venue, str(cur_year), hteam, ateam]}")
                    if fix_swapped:
                        hteam, ateam = ateam, hteam
                        hscore, ascore = ascore, hscore
            else:
                is_neutral = 0

            dummy = [hteam, ateam, hscore, ascore, is_neutral, cur_year]
            if include_venue: dummy.append(venue)
            result.append(dummy)

    cols = ["home", "away", "home_score", "away_score", "is_neutral", "year"]
    if include_venue: cols.append("venue")
    return DataFrame(result, columns=cols).reset_index(drop=True)
