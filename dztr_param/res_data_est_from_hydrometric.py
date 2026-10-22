import os
import requests
import numpy as np
import pandas as pd
from io import StringIO

def create_date_range(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date)

def get_area_volume_from_csv(csv_id, depth, folder_path):
    """Interpolate the legacy depth/area/storage table, retaining every row.

    ``depth`` retains its legacy meaning: distance below full-reservoir level.
    The CSV depth coordinate is obtained by subtracting it from dam height.
    """
    file_path = f"{folder_path}/{csv_id}.csv"

    with open(file_path, 'r') as file:
        lines = file.readlines()

    dam_height = None
    for line in lines:
        if line.startswith("Dam Height (m)"):
            dam_height = float(line.split('=')[1].strip())
            break

    if dam_height is None:
        raise ValueError("Dam Height (m) not found in the CSV file.")

    depth = dam_height - depth

    if depth > dam_height or depth < 0:
        raise ValueError("Depth must be within the range of 0 to Dam Height.")

    # Find the starting index of the actual data.
    data_start_idx = next((i for i, line in enumerate(lines) if line.lstrip().startswith("Depth(m)")), None)
    if data_start_idx is None:
        raise ValueError("Depth(m) table header not found in the CSV file.")
    data_lines = lines[data_start_idx + 1:]  # Skip the header line

    # The header has already been removed; header=None preserves the first point.
    df = pd.read_csv(
        StringIO(''.join(data_lines)),
        delimiter=',',
        skipinitialspace=True,
        header=None,
        names=['Depth(m)', 'Area(skm)', 'Storage(mcm)'],
    )
    df = df.apply(pd.to_numeric, errors='raise').sort_values('Depth(m)').reset_index(drop=True)
    if df.empty or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Geometry must contain finite numeric depth, area, and storage values.")
    if df['Depth(m)'].duplicated().any():
        raise ValueError("Geometry depths must be unique.")
    
    # Append a row for dam height if it does not exist
    if dam_height not in df['Depth(m)'].values:
        last_row = df.iloc[-1]
        new_row = pd.DataFrame({
            'Depth(m)': [dam_height],
            'Area(skm)': [last_row['Area(skm)']],
            'Storage(mcm)': [last_row['Storage(mcm)']]
        })
        df = pd.concat([df, new_row], ignore_index=True)

    if depth in df['Depth(m)'].values:
        area = df.loc[df['Depth(m)'] == depth, 'Area(skm)'].values[0]
        volume = df.loc[df['Depth(m)'] == depth, 'Storage(mcm)'].values[0]
        return area, volume

    lower_df = df[df['Depth(m)'] <= depth].tail(1)
    upper_df = df[df['Depth(m)'] > depth].head(1)

    if lower_df.empty or upper_df.empty:
        raise ValueError("Depth is out of the range of the available data.")

    lower_depth = lower_df['Depth(m)'].values[0]
    upper_depth = upper_df['Depth(m)'].values[0]

    if depth == lower_depth:
        return lower_df['Area(skm)'].values[0], lower_df['Storage(mcm)'].values[0]

    if depth == upper_depth:
        return upper_df['Area(skm)'].values[0], upper_df['Storage(mcm)'].values[0]

    area = lower_df['Area(skm)'].values[0] + (depth - lower_depth) * (upper_df['Area(skm)'].values[0] - lower_df['Area(skm)'].values[0]) / (upper_depth - lower_depth)
    volume = lower_df['Storage(mcm)'].values[0] + (depth - lower_depth) * (upper_df['Storage(mcm)'].values[0] - lower_df['Storage(mcm)'].values[0]) / (upper_depth - lower_depth)

    return area, volume

def fetch_hydrometric_data_ca(station_numbers, start_date, end_date, data_type='discharge', limit=1000):
    """Fetch daily means while preserving absent or incomplete measurements.

    Discharge sums require all selected stations on a day. Level measurements
    come from one station and are never summed. Repeated identical station/day
    values are ignored, while conflicting nonmissing values are rejected.
    """
    if data_type not in ['discharge', 'level']:
        raise ValueError("data_type must be either 'discharge' or 'level'")

    if isinstance(station_numbers, (str, bytes)):
        raise ValueError("station_numbers must be a nonempty sequence of station IDs.")
    try:
        station_numbers = list(station_numbers)
    except TypeError as exc:
        raise ValueError("station_numbers must be a nonempty sequence of station IDs.") from exc
    if not station_numbers or any(not isinstance(station, str) or not station.strip() for station in station_numbers):
        raise ValueError("Station IDs must be nonempty strings.")
    station_numbers = [station.strip() for station in station_numbers]
    if len(set(station_numbers)) != len(station_numbers):
        raise ValueError("Station IDs must be unique.")
    if data_type == 'level' and len(station_numbers) != 1:
        raise ValueError("Level observations require exactly one station; levels cannot be summed.")
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit <= 0:
        raise ValueError("limit must be a positive integer.")

    property_name = 'DISCHARGE' if data_type == 'discharge' else 'LEVEL'

    if start_date is None:
        start_date = '1900-01-01'
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    dates = create_date_range(start_date, end_date)
    if dates.empty:
        raise ValueError("start_date must be on or before end_date.")
    combined_df = pd.DataFrame({'Date': dates})

    for station_number in station_numbers:
        data_dict = {'Date': dates, station_number: [np.nan] * len(dates)}
        date_index_dict = {str(date.date()): idx for idx, date in enumerate(dates)}

        offset = 0
        full_data = []

        while True:
            url = f"https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"
            params = {
                'STATION_NUMBER': station_number,
                'datetime': f"{start_date}/{end_date}",
                'limit': limit,
                'offset': offset,
                'f': 'json'
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            if 'features' in response_data and response_data['features']:
                full_data.extend(response_data['features'])
                offset += limit
                if len(response_data['features']) < limit:
                    break
            else:
                break

        if full_data:
            data_list = [
                {
                    'Date': feature['properties']['DATE'],
                    'value': feature['properties'].get(property_name)
                }
                for feature in full_data
            ]

            data_df = pd.DataFrame(data_list)
            data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')
            data_df['Date'] = pd.to_datetime(data_df['Date']).dt.date.astype(str)

            for date, value in zip(data_df['Date'], data_df['value']):
                if date in date_index_dict:
                    date_index = date_index_dict[date]
                    existing_value = data_dict[station_number][date_index]
                    if pd.isna(value):
                        continue
                    if pd.isna(existing_value):
                        data_dict[station_number][date_index] = value
                    elif existing_value != value:
                        raise ValueError(f"Conflicting {data_type} values for station {station_number} on {date}.")

        # Add every station, including those with no returned observations.
        combined_df[station_number] = data_dict[station_number]

    if data_type == 'level':
        combined_df['value'] = combined_df[station_numbers[0]]
    else:
        combined_df['value'] = combined_df[station_numbers].sum(axis=1, min_count=len(station_numbers))
    combined_df = combined_df[['Date', 'value']]

    return combined_df

def merge_hydrometric_data(level_station_number, inflow_station_numbers, outflow_station_numbers, start_date, end_date, csv_id, folder_path, output_path):
    """Write observations using the historical level-to-storage alignment.

    The conversion still assumes the maximum observed level corresponds to full
    reservoir level (dam height). Verify the gauge and geometry vertical datums
    physically before relying on these storage estimates; this function does
    not establish that alignment from surveyed elevations.
    """
    storage_df = fetch_hydrometric_data_ca([level_station_number], start_date, end_date, data_type='level')
    storage_df = storage_df.rename(columns={'value': 'Storage(mcm)'})

    if inflow_station_numbers:
        inflow_df = fetch_hydrometric_data_ca(inflow_station_numbers, start_date, end_date, data_type='discharge')
        inflow_df = inflow_df.rename(columns={'value': 'Inflow(cms)'})
    else:
        inflow_df = None

    outflow_df = fetch_hydrometric_data_ca(outflow_station_numbers, start_date, end_date, data_type='discharge')
    outflow_df = outflow_df.rename(columns={'value': 'Outflow(cms)'})

    merged_df = storage_df.copy()

    if inflow_df is not None:
        merged_df = pd.merge(merged_df, inflow_df, on='Date', how='outer')

    merged_df = pd.merge(merged_df, outflow_df, on='Date', how='outer')

    max_level = merged_df['Storage(mcm)'].max()
    for i, row in merged_df.iterrows():
        if not pd.isna(row['Storage(mcm)']):
            depth = max_level - row['Storage(mcm)']
            _, volume = get_area_volume_from_csv(csv_id, depth, folder_path)
            merged_df.at[i, 'Storage(mcm)'] = volume

    merged_df.to_excel(os.path.join(output_path, f"{csv_id}_inf_sto_out.xlsx"), index=False)
