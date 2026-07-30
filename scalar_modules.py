import io

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# Load a CSV file into a pandas DataFrame.
def load_data(file):
    """Read a CSV file into a pandas DataFrame."""
    try:
        # Read the CSV file using pandas.
        return pd.read_csv(file)
    except FileNotFoundError as exc:
        print(f"Error: File not found - {exc}")
        return None
    except Exception as exc:
        print(f"Error loading data: {exc}")
        return None


# Find all numeric columns in the DataFrame.
def get_numeric_columns(df):
    """Return a list of numeric column names from a DataFrame."""
    try:
        # Keep columns whose data type is numeric.
        return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    except Exception as exc:
        print(f"Error identifying numeric columns: {exc}")
        return []


# Find all categorical or text columns in the DataFrame.
def get_categorical_columns(df):
    """Return a list of categorical or text column names from a DataFrame."""
    try:
        # Keep object/string/categorical columns.
        return [
            col
            for col in df.columns
            if pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
        ]
    except Exception as exc:
        print(f"Error identifying categorical columns: {exc}")
        return []


# Encode selected categorical columns using one-hot encoding.
def one_hot_encode(df, columns):
    """Apply OneHotEncoder to selected categorical columns and return a transformed DataFrame."""
    try:
        # Check that a valid DataFrame was provided.
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("A valid DataFrame is required.")

        if not columns:
            return df.copy()

        # Keep only columns that actually exist in the DataFrame.
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            return df.copy()

        # Separate the selected columns from the rest of the DataFrame.
        remaining_df = df.drop(columns=valid_columns)

        try:
            # Use the newer sklearn API when available.
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # Fallback for older sklearn versions.
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        # Transform the selected columns into binary features.
        encoded_array = encoder.fit_transform(df[valid_columns])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(valid_columns),
            index=df.index,
        )

        # Combine the encoded columns with the remaining DataFrame.
        return pd.concat([remaining_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    except Exception as exc:
        print(f"Error during one-hot encoding: {exc}")
        return None


# Standardize selected numeric columns using z-scores.
def standard_scale(df, columns):
    """Apply StandardScaler to selected numeric columns and return the transformed DataFrame and scaler."""
    try:
        # Check that a valid DataFrame was provided.
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("A valid DataFrame is required.")

        if not columns:
            return df.copy(), None

        # Keep only columns that actually exist in the DataFrame.
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            return df.copy(), None

        # Fit the scaler and transform the selected columns.
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[valid_columns])
        scaled_df = pd.DataFrame(scaled_values, columns=valid_columns, index=df.index)

        # Replace the original columns with the scaled values.
        transformed_df = df.copy()
        transformed_df[valid_columns] = scaled_df[valid_columns]
        return transformed_df, scaler
    except Exception as exc:
        print(f"Error during standard scaling: {exc}")
        return None, None


# Scale selected numeric columns to a fixed range between 0 and 1.
def minmax_scale(df, columns):
    """Apply MinMaxScaler to selected numeric columns and return the transformed DataFrame and scaler."""
    try:
        # Check that a valid DataFrame was provided.
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("A valid DataFrame is required.")

        if not columns:
            return df.copy(), None

        # Keep only columns that actually exist in the DataFrame.
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            return df.copy(), None

        # Fit the scaler and transform the selected columns.
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df[valid_columns])
        scaled_df = pd.DataFrame(scaled_values, columns=valid_columns, index=df.index)

        # Replace the original columns with the scaled values.
        transformed_df = df.copy()
        transformed_df[valid_columns] = scaled_df[valid_columns]
        return transformed_df, scaler
    except Exception as exc:
        print(f"Error during min-max scaling: {exc}")
        return None, None


# Prepare a DataFrame for download as a CSV file.
def download_csv(df):
    """Convert a DataFrame to CSV bytes suitable for download."""
    try:
        if df is None:
            return b""

        # Write the DataFrame to an in-memory buffer as CSV data.
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode("utf-8")
    except Exception as exc:
        print(f"Error preparing CSV download: {exc}")
        return b""
