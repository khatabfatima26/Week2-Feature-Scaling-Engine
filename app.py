import streamlit as st

from scalar_modules import (
    download_csv,
    get_categorical_columns,
    get_numeric_columns,
    load_data,
    minmax_scale,
    one_hot_encode,
    standard_scale,
)

# Configure the Streamlit page for a polished dashboard experience.
st.set_page_config(page_title="Feature Scaling Engine", page_icon="⚙️", layout="wide")

# Initialize session state so uploaded and transformed data persist between interactions.
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = ""


# ---------------------------
# App Header and Sidebar
# ---------------------------
st.title("Feature Scaling Engine - ML Data Preprocessing")
st.markdown("Upload a CSV file and prepare it for machine learning with encoding and scaling tools.")

with st.sidebar:
    st.header("ℹ️ About this App")
    st.markdown(
        "This app helps you inspect a dataset, identify numeric and categorical columns, "
        "and apply preprocessing steps such as one-hot encoding and scaling."
    )
    st.info("Tip: Upload a CSV file to begin preprocessing.")


# ---------------------------
# File Upload Section
# ---------------------------
with st.container():
    st.header("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Validate the uploaded file type and load it into a DataFrame.
        if uploaded_file.name.lower().endswith(".csv"):
            try:
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.raw_df = df
                    st.session_state.processed_df = df.copy()
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.success(f"Loaded {uploaded_file.name} successfully.")
                else:
                    st.error("The file could not be loaded. Please try another file.")
            except Exception as exc:
                st.error(f"Unexpected error while loading the file: {exc}")
        else:
            st.error("Please upload a valid CSV file.")
    elif st.session_state.last_uploaded_name:
        st.info(f"Current dataset: {st.session_state.last_uploaded_name}")
    else:
        st.warning("No file uploaded yet. Please upload a CSV file to continue.")


# ---------------------------
# Data Preview Section
# ---------------------------
if st.session_state.raw_df is not None:
    st.header("👀 Raw Data Preview")
    st.dataframe(st.session_state.raw_df.head(100), use_container_width=True)
    st.caption(
        f"Shape: {st.session_state.raw_df.shape[0]} rows x {st.session_state.raw_df.shape[1]} columns"
    )


# ---------------------------
# Column Analysis Section
# ---------------------------
if st.session_state.raw_df is not None:
    st.header("📊 Column Analysis")
    numeric_columns = get_numeric_columns(st.session_state.raw_df)
    categorical_columns = get_categorical_columns(st.session_state.raw_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Numeric Columns")
        if numeric_columns:
            st.write(numeric_columns)
        else:
            st.info("No numeric columns found.")

    with col2:
        st.subheader("Categorical Columns")
        if categorical_columns:
            st.write(categorical_columns)
        else:
            st.info("No categorical columns found.")


# ---------------------------
# One-Hot Encoding Section
# ---------------------------
if st.session_state.processed_df is not None:
    st.header("🔀 One-Hot Encoding")
    categorical_columns = get_categorical_columns(st.session_state.processed_df)

    selected_categories = st.multiselect(
        "Select categorical columns to encode",
        options=categorical_columns,
        default=[],
    )

    if st.button("Apply One-Hot Encoding"):
        if not selected_categories:
            st.warning("Please select at least one categorical column.")
        else:
            try:
                encoded_df = one_hot_encode(st.session_state.processed_df, selected_categories)
                if encoded_df is not None:
                    st.session_state.processed_df = encoded_df
                    st.success("One-hot encoding applied successfully.")
                    st.dataframe(encoded_df.head(100), use_container_width=True)
                else:
                    st.error("Encoding failed. Please check your selected columns.")
            except Exception as exc:
                st.error(f"Encoding error: {exc}")


# ---------------------------
# Standard Scaler Section
# ---------------------------
if st.session_state.processed_df is not None:
    st.header("📏 Standard Scaler")
    numeric_columns = get_numeric_columns(st.session_state.processed_df)

    selected_numeric_standard = st.multiselect(
        "Select numeric columns to standardize",
        options=numeric_columns,
        default=[],
        key="standard_columns",
    )

    if st.button("Apply Standard Scaling"):
        if not selected_numeric_standard:
            st.warning("Please select at least one numeric column.")
        else:
            try:
                before_df = st.session_state.processed_df[selected_numeric_standard]
                scaled_df, scaler = standard_scale(st.session_state.processed_df, selected_numeric_standard)
                if scaled_df is not None:
                    st.session_state.processed_df = scaled_df
                    st.success("Standard scaling applied successfully.")

                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.subheader("Before")
                        st.dataframe(before_df.head(50), use_container_width=True)
                    with col_after:
                        st.subheader("After")
                        st.dataframe(scaled_df[selected_numeric_standard].head(50), use_container_width=True)
                    st.caption(f"Scaler fitted: {type(scaler).__name__}")
                else:
                    st.error("Standard scaling failed.")
            except Exception as exc:
                st.error(f"Standard scaling error: {exc}")


# ---------------------------
# MinMax Scaler Section
# ---------------------------
if st.session_state.processed_df is not None:
    st.header("📐 MinMax Scaler")
    numeric_columns = get_numeric_columns(st.session_state.processed_df)

    selected_numeric_minmax = st.multiselect(
        "Select numeric columns to scale to [0, 1]",
        options=numeric_columns,
        default=[],
        key="minmax_columns",
    )

    if st.button("Apply MinMax Scaling"):
        if not selected_numeric_minmax:
            st.warning("Please select at least one numeric column.")
        else:
            try:
                before_df = st.session_state.processed_df[selected_numeric_minmax]
                scaled_df, scaler = minmax_scale(st.session_state.processed_df, selected_numeric_minmax)
                if scaled_df is not None:
                    st.session_state.processed_df = scaled_df
                    st.success("MinMax scaling applied successfully.")

                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.subheader("Before")
                        st.dataframe(before_df.head(50), use_container_width=True)
                    with col_after:
                        st.subheader("After")
                        st.dataframe(scaled_df[selected_numeric_minmax].head(50), use_container_width=True)
                    st.caption(f"Scaler fitted: {type(scaler).__name__}")
                else:
                    st.error("MinMax scaling failed.")
            except Exception as exc:
                st.error(f"MinMax scaling error: {exc}")


# ---------------------------
# Download Section
# ---------------------------
if st.session_state.processed_df is not None:
    st.header("⬇️ Download Processed Data")
    csv_bytes = download_csv(st.session_state.processed_df)

    if csv_bytes:
        st.download_button(
            label="Download processed CSV",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
        )
    else:
        st.info("The processed dataset is empty or could not be prepared for download.")
