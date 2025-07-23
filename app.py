import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Plotly Navigator", layout="wide", initial_sidebar_state="expanded")

# --- 1. CACHED DATA LOADER ---
@st.cache_data
def load_data(uploaded_file):
    """
    Loads data from an uploaded CSV or Excel file.
    Uses st.cache_data to prevent re-loading on every rerun.
    """
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        # parse all possible dates for Excel files
        return pd.read_excel(uploaded_file, parse_dates=True)
    else:
        st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
        return None

# --- 2. COLUMN TYPE DETECTION ---
def detect_columns(df):
    """
    Detects and categorizes columns into numeric, categorical, and datetime types.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # Detect datetime more robustly, including pandas datetime objects
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    return num_cols, cat_cols, dt_cols

# --- 3. CHART RENDERING FUNCTIONS (Modular) ---

def render_single_column_viz(df, col, chart_type):
    """
    Renders visualizations for a single numeric column.
    Supports Histogram, Box Plot, ECDF, and Violin Plot.
    """
    st.subheader(f"Exploring: {col} ({chart_type})")

    data = df[col].dropna()
    if data.empty:
        st.warning(f"No non-missing data to plot for '{col}'.")
        return

    fig = None
    interpretation = ""
    plotly_code = ""

    if chart_type == "Histogram":
        bins = st.slider(f"Number of bins for {col}", min_value=5, max_value=50, value=20, key=f"hist_bins_{col}")
        fig = px.histogram(data, x=col, nbins=bins, title=f"Histogram of {col}")
        interpretation = (
            "This histogram shows the distribution of values in your data. "
            "Tall bars indicate ranges with many data points, while short bars or gaps mean fewer points. "
            "Look for peaks (modes) and tails (skewness) to understand the shape of your data."
        )
        plotly_code = f"px.histogram(df['{col}'], nbins={bins})"

    elif chart_type == "Box Plot":
        fig = px.box(data, y=col, title=f"Box Plot of {col}")
        fig.update_traces(orientation='v') # Ensure vertical for single column
        interpretation = (
            "The box plot summarizes the distribution. The box represents the middle 50% of the data (Interquartile Range - IQR), "
            "with the line inside being the median. The 'whiskers' extend to typical min/max values, and individual points "
            "outside the whiskers are considered outliers."
        )
        plotly_code = f"px.box(df['{col}'], y='{col}')"

    elif chart_type == "ECDF":
        fig = px.ecdf(data, x=col, title=f"ECDF of {col}")
        interpretation = (
            "The Empirical Cumulative Distribution Function (ECDF) shows the proportion of data points "
            "that are less than or equal to a given value. For example, to find the 50th percentile (median), "
            "find where the curve crosses the 0.5 mark on the y-axis."
        )
        plotly_code = f"px.ecdf(df['{col}'], x='{col}')"

    elif chart_type == "Violin Plot":
        fig = px.violin(data, y=col, title=f"Violin Plot of {col}")
        fig.update_traces(orientation='v') # Ensure vertical for single column
        interpretation = (
            "A violin plot combines a box plot with a kernel density estimate (KDE) to show the distribution shape. "
            "The wider sections indicate higher density of data points, similar to a smoothed histogram."
        )
        plotly_code = f"px.violin(df['{col}'], y='{col}')"

    if fig:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Layman's Interpretation"):
            st.markdown(interpretation)
        with st.expander("Plotly Code Snippet"):
            st.code(plotly_code, language="python")

def render_categorical_composition_viz(df, cat_col, num_cols):
    """
    Renders visualizations for the composition of a categorical column, primarily Pie Charts.
    """
    st.subheader(f"Composition: {cat_col}")

    if not cat_col:
        st.info("Please select a categorical column.")
        return

    # Allow user to select a numeric column for values, or default to count
    value_col_options = ['Count of Records'] + num_cols
    selected_value_col = st.selectbox(
        f"Slice sizes by:",
        value_col_options,
        key=f"pie_value_col_{cat_col}"
    )

    fig = None
    interpretation = ""
    plotly_code = ""

    if selected_value_col == 'Count of Records':
        # Count occurrences of each category
        counts_df = df[cat_col].value_counts().reset_index()
        counts_df.columns = [cat_col, 'Count']
        fig = px.pie(
            counts_df,
            names=cat_col,
            values='Count',
            title=f"Distribution of {cat_col}"
        )
        interpretation = (
            "This pie chart shows the proportion of each category within the selected column. "
            "Each slice represents a category, and its size corresponds to the number of records in that category. "
            "Use it to quickly see the most frequent categories."
        )
        plotly_code = (
            f"counts_df = df['{cat_col}'].value_counts().reset_index()\n"
            f"counts_df.columns = ['{cat_col}', 'Count']\n"
            f"px.pie(counts_df, names='{cat_col}', values='Count')"
        )
    else:
        # Aggregate the selected numeric column by the categorical column
        agg_df = df.groupby(cat_col)[selected_value_col].sum().reset_index()
        fig = px.pie(
            agg_df,
            names=cat_col,
            values=selected_value_col,
            title=f"Sum of {selected_value_col} by {cat_col}"
        )
        interpretation = (
            f"This pie chart shows the sum of '{selected_value_col}' for each category in '{cat_col}'. "
            "Each slice represents a category, and its size corresponds to the total value of the numeric column for that category. "
            "This helps understand how a quantitative measure is distributed across categories."
        )
        plotly_code = (
            f"agg_df = df.groupby('{cat_col}')['{selected_value_col}'].sum().reset_index()\n"
            f"px.pie(agg_df, names='{cat_col}', values='{selected_value_col}')"
        )

    if fig:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Layman's Interpretation"):
            st.markdown(interpretation)
        with st.expander("Plotly Code Snippet"):
            st.code(plotly_code, language="python")

def render_two_column_viz(df, x_col, y_col, chart_type, all_cols):
    """
    Renders visualizations for the relationship between two columns.
    Supports Scatter, Box Plot by Category, Violin Plot by Category, and Bar Chart (aggregated).
    """
    st.subheader(f"Relationship: {x_col} vs {y_col} ({chart_type})")

    data = df[[x_col, y_col]].dropna()
    if data.empty:
        st.warning(f"No non-missing data to plot for '{x_col}' and '{y_col}'.")
        return

    fig = None
    interpretation = ""
    plotly_code = ""

    if chart_type == "Scatter Plot":
        color_by = st.selectbox("Color points by (optional):", ['None'] + [c for c in all_cols if c not in [x_col, y_col]], key=f"scatter_color_{x_col}_{y_col}")
        size_by = st.selectbox("Size points by (optional, numeric):", ['None'] + [c for c in all_cols if c not in [x_col, y_col] and np.issubdtype(df[c].dtype, np.number)], key=f"scatter_size_{x_col}_{y_col}")

        hover_data = {x_col: True, y_col: True}
        if color_by != 'None':
            hover_data[color_by] = True
        if size_by != 'None':
            hover_data[size_by] = True

        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_by if color_by != 'None' else None,
            size=size_by if size_by != 'None' else None,
            title=f"Scatter Plot: {x_col} vs {y_col}",
            hover_data=hover_data
        )
        interpretation = (
            "A scatter plot shows the relationship between two variables. "
            "Look for patterns: points forming a line suggest a linear relationship, "
            "while a scattered cloud suggests no strong relationship. "
            "Color or size can reveal patterns across a third variable."
        )
        plotly_code = f"px.scatter(df, x='{x_col}', y='{y_col}', color='{color_by if color_by != 'None' else ''}', size='{size_by if size_by != 'None' else ''}')"

    elif chart_type == "Box Plot by Category":
        fig = px.box(data, x=x_col, y=y_col, title=f"Box Plot of {y_col} by {x_col}")
        interpretation = (
            "This chart compares the distribution of the numeric variable across different categories. "
            "You can easily see differences in medians, spread (box size), and outliers for each group."
        )
        plotly_code = f"px.box(df, x='{x_col}', y='{y_col}')"

    elif chart_type == "Violin Plot by Category":
        fig = px.violin(data, x=x_col, y=y_col, title=f"Violin Plot of {y_col} by {x_col}")
        interpretation = (
            "Similar to a box plot, a violin plot shows the distribution of the numeric variable for each category, "
            "but also visualizes the density of data points, giving a clearer picture of where values are concentrated."
        )
        plotly_code = f"px.violin(df, x='{x_col}', y='{y_col}')"

    elif chart_type == "Aggregated Bar Chart":
        agg_func = st.selectbox("Aggregation Function:", ["mean", "sum", "median", "count"], key=f"bar_agg_{x_col}_{y_col}")

        # Perform aggregation
        if agg_func == "count":
            agg_df = data.groupby(x_col).size().reset_index(name='count')
            fig = px.bar(agg_df, x=x_col, y='count', title=f"Count of {y_col} by {x_col}")
            plotly_code = f"df.groupby('{x_col}').size().reset_index(name='count')"
            plotly_code += f"\npx.bar(agg_df, x='{x_col}', y='count')"
        else:
            agg_df = data.groupby(x_col)[y_col].agg(agg_func).reset_index(name=f'{agg_func}_{y_col}')
            fig = px.bar(agg_df, x=x_col, y=f'{agg_func}_{y_col}', title=f"{agg_func.capitalize()} of {y_col} by {x_col}")
            plotly_code = f"df.groupby('{x_col}')['{y_col}'].{agg_func}().reset_index(name='{agg_func}_{y_col}')"
            plotly_code += f"\npx.bar(agg_df, x='{x_col}', y='{agg_func}_{y_col}')"

        interpretation = (
            f"This bar chart shows the {agg_func} of '{y_col}' for each category in '{x_col}'. "
            "It helps to quickly compare central tendencies or totals across different groups."
        )

    if fig:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Layman's Interpretation"):
            st.markdown(interpretation)
        with st.expander("Plotly Code Snippet"):
            st.code(plotly_code, language="python")

def render_time_series_viz(df, dt_col, val_col):
    """
    Renders a line chart for time series data.
    Allows selection of aggregation granularity.
    """
    st.subheader(f"Time Series: {val_col} over {dt_col}")

    # Ensure datetime column is proper datetime objects
    df_copy = df.copy()
    df_copy[dt_col] = pd.to_datetime(df_copy[dt_col], errors='coerce')
    data = df_copy[[dt_col, val_col]].dropna()

    if data.empty:
        st.warning(f"No non-missing data to plot for '{dt_col}' and '{val_col}'.")
        return

    granularity = st.selectbox(
        "Select Time Granularity:",
        ["Day", "Month", "Year"],
        key=f"time_granularity_{dt_col}_{val_col}"
    )

    # Aggregate data based on granularity
    if granularity == "Day":
        agg_data = data.set_index(dt_col).resample('D')[val_col].mean().reset_index()
    elif granularity == "Month":
        agg_data = data.set_index(dt_col).resample('M')[val_col].mean().reset_index()
    elif granularity == "Year":
        agg_data = data.set_index(dt_col).resample('Y')[val_col].mean().reset_index()

    # Rename the aggregated value column for clarity
    agg_data.rename(columns={val_col: f'Mean {val_col}'}, inplace=True)

    fig = px.line(
        agg_data,
        x=dt_col,
        y=f'Mean {val_col}',
        title=f"Mean {val_col} by {granularity}"
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    interpretation = (
        f"This line chart shows the average of '{val_col}' over time, aggregated by {granularity}. "
        "Look for trends (upward/downward), seasonality (repeating patterns), or sudden spikes/drops."
    )
    plotly_code = (
        f"df_copy = df.copy()\n"
        f"df_copy['{dt_col}'] = pd.to_datetime(df_copy['{dt_col}'], errors='coerce')\n"
        f"data = df_copy[['{dt_col}', '{val_col}']].dropna()\n"
        f"agg_data = data.set_index('{dt_col}').resample('{granularity[0]}')[val_col].mean().reset_index()\n"
        f"px.line(agg_data, x='{dt_col}', y='Mean {val_col}')"
    )

    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Layman's Interpretation"):
        st.markdown(interpretation)
    with st.expander("Plotly Code Snippet"):
        st.code(plotly_code, language="python")

def render_multi_dim_viz(df, num_cols, cat_cols, chart_type, all_cols):
    """
    Renders multi-dimensional visualizations.
    Supports Pair Plots, Parallel Coordinates, Correlation Heatmap, and Bubble Chart.
    """
    st.subheader(f"Multi-Dimensional Analysis ({chart_type})")

    fig = None
    interpretation = ""
    plotly_code = ""

    if chart_type == "Pair Plot (Scatter Matrix)":
        selected_num_cols = st.multiselect(
            "Select numeric columns for pair plot (min 2):",
            num_cols,
            default=num_cols[:min(5, len(num_cols))], # Default to first 5 or fewer
            key="pair_plot_cols"
        )
        if len(selected_num_cols) < 2:
            st.info("Please select at least two numeric columns for a pair plot.")
            return

        color_by = st.selectbox("Color points by (optional, categorical):", ['None'] + cat_cols, key="pair_plot_color")

        fig = px.scatter_matrix(
            df,
            dimensions=selected_num_cols,
            color=color_by if color_by != 'None' else None,
            title="Pair Plot (Scatter Matrix)"
        )
        interpretation = (
            "A pair plot displays scatter plots for all pairwise combinations of selected numeric columns. "
            "It helps to quickly identify relationships between multiple variables and see how different groups (if colored) behave."
        )
        plotly_code = (
            f"px.scatter_matrix(df, dimensions={selected_num_cols}, "
            f"color='{color_by if color_by != 'None' else ''}')"
        )

    elif chart_type == "Parallel Coordinates Plot":
        selected_cols = st.multiselect(
            "Select columns for parallel coordinates (numeric and/or categorical):",
            num_cols + cat_cols,
            default=(num_cols + cat_cols)[:min(5, len(num_cols) + len(cat_cols))],
            key="parallel_coords_cols"
        )
        if len(selected_cols) < 2:
            st.info("Please select at least two columns for a parallel coordinates plot.")
            return

        color_by = st.selectbox("Color lines by (optional, numeric or categorical):", ['None'] + selected_cols, key="parallel_coords_color")

        fig = px.parallel_coordinates(
            df,
            dimensions=selected_cols,
            color=color_by if color_by != 'None' else None,
            color_continuous_scale=px.colors.sequential.Viridis if color_by != 'None' and df[color_by].dtype in np.number else None,
            title="Parallel Coordinates Plot"
        )
        interpretation = (
            "A parallel coordinates plot visualizes multi-dimensional data. Each vertical axis represents a variable, "
            "and each line represents a single data point. You can drag axes to reorder them and brush ranges to filter data, "
            "revealing clusters and patterns across many features."
        )
        plotly_code = (
            f"px.parallel_coordinates(df, dimensions={selected_cols}, "
            f"color='{color_by if color_by != 'None' else ''}')"
        )

    elif chart_type == "Correlation Heatmap":
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns for a correlation heatmap.")
            return

        corr = df[num_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f", # Show correlation values on heatmap
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            aspect="auto",
            title="Correlation Heatmap (Numeric Columns)"
        )
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
        interpretation = (
            "This heatmap shows the Pearson correlation coefficient between all pairs of numeric columns. "
            "Values close to +1 indicate a strong positive linear relationship, -1 a strong negative linear relationship, "
            "and 0 no linear relationship. Red indicates negative correlation, blue positive."
        )
        plotly_code = (
            f"corr_matrix = df[{num_cols}].corr()\n"
            f"px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu', range_color=[-1, 1])"
        )

    elif chart_type == "Bubble Chart":
        st.markdown("A Bubble Chart shows the relationship between two numeric variables (X, Y) where points are sized by a third numeric variable, and optionally colored by a fourth variable.")

        # Add a check for minimum numeric columns required for a bubble chart
        if len(num_cols) < 3:
            st.info("Bubble Chart requires at least 3 numeric columns (for X, Y, and Size). Please upload a dataset with more numeric columns or choose a different chart type.")
            return

        # Ensure that the options for selectboxes are always valid lists, even if filtered
        # Add None as a default first option to allow explicit non-selection
        available_x = ['None'] + num_cols
        x_col = st.selectbox("Select X-axis (numeric):", available_x, key="bubble_x")

        # Filter available options for Y and Size based on X and Y selections
        available_y = ['None'] + [c for c in num_cols if c != x_col and c != 'None'] # Exclude 'None' from actual columns
        y_col = st.selectbox("Select Y-axis (numeric):", available_y, key="bubble_y")

        available_size = ['None'] + [c for c in num_cols if c != x_col and c != y_col and c != 'None'] # Exclude 'None' from actual columns
        size_col = st.selectbox("Select Size by (numeric):", available_size, key="bubble_size")

        available_color = ['None'] + [c for c in all_cols if c != x_col and c != y_col and c != size_col and c != 'None'] # Exclude 'None' from actual columns
        color_col = st.selectbox("Color by (optional):", available_color, key="bubble_color")

        # Now, explicitly check if any required column is 'None' (meaning not selected by user or no valid options)
        if x_col == 'None' or y_col == 'None' or size_col == 'None':
            st.info("Please select valid X, Y, and Size columns for the Bubble Chart.")
            return

        # Corrected: Use np.issubdtype for type checking
        if not (np.issubdtype(df[x_col].dtype, np.number) and np.issubdtype(df[y_col].dtype, np.number) and np.issubdtype(df[size_col].dtype, np.number)):
            st.warning("X, Y, and Size columns must be numeric for a Bubble Chart.")
            return

        hover_data = {x_col: True, y_col: True, size_col: True}
        if color_col != 'None':
            hover_data[color_col] = True

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col if color_col != 'None' else None,
            hover_data=hover_data,
            title=f"Bubble Chart: {x_col} vs {y_col} (Size: {size_col})"
        )
        interpretation = (
            "This bubble chart visualizes the relationship between the X and Y axes, with the size of each bubble "
            "representing a third numeric variable. If a color variable is selected, it adds a fourth dimension. "
            "Look for clusters, trends, and how the size/color of points vary across the plot."
        )
        plotly_code = (
            f"px.scatter(df, x='{x_col}', y='{y_col}', size='{size_col}', "
            f"color='{color_col if color_col != 'None' else ''}')"
        )

    if fig:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Layman's Interpretation"):
            st.markdown(interpretation)
        with st.expander("Plotly Code Snippet"):
            st.code(plotly_code, language="python")

# --- MAIN APP FUNCTION ---
def main():
    st.title("📊 Plotly Navigator: Interactive Viz Builder")

    st.markdown("""
    Welcome to the **Plotly Navigator**! 🚀 This app helps you visualize your data step-by-step using the powerful Plotly library.
    No coding required—just upload your data and let the Navigator guide you!

    **How to get started:**
    1.  **Upload** your CSV or Excel file below.
    2.  **Choose** what kind of visualization you want to create from the sidebar.
    3.  **Select** your columns and customize the chart.
    4.  **Explore** the interactive plots and their interpretations!
    """)
    st.write("---")

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if not uploaded_file:
        st.info("Please upload a file to begin your visualization journey.")
        return

    try:
        df = load_data(uploaded_file)
        if df is None: # Handle unsupported file type from load_data
            return
    except Exception as e:
        st.error(f"Failed to load data: {e}. Please ensure it's a valid CSV or Excel file.")
        return

    num_cols, cat_cols, dt_cols = detect_columns(df)
    all_cols = df.columns.tolist()

    if df.empty:
        st.warning("The uploaded file is empty or contains no valid data after parsing.")
        return

    st.sidebar.header("Choose Your Visualization Goal")
    viz_goal = st.sidebar.radio(
        "What do you want to visualize?",
        [
            "1. Explore a Single Column's Distribution",
            "1.5. Show Proportions of Categories", # New goal for Pie Chart
            "2. Show Relationship Between Two Columns",
            "3. Track Trends Over Time",
            "4. Visualize Multi-Dimensional Relationships"
        ],
        index=0 # Default to first option
    )

    st.write("---") # Separator for main content

    # --- Conditional Rendering based on Visualization Goal ---

    if viz_goal == "1. Explore a Single Column's Distribution":
        st.header("1. Explore a Single Column's Distribution")
        if not num_cols:
            st.info("No numeric columns detected to explore their distribution.")
            return

        selected_col = st.selectbox(
            "Select a numeric column to analyze:",
            num_cols,
            key="single_col_select"
        )
        if selected_col:
            chart_type = st.radio(
                "Choose chart type:",
                ["Histogram", "Box Plot", "ECDF", "Violin Plot"],
                key="single_chart_type"
            )
            render_single_column_viz(df, selected_col, chart_type)

    elif viz_goal == "1.5. Show Proportions of Categories": # New section for Pie Chart
        st.header("1.5. Show Proportions of Categories")
        if not cat_cols:
            st.info("No categorical columns detected to show proportions.")
            return

        selected_cat_col = st.selectbox(
            "Select a categorical column to analyze:",
            cat_cols,
            key="pie_cat_select"
        )
        if selected_cat_col:
            # For now, only Pie Chart is offered here, but could add others like Bar Chart (counts)
            render_categorical_composition_viz(df, selected_cat_col, num_cols)

    elif viz_goal == "2. Show Relationship Between Two Columns":
        st.header("2. Show Relationship Between Two Columns")
        if len(all_cols) < 2:
            st.info("Need at least two columns to show relationships.")
            return

        x_col = st.selectbox("Select X-axis column:", all_cols, key="two_col_x")
        y_col = st.selectbox("Select Y-axis column:", [c for c in all_cols if c != x_col], key="two_col_y")

        if x_col and y_col:
            # Determine suitable chart types based on column types
            x_is_num = x_col in num_cols
            y_is_num = y_col in num_cols
            x_is_cat = x_col in cat_cols
            y_is_cat = y_col in cat_cols

            possible_charts = []
            if x_is_num and y_is_num:
                possible_charts.append("Scatter Plot")
            if (x_is_cat and y_is_num) or (x_is_num and y_is_cat):
                possible_charts.extend(["Box Plot by Category", "Violin Plot by Category", "Aggregated Bar Chart"])
            # Add more combinations as needed (e.g., two categorical)

            if not possible_charts:
                st.info(f"No standard Plotly chart suggestions for '{x_col}' and '{y_col}' combination. Try different columns.")
                return

            chart_type = st.radio(
                "Choose chart type:",
                possible_charts,
                key="two_col_chart_type"
            )
            render_two_column_viz(df, x_col, y_col, chart_type, all_cols)

    elif viz_goal == "3. Track Trends Over Time":
        st.header("3. Track Trends Over Time")
        if not dt_cols:
            st.info("No datetime columns detected to track trends over time.")
            return
        if not num_cols:
            st.info("No numeric columns detected to plot values over time.")
            return

        selected_dt_col = st.selectbox("Select a datetime column:", dt_cols, key="time_dt_col")
        selected_val_col = st.selectbox("Select a numeric column to track:", num_cols, key="time_val_col")

        if selected_dt_col and selected_val_col:
            render_time_series_viz(df, selected_dt_col, selected_val_col)

    elif viz_goal == "4. Visualize Multi-Dimensional Relationships":
        st.header("4. Visualize Multi-Dimensional Relationships")
        # Ensure enough columns for multi-dimensional plots
        if len(num_cols) + len(cat_cols) < 2:
            st.info("Need at least two columns (numeric or categorical) for multi-dimensional analysis.")
            return

        # Add Bubble Chart to options
        multi_dim_chart_options = ["Pair Plot (Scatter Matrix)", "Parallel Coordinates Plot", "Correlation Heatmap", "Bubble Chart"]

        chart_type = st.radio(
            "Choose multi-dimensional chart type:",
            multi_dim_chart_options,
            key="multi_dim_chart_type"
        )

        render_multi_dim_viz(df, num_cols, cat_cols, chart_type, all_cols)

if __name__ == "__main__":
    main()
