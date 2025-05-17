import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Enhanced UI Configuration
st.set_page_config(
    page_title="Taal Lake Water Quality Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Improved Container Styling */
    .stApp {
        background-color: #f0f6fc;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Enhancements */
    .css-1aumxhk {
        background-color: white;
        border-right: 1px solid #e0e4e8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Card-like Containers */
    .stContainer, .stDataFrame {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Colorful Metric Boxes */
    .metric-container {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Buttons and Inputs */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    /* Plotly Chart Improvements */
    .plotly-chart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("Water quality.xlsx")

df = load_data()

# Sidebar Navigation
with st.sidebar:
    st.image("Taal lake Wa.png", use_container_width=True)
    st.markdown("## 💧 Taal Lake Monitoring")

    selected = option_menu(
    menu_title=None,
    options=["Overview", "Time Series", "Correlations", "Relationships", "Predictions", "Developer Info"],
    icons=["house-fill", "graph-up", "link", "diagram-3-fill", "magic", "info-circle"],  # <- updated
    default_index=0,
    styles={
        "container": {
            "padding": "10px",
            "background-color": "transparent"
        },
        "icon": {"color": "#2563EB", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "5px",
            "transition": "all 0.3s",
            "border-radius": "10px"
        },
        "nav-link-selected": {
            "background-color": "#2563EB",
            "color": "white"
        },
    }
)


# Date Range Filter
if not df.empty:
    date_col = next((col for col in df.columns if "date" in col.lower() or "time" in col.lower()), None)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()

        with st.sidebar:
            st.markdown("### 📅 Date Range")
            use_full_range = st.checkbox("Use Full Date Range", value=True)

            if not use_full_range:
                date_range = st.date_input(
                    "Select Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date

            # Filter the dataframe
            mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))
            df = df.loc[mask]

# Overview Page
if selected == "Overview":
    st.title("📊 Water Quality Overview")

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>Total Records</h3>
            <h1>{}</h1>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>Parameters</h3>
            <h1>{}</h1>
        </div>
        """.format(len(df.columns) - 1), unsafe_allow_html=True)

    with col3:
        # Use the detected date column instead of assuming first column is date
        if date_col:
            start_date = df[date_col].min().strftime('%Y-%m-%d')
        else:
            start_date = "N/A"

        st.markdown("""
        <div class="metric-container">
            <h3>Start Date</h3>
            <h1>{}</h1>
        </div>
        """.format(start_date), unsafe_allow_html=True)

    with col4:
        # Use the detected date column instead of assuming first column is date
        if date_col:
            end_date = df[date_col].max().strftime('%Y-%m-%d')
        else:
            end_date = "N/A"

        st.markdown("""
        <div class="metric-container">
            <h3>End Date</h3>
            <h1>{}</h1>
        </div>
        """.format(end_date), unsafe_allow_html=True)

    # Raw Data Display with options
    st.subheader("📋 Complete Raw Data")

    # Add data view options
    data_view_options = st.radio(
        "Data Display Options:",
        ["View All Data", "Filter by Column", "Search Records"],
        horizontal=True
    )

    if data_view_options == "View All Data":
        # Show the complete dataframe with pagination
        st.dataframe(df, use_container_width=True, height=500)

        # Add download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Data as CSV",
            data=csv,
            file_name="taal_lake_water_quality_data.csv",
            mime="text/csv",
        )

    elif data_view_options == "Filter by Column":
        # Let users select columns to display
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select columns to display:", all_columns, default=all_columns[:5])

        if selected_columns:
            st.dataframe(df[selected_columns], use_container_width=True, height=500)
        else:
            st.info("Please select at least one column to display data.")

    elif data_view_options == "Search Records":
        # Simple search functionality
        search_col = st.selectbox("Select column to search:", df.columns.tolist())

        if df[search_col].dtype == 'object':  # For text columns
            search_term = st.text_input("Enter search term:")
            if search_term:
                filtered_df = df[df[search_col].astype(str).str.contains(search_term, case=False)]
                st.dataframe(filtered_df, use_container_width=True, height=500)
                st.write(f"Found {len(filtered_df)} matching records")
        else:  # For numeric columns
            min_val = float(df[search_col].min())
            max_val = float(df[search_col].max())
            range_val = st.slider(f"Select range for {search_col}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val))
            filtered_df = df[(df[search_col] >= range_val[0]) & (df[search_col] <= range_val[1])]
            st.dataframe(filtered_df, use_container_width=True, height=500)
            st.write(f"Found {len(filtered_df)} matching records")


# Time Series
elif selected == "Time Series":
    st.title("📈 Time Series Analysis")

    if not df.empty:
        time_col = next((col for col in df.columns if "date" in col.lower() or "time" in col.lower()), None)

        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])

            # Get numeric columns and exclude 'year' and 'month'
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            numeric_cols = [col for col in numeric_cols if col.lower() not in ['year', 'month']]

            # Initialize session state
            if "time_series_selected_params" not in st.session_state:
                st.session_state.time_series_selected_params = []

            if "select_all_clicked" not in st.session_state:
                st.session_state.select_all_clicked = False

            st.markdown("#### Select Parameters to Plot")

            # Select All checkbox
            select_all = st.checkbox("Select All Parameters")

            # Handle Select All logic
            if select_all and not st.session_state.select_all_clicked:
                st.session_state.time_series_selected_params = numeric_cols
                st.session_state.select_all_clicked = True
            elif not select_all and st.session_state.select_all_clicked:
                st.session_state.time_series_selected_params = []
                st.session_state.select_all_clicked = False

            # Parameter multiselect
            selected_params = st.multiselect(
                "Choose parameters:",
                options=numeric_cols,
                default=st.session_state.time_series_selected_params
            )

            # Sync selected parameters
            st.session_state.time_series_selected_params = selected_params

            # Plot if parameters selected
            if selected_params:
                fig = px.line(
                    df,
                    x=time_col,
                    y=selected_params,
                    labels={"value": "Measurement", "variable": "Parameter"},
                    template="simple_white"
                )

                fig.update_traces(mode="lines", line=dict(width=2))
                fig.update_layout(
                    title="Water Quality Trends Over Time",
                    title_font=dict(size=18),
                    legend_title="Parameters",
                    height=450,
                    margin=dict(t=40, l=20, r=20, b=40),
                    xaxis_title="Date",
                    yaxis_title="",
                    font=dict(family="Segoe UI, sans-serif", size=13),
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one parameter to generate the plot.")
        else:
            st.warning("⚠️ No date/time column detected in the dataset.")

# Correlations
elif selected == "Correlations":
    st.title("🔗 Correlation Heatmap")

    if not df.empty:
        # Drop 'Year' column if it exists
        if 'Year' in df.columns:
            df = df.drop(columns=['Year'])

        # Location filter using the 'Site' column
        site_options = df['Site'].dropna().unique()
        view_mode = st.radio("Select View Mode:", ["All Sites", "By Site"], horizontal=True)

        if view_mode == "By Site":
            selected_site = st.selectbox("Select Site:", sorted(site_options))
            filtered_df = df[df['Site'] == selected_site]
            st.markdown(f"#### Correlation Heatmap for {selected_site}")
        else:
            filtered_df = df.copy()
            st.markdown("#### Correlation Heatmap for All Sites")

        # Compute correlation matrix on numeric columns only
        numeric_df = filtered_df.select_dtypes(include='number')
        corr_matrix = numeric_df.corr().round(2)

        # Plot heatmap with annotations
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="Correlation Matrix",
            xaxis_title="Parameters",
            yaxis_title="Parameters",
            autosize=False,
            width=850,
            height=800,
        )
        st.plotly_chart(fig)


        # Top Correlations
        st.subheader("📊 Top Correlations")
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_corrs = upper.unstack().dropna().sort_values(ascending=False).head(5)

        cols = st.columns(5)
        for i, ((var1, var2), val) in enumerate(top_corrs.items()):
            cols[i].metric(
                f"{var1} & {var2}",
                f"{val:.2f}",
                delta_color="inverse"
            )

# Relationships
elif selected == "Relationships":
    st.title("📊 Parameter Relationships")

    if not df.empty:
        # Define the specific parameters of interest
        selected_parameters = ["pH", "Ammonia", "Nitrate", "Phosphate", "Dissolved Oxygen", "Sulfide", "Carbon Dioxide"]

        # Filter only those that exist in the dataset
        valid_params = [p for p in selected_parameters if p in df.columns]

        if len(valid_params) < 2:
            st.warning("Not enough valid parameters found in the dataset.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x = st.selectbox("X-axis Parameter", options=valid_params, index=None, placeholder="Please choose", key="x_axis_rel")
            with col2:
                # Filter out selected X from Y options
                y_options = [p for p in valid_params if p != x] if x else valid_params
                y = st.selectbox("Y-axis Parameter", options=y_options, index=None, placeholder="Please choose", key="y_axis_rel")

            if x and y:
                # Create an interactive scatter plot with Plotly
                fig = px.scatter(df, x=x, y=y, title=f"Relationship between {x} and {y}",
                                 labels={x: x, y: y},
                                 hover_data={x: True, y: True, 'index': df.index},
                                 template='plotly_white')

                # Compute and display correlation
                correlation = df[[x, y]].corr().iloc[0, 1]

                # Add correlation as a text annotation
                fig.add_annotation(
                    text=f"Correlation: {correlation:.2f}",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.1,
                    showarrow=False,
                    font=dict(size=14)
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please choose both X and Y parameters to view the relationship.")

# Predictions Page
elif selected == "Predictions":
    st.title("🔮 Predictions - CNN, LSTM & CNN-LSTM")

    # Water parameters and external factors
    water_parameters = ["pH", "Ammonia", "Nitrate", "Phosphate", "Dissolved Oxygen", "Water Temperature"]
    external_factors = ["Sulfide", "Carbon Dioxide", "Air Temperature", "Weather Condition", "Wind Direction"]

    # Dropdown to select target parameter (water parameter only)
    target_col = st.selectbox("🎯 Select Water Parameter to Predict:", water_parameters)

    # Checkbox to include external factors
    include_external = st.checkbox("Include External Factors (Sulfide, Carbon Dioxide, etc.)", value=False)

    # Dropdown to select prediction time (next week, next month, next year)
    prediction_time = st.selectbox("📅 Select Prediction Time:", ["Next Week", "Next Month", "Next Year"])

    locations = df['Site'].dropna().unique()
    selected_site = st.selectbox("📍 Select Location (Site):", ["All"] + sorted(locations.tolist()))

    if selected_site != "All":
        filtered_df = df[df['Site'] == selected_site]
    else:
        filtered_df = df.copy()

    # Selecting the data based on the selected parameters
    if include_external:
        selected_columns = water_parameters + external_factors
    else:
        selected_columns = water_parameters

    data = filtered_df[selected_columns].dropna()

    if target_col not in data.columns or len(data) < 50:
        st.warning("Not enough data for training.")
    else:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        def create_sequences(data, target_index, lookback=5):
            X, y = [], []
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i])
                y.append(data[i, target_index])
            return np.array(X), np.array(y)

        lookback = 5
        target_index = data.columns.get_loc(target_col)
        X, y = create_sequences(data_scaled, target_index, lookback)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        def evaluate_model(y_true, y_pred):
            # Flatten arrays to ensure they match in shape
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            return {
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R²": r2_score(y_true, y_pred)
            }

        def plot_predictions(y_true, y_pred, title):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(y_true, label="Actual", color='black')
            ax.plot(y_pred, label="Predicted", linestyle='--')
            ax.set_title(title)
            ax.legend()
            st.pyplot(fig)

        # Button to trigger the prediction process
        if st.button("Start Prediction"):
            # CNN Model
            cnn_model = Sequential([
                Conv1D(64, kernel_size=2, activation='relu', input_shape=(lookback, X.shape[2])),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(50, activation='relu'),
                Dense(1)
            ])
            cnn_model.compile(optimizer='adam', loss='mse')
            cnn_model.fit(X_train, y_train, epochs=50, verbose=0)
            y_pred_cnn = cnn_model.predict(X_test).flatten()
            cnn_metrics = evaluate_model(y_test, y_pred_cnn)

            # LSTM Model
            lstm_model = Sequential([
                LSTM(64, activation='tanh', input_shape=(lookback, X.shape[2])),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_train, y_train, epochs=50, verbose=0)
            y_pred_lstm = lstm_model.predict(X_test).flatten()
            lstm_metrics = evaluate_model(y_test, y_pred_lstm)

            # CNN-LSTM Model
            cnn_lstm_model = Sequential([
                Conv1D(64, kernel_size=2, activation='relu', input_shape=(lookback, X.shape[2])),
                MaxPooling1D(pool_size=2),
                LSTM(64, activation='tanh'),
                Dense(1)
            ])
            cnn_lstm_model.compile(optimizer='adam', loss='mse')
            cnn_lstm_model.fit(X_train, y_train, epochs=50, verbose=0)
            y_pred_cnn_lstm = cnn_lstm_model.predict(X_test).flatten()
            cnn_lstm_metrics = evaluate_model(y_test, y_pred_cnn_lstm)

            # Display the evaluation results and plots
            st.markdown("### 📊 Performance Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 🧐 CNN")
                st.write(cnn_metrics)
                plot_predictions(y_test, y_pred_cnn, "CNN: Actual vs Predicted")

            with col2:
                st.markdown("#### 🔁 LSTM")
                st.write(lstm_metrics)
                plot_predictions(y_test, y_pred_lstm, "LSTM: Actual vs Predicted")

            with col3:
                st.markdown("#### 🧚 CNN-LSTM")
                st.write(cnn_lstm_metrics)
                plot_predictions(y_test, y_pred_cnn_lstm, "CNN-LSTM: Actual vs Predicted")

            # Prepare predictions for download
            predictions_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted (CNN)": y_pred_cnn,
                "Predicted (LSTM)": y_pred_lstm,
                "Predicted (CNN-LSTM)": y_pred_cnn_lstm,
                "Prediction Time": prediction_time
            })

            # Download button for predictions
            st.download_button(
                label="Download Predictions as CSV",
                data=predictions_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )


# Developer Info
elif selected == "Developer Info":
    st.title("👨‍💻 Developer Information")
    st.markdown("""
This interactive dashboard helps visualize and analyze **Taal Lake's water quality** data.

**🔧 Technologies Used:**
- Python
- Streamlit
- Pandas, Matplotlib, Plotly

**📍 Purpose:**
- Aid researchers and environmental engineers in tracking water quality trends.
- Support evidence-based environmental decisions.

**📁 Data Source:**
- Collected from Taal Lake monitoring stations
""")

    # Developer data
    devs = [
        {
            "name": "Clark Patrick G. Agravante",
            "role": "Project Lead / Full-Stack Developer",
            "desc": "Coordinates the team and ensures the quality of the overall application.",
            "image": "dev1.jpg"
        },
        {
            "name": "Lebron James G. Larido",
            "role": "Backend Developer",
            "desc": "Developed the backend data pipelines and API integrations.",
            "image": "dev2.jpg"
        },
        {
            "name": "Nel Johnceen Pulido",
            "role": "Data Analyst",
            "desc": "Analyzed the water quality data and generated visual insights.",
            "image": "dev3.jpg"
        },
        {
            "name": "Johndel M. Orosco",
            "role": "UI/UX Designer",
            "desc": "Designed the dashboard interface and user experience.",
            "image": "dev4.jpg"
        },
        {
            "name": "Carl Louise Sambrano",
            "role": "Frontend Developer",
            "desc": "Implemented the frontend components and visualizations.",
            "image": "dev5.jpg"
        },
        {
            "name": "Precious Erica G. Sueño",
            "role": "Documentation & Deployment",
            "desc": "Handled deployment, packaging, and writing user documentation.",
            "image": "dev6.png"
        },
    ]

    # Display 3 developers per row
    for i in range(0, len(devs), 3):
        cols = st.columns(3)
        for j, dev in enumerate(devs[i:i+3]):
            with cols[j]:
                if dev["image"]:
                    st.image(dev["image"], width=150)
                st.markdown(f"**{dev['name']}**  \n*{dev['role']}*")
                st.caption(dev["desc"])

# Enhanced Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2025 Taal Lake Water Quality Dashboard | Sustainable Monitoring Initiative</p>",
    unsafe_allow_html=True
)
