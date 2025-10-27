import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(
    page_title="F1 Race Predictor V2",
    page_icon="🏎️",
    layout="wide"
)


# Load model and data
@st.cache_resource
def load_model():
    with open('models/f1_predictor_v2_optimized.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/f1_dataset_features_v2_complete.csv')
    return df


model = load_model()
df = load_data()

# Get 2025 current standings
latest_round = df[df['season'] == 2025]['round'].max()
current_drivers = df[
    (df['season'] == 2025) &
    (df['round'] == latest_round)  # Only drivers from most recent race
].groupby('driverCode').agg({
    'driver_season_points': 'last',
    'driver_last5_avg_points': 'last',
    'constructor_season_points': 'last',
    'constructorName': 'last'
}).reset_index().sort_values('driver_season_points', ascending=False).head(20)  # Limit to 20


# Circuit mapping
CIRCUITS = {
    'Mexico City': {'round': 20, 'length_km': 4.304, 'downforce': 0, 'overtaking': 2, 'tire_wear': 1},
    'São Paulo (Brazil)': {'round': 21, 'length_km': 4.309, 'downforce': 2, 'overtaking': 2, 'tire_wear': 1},
    'Las Vegas': {'round': 22, 'length_km': 6.120, 'downforce': 0, 'overtaking': 2, 'tire_wear': 0},
    'Qatar': {'round': 23, 'length_km': 5.380, 'downforce': 2, 'overtaking': 2, 'tire_wear': 2},
    'Abu Dhabi': {'round': 24, 'length_km': 5.281, 'downforce': 2, 'overtaking': 2, 'tire_wear': 1},
}

# Title
st.title("🏎️ F1 Race Winner Predictor V2")
st.markdown("**Predict Top 3 podium finishers for the next race**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Info")
    st.metric("Accuracy", "91.29%")
    st.metric("ROC AUC", "0.947")
    st.metric("Features", "33")

    st.markdown("---")
    st.markdown("### 🏆 Current Top 5 Drivers")
    top5 = current_drivers.head(5)
    for idx, row in top5.iterrows():
        st.metric(row['driverCode'], f"{int(row['driver_season_points'])} pts")

# Main content
st.header("🔮 Predict Next Race")

col1, col2 = st.columns([2, 1])

with col1:
    circuit = st.selectbox(
        "**Select Upcoming Race Circuit**",
        options=list(CIRCUITS.keys()),
        index=0
    )

    circuit_info = CIRCUITS[circuit]
    st.info(f"📍 **{circuit}** | Round {circuit_info['round']} | {circuit_info['length_km']} km")

with col2:
    st.markdown("### Grid Position")
    grid_scenario = st.radio(
        "Qualifying assumption:",
        ["Use Championship Order", "Custom Grid (After Qualifying)"],
        help="Select Custom Grid to enter actual qualifying positions"
    )

# Custom grid input
custom_grid = {}
if "Custom Grid" in grid_scenario:
    st.markdown("---")
    st.subheader("📝 Enter Grid Positions (After Qualifying)")
    st.caption("Enter the starting position for each driver based on qualifying results (1-20)")

    # Create 4 columns for driver selection
    cols = st.columns(4)

    for idx, row in current_drivers.iterrows():
        col_idx = idx % 4
        with cols[col_idx]:
            # Default grid position = championship position
            default_pos = list(current_drivers['driverCode']).index(row['driverCode']) + 1
            grid_value = st.number_input(
                f"{row['driverCode']} ({row['constructorName']})",
                min_value=1,
                value=min(default_pos, 20),  # Cap default at 20
                step=1,
                key=f"grid_{row['driverCode']}"
            )
            # Clamp value between 1-20
            custom_grid[row['driverCode']] = max(1, min(20, grid_value))

st.markdown("---")

if st.button("🏁 Predict Top 3 Podium Finishers", type="primary", use_container_width=True):

    with st.spinner("Analyzing all 20 drivers..."):

        predictions = []

        # Process each driver
        for idx, driver_row in current_drivers.iterrows():

            driver = driver_row['driverCode']

            # Use custom grid if selected, otherwise championship order
            if "Custom Grid" in grid_scenario:
                grid_pos = custom_grid[driver]
            else:
                grid_pos = min(list(current_drivers['driverCode']).index(driver) + 1, 20)

            # Get circuit-specific stats
            driver_circuit_history = df[
                (df['driverCode'] == driver) &
                (df['round'] == circuit_info['round']) &
                (df['season'] < 2025)
                ]

            if len(driver_circuit_history) > 0:
                circuit_wins = (driver_circuit_history['position'] == 1).sum()
                circuit_podiums = (driver_circuit_history['position'] <= 3).sum()
                circuit_avg_finish = driver_circuit_history['position'].mean()
                circuit_races = len(driver_circuit_history)
            else:
                circuit_wins = 0
                circuit_podiums = 0
                circuit_avg_finish = 10.0
                circuit_races = 0

            # Create feature vector
            features = np.array([[
                grid_pos,
                driver_row['driver_last5_avg_points'],
                driver_row['driver_last5_avg_points'],
                10.0,
                2,
                0,
                driver_row['driver_season_points'],
                19,
                (driver_row['driver_season_points'] / 19 * 0.15) if driver_row['driver_season_points'] > 0 else 0,
                (driver_row['driver_season_points'] / 19 * 0.05) if driver_row['driver_season_points'] > 0 else 0,
                0.0,
                0.15,
                0.05,
                driver_row['constructor_season_points'],
                driver_row['driver_last5_avg_points'],
                38,
                1 if grid_pos <= 10 else 0,
                1 if grid_pos <= 5 else 0,
                21 - grid_pos,
                circuit_wins,
                circuit_podiums,
                circuit_races,
                circuit_avg_finish,
                min(circuit_avg_finish, 10),
                circuit_avg_finish,
                circuit_wins / circuit_races if circuit_races > 0 else 0,
                circuit_info['length_km'],
                15,
                2,
                200,
                circuit_info['overtaking'],
                circuit_info['tire_wear'],
                circuit_info['downforce']
            ]])

            # Predict
            prob = model.predict_proba(features)[0][1]

            predictions.append({
                'Driver': driver,
                'Team': driver_row['constructorName'],
                'Grid': grid_pos,
                'Podium_Probability': prob * 100,
                'Current_Points': driver_row['driver_season_points'],
                'Recent_Form': driver_row['driver_last5_avg_points']
            })

        # Sort by probability
        predictions_df = pd.DataFrame(predictions).sort_values('Podium_Probability', ascending=False)

        # Display Top 3
        st.success("### 🏆 Predicted Podium Finish")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🥇 1st Place")
            winner = predictions_df.iloc[0]
            st.metric(winner['Driver'], winner['Team'])
            st.metric("Podium Chance", f"{winner['Podium_Probability']:.1f}%")
            st.caption(
                f"🏁 Starts P{int(winner['Grid'])} | 📊 {int(winner['Current_Points'])} pts | ⚡ {winner['Recent_Form']:.1f} avg")

        with col2:
            st.markdown("#### 🥈 2nd Place")
            second = predictions_df.iloc[1]
            st.metric(second['Driver'], second['Team'])
            st.metric("Podium Chance", f"{second['Podium_Probability']:.1f}%")
            st.caption(
                f"🏁 Starts P{int(second['Grid'])} | 📊 {int(second['Current_Points'])} pts | ⚡ {second['Recent_Form']:.1f} avg")

        with col3:
            st.markdown("#### 🥉 3rd Place")
            third = predictions_df.iloc[2]
            st.metric(third['Driver'], third['Team'])
            st.metric("Podium Chance", f"{third['Podium_Probability']:.1f}%")
            st.caption(
                f"🏁 Starts P{int(third['Grid'])} | 📊 {int(third['Current_Points'])} pts | ⚡ {third['Recent_Form']:.1f} avg")

        # Full standings
        st.markdown("---")
        st.subheader("📊 All 20 Drivers - Podium Probabilities")

        # Create bar chart
        fig = px.bar(
            predictions_df.head(10),
            x='Podium_Probability',
            y='Driver',
            orientation='h',
            color='Podium_Probability',
            color_continuous_scale='Viridis',
            labels={'Podium_Probability': 'Podium Probability (%)'},
            title=f"Top 10 Drivers - {circuit}",
            hover_data=['Grid', 'Team']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        with st.expander("📋 See Full 20-Driver Rankings"):
            display_df = predictions_df[
                ['Driver', 'Team', 'Grid', 'Podium_Probability', 'Current_Points', 'Recent_Form']].copy()
            display_df['Podium_Probability'] = display_df['Podium_Probability'].apply(lambda x: f"{x:.1f}%")
            display_df['Grid'] = display_df['Grid'].astype(int)
            display_df['Current_Points'] = display_df['Current_Points'].astype(int)
            display_df['Recent_Form'] = display_df['Recent_Form'].apply(lambda x: f"{x:.1f}")
            display_df.index = range(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🏎️ **F1 Race Predictor V2** | 91.29% Accuracy | Powered by XGBoost")
