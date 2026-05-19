import streamlit as st
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os, time, warnings

import charts

# ignore warnings to keep terminal clean
warnings.filterwarnings('ignore')
st.set_page_config(page_title="2026 F1 Pit Wall", layout="wide", page_icon="formula1_logo.png")

# setup cache directory
# This saves time during initial boot of the website
cache_directory = 'f1_cache'
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory)
fastf1.Cache.enable_cache(cache_directory)

# 2026 grid info - mapping drivers to teams
# dict for the drivers
drivers = {
    'VER': 'Max Verstappen', 'HAD': 'Isack Hadjar', 'HAM': 'Lewis Hamilton', 'LEC': 'Charles Leclerc',
    'NOR': 'Lando Norris', 'PIA': 'Oscar Piastri', 'RUS': 'George Russell', 'ANT': 'Kimi Antonelli',
    'ALO': 'Fernando Alonso', 'STR': 'Lance Stroll', 'SAI': 'Carlos Sainz', 'ALB': 'Alexander Albon',
    'GAS': 'Pierre Gasly', 'COL': 'Franco Colapinto', 'OCO': 'Esteban Ocon', 'BEA': 'Oliver Bearman',
    'HUL': 'Nico Hulkenberg', 'BOR': 'Gabriel Bortoleto', 'LAW': 'Liam Lawson', 'LIN': 'Arvid Lindblad',
    'PER': 'Sergio Perez', 'BOT': 'Valtteri Bottas'
}

# dict for the teams
teams = {
    'VER': 'Red Bull Racing', 'HAD': 'Red Bull Racing', 'HAM': 'Ferrari', 'LEC': 'Ferrari',
    'NOR': 'McLaren', 'PIA': 'McLaren', 'RUS': 'Mercedes', 'ANT': 'Mercedes',
    'ALO': 'Aston Martin', 'STR': 'Aston Martin', 'SAI': 'Williams', 'ALB': 'Williams',
    'GAS': 'Alpine', 'COL': 'Alpine', 'OCO': 'Haas F1 Team', 'BEA': 'Haas F1 Team',
    'HUL': 'Audi', 'BOR': 'Audi', 'LAW': 'RB', 'LIN': 'RB',
    'PER': 'Cadillac', 'BOT': 'Cadillac'
}


# load data and models from the cache
@st.cache_resource
def load_data_and_models():
    f = "f1_stats_main.csv"
    if not os.path.exists(f):
        st.error("hist data file missing")
        return None, None, None, None

    df = pd.read_csv(f)
    mappings = {}

    # define all possible 2026 compounds so encoder
    # doesn't crash on unseen labels
    compounds = ["SOFT", "MEDIUM", "HARD", "INTER", "WET"]

    for col in ['Track', 'Driver', 'Team', 'Compound']:
        le = LabelEncoder()
        if col == 'Compound':
            # fit on all possible types + what's in the data
            combined = list(df[col].astype(str).unique()) + compounds
            le.fit(list(set(combined)))
            df[col] = le.transform(df[col].astype(str))
        else:
            df[col] = le.fit_transform(df[col].astype(str))
        mappings[col] = le


    # Data cleaning
    clean_df = df.dropna(subset=['TyreLife', 'Secs', 'Compound'])

    # train regressors
    # lap time model
    m_time = RandomForestRegressor(n_estimators=35, random_state=42)
    m_time.fit(clean_df[['Track', 'Driver', 'Team', 'Compound', 'TyreLife', 'Stint']], clean_df['Secs'])

    #lap time model
    deg_data = clean_df.groupby(['Track', 'Driver', 'Team', 'Compound'])['TyreLife'].max().reset_index()
    m_life = RandomForestRegressor(n_estimators=35, random_state=42)
    m_life.fit(deg_data[['Track', 'Driver', 'Team', 'Compound']], deg_data['TyreLife'])

    return df, m_time, m_life, mappings


full_df, model_time, model_life, mappings = load_data_and_models()

# creating the sidebar
# user inputs for race parameters
st.sidebar.header("Race Setup")
track_list = sorted(mappings['Track'].classes_)
# drop down menu to select the track from the track list
sel_track = st.sidebar.selectbox("Grand Prix", track_list)

# calc total laps from historical data (looking at last years race)
try:
    track_temp = mappings['Track'].transform([sel_track])[0]
    total_laps = int(full_df[full_df['Track'] == track_temp]['LapNumber'].max())
except:
    total_laps = 56

# selecting driver drop down menu from the drivers list
sel_drv_code = st.sidebar.selectbox("Driver", list(drivers.keys()), format_func=lambda x: f"{x} - {drivers[x]} ({teams[x]})")
# starting tyre selection via drop down menu
start_tyre = st.sidebar.selectbox("Starting Tyre", ["SOFT", "MEDIUM", "HARD"])

# safety car and wet adaptaptions to the programme
# safety car strategy
st.sidebar.divider()
st.sidebar.subheader("Live Context")
# safety car tickbox
sc_active = st.sidebar.checkbox("Safety Car / VSC", value=False)
# safety car lap selector ( set to 15 laps current)
sc_lap = st.sidebar.number_input("Incident Lap", 1, total_laps, 15) if sc_active else -1

# wet strategy
# tick box for if there is rain
rain_active = st.sidebar.checkbox("Rain in Forecast?", value=False)
# tick box to say the rain is heavy therefore full wet tyres are needed.
heavy_rain = st.sidebar.checkbox("Heavy Rain? (Wets)", value=False) if rain_active else False
# slider to select the lap that the rain starts
rain_lap = st.sidebar.slider("Rain Expected Lap", 1, total_laps, 15) if rain_active else -1


# Creating the engine (the brain of the model)
def get_all_strategies(base_life):
    # helper to pick the fastest compound for a stint using ml
    def pick_best_tyre(current_stint, remaining_laps, previous_tyre=None):
        options = ["SOFT", "MEDIUM", "HARD"]

        # calculate predicted times for all options
        results = []
        for tyre in options:
            c_val = mappings['Compound'].transform([tyre])[0]
            pred = model_time.predict([[track, driver, team, c_val, 1, current_stint + 1]])[0]
            results.append((tyre, pred))

        # sort by fastest lap time
        results.sort(key=lambda x: x[1])

        # rule: compound must be different at least once
        best_tyre = results[0][0]
        if previous_tyre and best_tyre == previous_tyre:
            best_tyre = results[1][0]

        return best_tyre

    # adj for sc if near pit window
    sc_offset = 0
    if sc_active and abs(sc_lap - base_life) <= 6:
        sc_offset = sc_lap - base_life

    # handle wet commands
    wet_comp = "WET" if heavy_rain else "INTER"

    # branch logic based on weather selection
    if rain_active:
        return {
            "Wet Aggressive": {
                "laps": [rain_lap - 2, rain_lap + 18],
                "tyres": [start_tyre, wet_comp, pick_best_tyre(2, 10)],
                "desc": "Early crossover to wet compound"
            },
            "Wet Alternate": {
                "laps": [rain_lap + 2],
                "tyres": [start_tyre, wet_comp],
                "desc": "Late switch (wait for track sat)"
            }
        }
    else:
        # dynamic dry tyre selection via ml
        t2_balanced = pick_best_tyre(1, total_laps - base_life, previous_tyre=start_tyre)
        t2_aggressive = pick_best_tyre(1, total_laps - (base_life - 5), previous_tyre=start_tyre)

        return {
            "Balanced": {
                "laps": [base_life + sc_offset, (base_life + sc_offset) + (total_laps - base_life) // 2],
                "tyres": [start_tyre, t2_balanced, "SOFT" if t2_balanced != "SOFT" else "MEDIUM"],
                "desc": "Standard pacing (low risk)"
            },
            "Aggressive": {
                "laps": [base_life - 5 + sc_offset, (base_life - 5 + sc_offset) + (total_laps - base_life) // 2],
                "tyres": [start_tyre, t2_aggressive, "SOFT" if t2_aggressive != "SOFT" else "MEDIUM"],
                "desc": "High pace undercut (tire wear risk)"
            },
            "Alternate": {
                "laps": [base_life + 8 + sc_offset],
                "tyres": [start_tyre, "HARD" if start_tyre != "HARD" else "MEDIUM"],
                "desc": "Long stint overcut (track position focus)"
            }
        }


# The user interface
st.title("F1 Pit Wall")

try:
    # Linking inputs to the engine
    track = mappings['Track'].transform([sel_track])[0]
    driver = mappings['Driver'].transform([sel_drv_code])[0]
    team = mappings['Team'].transform([teams[sel_drv_code]])[0]
    compound = mappings['Compound'].transform([start_tyre])[0]

    # predict tire life via ml model
    pred_life = int(model_life.predict([[track, driver, team, compound]])[0])
    all_strats = get_all_strategies(pred_life)

    # display strategy options dynamically
    # only show wet strategies if it is raining else show dry strats
    st.subheader("Strategy Forecasts")

    # Global Strategy Selector (This now controls everything)
    selected_strat_key = st.selectbox("Active Strategy Selection", list(all_strats.keys()))
    current_data = all_strats[selected_strat_key]

    # Quick pacing math logic
    total_race_time_secs = 0
    stops_count = len(current_data['laps'])
    pit_loss = 22  # avg f1 pit loss

    current_lap = 1
    pit_laps = current_data['laps']

    for i, lap_limit in enumerate(pit_laps + [total_laps]):
        stint_tyre = current_data['tyres'][i]
        tyre_val = mappings['Compound'].transform([stint_tyre])[0]

        for l in range(current_lap, lap_limit + 1):
            age = l - current_lap + 1
            l_time = model_time.predict([[track, driver, team, tyre_val, age, i + 1]])[0]
            total_race_time_secs += l_time
        current_lap = lap_limit + 1

    total_race_time_secs += (stops_count * pit_loss)
    avg_lap_time = total_race_time_secs / total_laps

    # Display Metrics
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        # makes the values into minutes, seconds, and microseconds for the race times
        st.metric("Est. Average Lap", f"{int(avg_lap_time // 60)}:{(avg_lap_time % 60):06.3f}")
    with perf_col2:
        st.metric("Total Race Time",
                  # makes the values into hours minutes seconds for the race times
                  f"{int(total_race_time_secs // 3600)}h {int((total_race_time_secs % 3600) // 60)}m {int(total_race_time_secs % 60)}s")

    st.divider()

    # Display summary cards for all available strats
    cols = st.columns(len(all_strats))
    for i, (name, data) in enumerate(all_strats.items()):
        with cols[i]:
            # Highlight the active strategy
            if name == selected_strat_key:
                st.markdown(f"#### :green[{name}]")
            else:
                st.markdown(f"**{name}**")
            st.caption(data['desc'])
            for j, lap in enumerate(data['laps']):
                st.metric(f"Stop {j + 1}", f"L{lap}")
            st.info(f"{'→'.join(data['tyres'])}")

    st.divider()

    # graphs tabs
    tab1, tab2, tab3 = st.tabs(["Telemetry Forecast", "Historical Analysis", "Manual Stint Planner"])

    with tab1:
        # Visualizing the strategy selected in the main dropdown above
        target_strat = all_strats[selected_strat_key]

        fig = charts.plot_telemetry(drivers[sel_drv_code], sel_track, target_strat['laps'],
                                    target_strat['tyres'], total_laps, sc_lap, model_time,
                                    track, driver, team, mappings)
        st.pyplot(fig)

    with tab2:
        # Pass the mappings dict as the new 4th argument
        st.pyplot(charts.plot_sawtooth(full_df, sel_track, sel_drv_code, mappings))

    with tab3:
        # manual planner logic
        st.write("### Build Your Own Strategy")
        m_stops = st.number_input("Num Planned Stops", 1, 4, 1)
        manual_laps = []
        manual_tyres = [start_tyre]

        m_cols = st.columns(int(m_stops))
        for i in range(int(m_stops)):
            with m_cols[i]:
                l = st.number_input(f"Stop {i + 1} Lap", 1, total_laps, 20 + (i * 15), key=f"m_l_{i}")
                t = st.selectbox(f"Fit Tyre {i + 1}", ["SOFT", "MEDIUM", "HARD", "INTER", "WET"], key=f"m_t_{i}")
                manual_laps.append(l)
                manual_tyres.append(t)

        # check if they are trying to run the same compound the whole race
        # f1 rules say you need two different sets if it's dry
        is_legal = True
        if not rain_active and len(set(manual_tyres)) == 1:
            is_legal = False
            st.error("Regulations Error: You must use at least two different tyre compounds in a dry race.")

        if st.button("Gen Custom Telemetry"):
            if not is_legal:
                st.warning("Please change one of your tyre selections to meet the regulations.")
            else:
                fig_custom = charts.plot_telemetry(drivers[sel_drv_code], sel_track, manual_laps,
                                                   manual_tyres, total_laps, sc_lap, model_time,
                                                   track, driver, team, mappings)
                st.pyplot(fig_custom)

except Exception as e:
    st.warning(f"err in execution {e}")

st.write("by andrew pepper")