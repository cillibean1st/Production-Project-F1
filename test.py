import streamlit as st
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os, warnings
import matplotlib.pyplot as plt
import requests

# Ensure charts.py logic is available locally or wrapped within the execution pipeline
try:
    import charts
except ImportError:
    charts = None

# --- 1. SETTINGS, THEME & COLOR MAPPING ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="2026 F1 Pit Wall", layout="wide", page_icon="formula1_logo.png")

# Official Pirelli/F1 Hex Codes
TYRE_COLORS = {
    "SOFT": "#FF3333",  # Red
    "MEDIUM": "#FAD400",  # Yellow
    "HARD": "#FFFFFF",  # White
    "INTER": "#43B02A",  # Green
    "WET": "#0067B9"  # Blue
}

# Team Color Matrix for Live Tracker Dots
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6', 'Ferrari': '#E8002D', 'McLaren': '#FF8000',
    'Mercedes': '#27F4D2', 'Aston Martin': '#229971', 'Williams': '#64C4FF',
    'Alpine': '#0093CC', 'Haas F1 Team': '#B6BABD', 'Audi': '#F50A25', 'Cadillac': '#7D26CD',
    'RB': '#6692FF'
}

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #111111; color: #ffffff; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] label { color: #ffffff !important; }

    /* METRIC BOXES STYLING */
    div[data-testid="stMetric"] {
        background-color: #1a1a1a !important;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    div[data-testid="stMetric"] label, 
    div[data-testid="stMetric"] div, 
    div[data-testid="stMetric"] p,
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricDelta"] > div {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

if 'zoomed_view' not in st.session_state:
    st.session_state.zoomed_view = None


# --- 2. FORMATTING HELPERS ---
def format_f1_time(seconds, is_lap_time=False):
    if seconds is None or np.isnan(seconds): return "00:00.000"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if is_lap_time:
        return f"{int(m):02d}:{s:06.3f}"
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


# --- 3. DATA & MODELS ---
cache_directory = 'f1_cache'
if not os.path.exists(cache_directory): os.makedirs(cache_directory)
fastf1.Cache.enable_cache(cache_directory)

drivers = {
    'VER': 'Max Verstappen', 'HAD': 'Isack Hadjar', 'HAM': 'Lewis Hamilton', 'LEC': 'Charles Leclerc',
    'NOR': 'Lando Norris', 'PIA': 'Oscar Piastri', 'RUS': 'George Russell', 'ANT': 'Kimi Antonelli',
    'ALO': 'Fernando Alonso', 'STR': 'Lance Stroll', 'SAI': 'Carlos Sainz', 'ALB': 'Alexander Albon',
    'GAS': 'Pierre Gasly', 'COL': 'Franco Colapinto', 'OCO': 'Esteban Ocon', 'BEA': 'Oliver Bearman',
    'HUL': 'Nico Hulkenberg', 'BOR': 'Gabriel Bortoleto', 'LAW': 'Liam Lawson', 'LIN': 'Arvid Lindblad',
    'PER': 'Sergio Perez', 'BOT': 'Valtteri Bottas'
}

DRIVER_NUMBERS = {
    'VER': 1, 'HAM': 44, 'LEC': 16, 'NOR': 4, 'PIA': 81, 'RUS': 63, 'ALO': 14, 'STR': 18,
    'SAI': 55, 'ALB': 23, 'GAS': 10, 'OCO': 31, 'HUL': 27, 'PER': 11, 'BOT': 77, 'LAW': 30
}

teams = {
    'VER': 'Red Bull Racing', 'HAD': 'Red Bull Racing', 'HAM': 'Ferrari', 'LEC': 'Ferrari',
    'NOR': 'McLaren', 'PIA': 'McLaren', 'RUS': 'Mercedes', 'ANT': 'Mercedes',
    'ALO': 'Aston Martin', 'STR': 'Aston Martin', 'SAI': 'Williams', 'ALB': 'Williams',
    'GAS': 'Alpine', 'COL': 'Alpine', 'OCO': 'Haas F1 Team', 'BEA': 'Haas F1 Team',
    'HUL': 'Audi', 'BOR': 'Audi', 'LAW': 'RB', 'LIN': 'RB',
    'PER': 'Cadillac', 'BOT': 'Cadillac'
}

TRACK_COORDS = {
    "Australian Grand Prix": [-37.8497, 144.9680], "Chinese Grand Prix": [31.3389, 121.2222],
    "Japanese Grand Prix": [34.8431, 136.5410], "Bahrain Grand Prix": [26.0325, 50.5106],
    "Saudi Arabian Grand Prix": [21.6319, 39.1044], "Miami Grand Prix": [25.9581, -80.2389],
    "Grand Prix du Canada": [45.5005, -73.5225], "Monaco Grand Prix": [43.7347, 7.4206],
    "Spanish Grand Prix": [41.5700, 2.2611], "Austrian Grand Prix": [47.2197, 14.7647],
    "British Grand Prix": [52.0786, -1.0169], "Belgian Grand Prix": [50.4372, 5.9714],
    "Hungarian Grand Prix": [47.5830, 19.2486], "Dutch Grand Prix": [52.3888, 4.5409],
    "Italian Grand Prix": [45.6156, 9.2811], "Azerbaijan Grand Prix": [40.3725, 49.8533],
    "Singapore Grand Prix": [1.2914, 103.8640], "United States Grand Prix": [30.1328, -97.6411],
    "Mexico City Grand Prix": [19.4042, -99.0907], "Sao Paulo Grand Prix": [-23.7011, -46.6972],
    "Las Vegas Grand Prix": [36.1147, -115.1728], "Qatar Grand Prix": [25.4900, 51.4542],
    "Abu Dhabi Grand Prix": [24.4672, 54.6031], "Madrid Grand Prix": [40.4168, -3.7038]
}

OPENF1_2025_SESSIONS = {
    "Australian Grand Prix": "9495", "Chinese Grand Prix": "9507", "Japanese Grand Prix": "9483",
    "Bahrain Grand Prix": "9465", "Saudi Arabian Grand Prix": "9471", "Miami Grand Prix": "9513",
    "Monaco Grand Prix": "9525", "Spanish Grand Prix": "9537", "Austrian Grand Prix": "9543",
    "British Grand Prix": "9549", "Hungarian Grand Prix": "9555", "Belgian Grand Prix": "9561",
    "Dutch Grand Prix": "9567", "Italian Grand Prix": "9573", "Azerbaijan Grand Prix": "9579",
    "Singapore Grand Prix": "9585", "United States Grand Prix": "9597", "Mexico City Grand Prix": "9603",
    "Sao Paulo Grand Prix": "9609", "Las Vegas Grand Prix": "9615", "Qatar Grand Prix": "9621",
    "Abu Dhabi Grand Prix": "9627"
}


@st.cache_resource
def load_data_and_models():
    f = "f1_stats_main.csv"
    if not os.path.exists(f): return None, None, None, None
    df = pd.read_csv(f)
    mappings = {}
    compounds = list(TYRE_COLORS.keys())
    for col in ['Track', 'Driver', 'Team', 'Compound']:
        le = LabelEncoder()
        if col == 'Compound':
            le.fit(list(set(list(df[col].astype(str).unique()) + compounds)))
            df[col] = le.transform(df[col].astype(str))
        else:
            df[col] = le.fit_transform(df[col].astype(str))
        mappings[col] = le
    clean_df = df.dropna(subset=['TyreLife', 'Secs', 'Compound'])
    m_time = RandomForestRegressor(n_estimators=35, random_state=42).fit(
        clean_df[['Track', 'Driver', 'Team', 'Compound', 'TyreLife', 'Stint']], clean_df['Secs'])
    deg_data = clean_df.groupby(['Track', 'Driver', 'Team', 'Compound'])['TyreLife'].max().reset_index()
    m_life = RandomForestRegressor(n_estimators=35, random_state=42).fit(
        deg_data[['Track', 'Driver', 'Team', 'Compound']], deg_data['TyreLife'])
    return df, m_time, m_life, mappings


full_df, model_time, model_life, mappings = load_data_and_models()


# --- 4. OPENF1 DATA CONDUIT ENGINE ---
@st.cache_data(ttl=15)
def fetch_openf1_live_telemetry(driver_code, target_lap, session_key_input):
    base_url = "https://api.openf1.org/v1/"
    driver_num = DRIVER_NUMBERS.get(driver_code, 55)
    try:
        car_url = f"{base_url}car_data?session_key={session_key_input}&driver_number={driver_num}"
        car_res = requests.get(car_url, timeout=5).json()

        loc_url = f"{base_url}location?session_key={session_key_input}&driver_number={driver_num}"
        loc_res = requests.get(loc_url, timeout=5).json()

        lap_url = f"{base_url}laps?session_key={session_key_input}&driver_number={driver_num}&lap_number={target_lap}"
        lap_res = requests.get(lap_url, timeout=5).json()

        if car_res and loc_res:
            df_car = pd.DataFrame(car_res).tail(400).reset_index(drop=True)
            df_loc = pd.DataFrame(loc_res).tail(400).reset_index(drop=True)
            min_len = min(len(df_car), len(df_loc))
            if min_len > 0:
                speeds = df_car['speed'].iloc[:min_len].astype(float).values
                time_step = 1.0 / 3.7
                distance = np.cumsum((speeds / 3.6) * time_step)
                s1, s2, s3, cmp, age = 28.5, 33.0, 22.5, "MEDIUM", target_lap
                if lap_res:
                    s1 = lap_res[0].get("duration_sector_1", s1) or s1
                    s2 = lap_res[0].get("duration_sector_2", s2) or s2
                    s3 = lap_res[0].get("duration_sector_3", s3) or s3
                    cmp = str(lap_res[0].get("backing_compound", "MEDIUM")).upper()
                    age = lap_res[0].get("tyre_age", target_lap) or target_lap
                return {
                    "compound": cmp if cmp in TYRE_COLORS else "MEDIUM", "life": age,
                    "s1": s1, "s2": s2, "s3": s3, "distance": distance, "speed": speeds,
                    "x": df_loc['x'].iloc[:min_len].astype(float).values,
                    "y": df_loc['y'].iloc[:min_len].astype(float).values
                }
    except Exception:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_openf1_spatial_map(driver_code, session_key_input):
    base_url = "https://api.openf1.org/v1/"
    driver_num = DRIVER_NUMBERS.get(driver_code, 55)
    try:
        loc_url = f"{base_url}location?session_key={session_key_input}&driver_number={driver_num}"
        loc_res = requests.get(loc_url, timeout=7).json()
        car_url = f"{base_url}car_data?session_key={session_key_input}&driver_number={driver_num}"
        car_res = requests.get(car_url, timeout=7).json()

        if loc_res and car_res:
            df_loc = pd.DataFrame(loc_res)
            df_car = pd.DataFrame(car_res)

            df_loc = df_loc.iloc[::6].reset_index(drop=True)
            df_car = df_car.iloc[::6].reset_index(drop=True)
            min_len = min(len(df_car), len(df_loc))
            if min_len > 10:
                # OpenF1 flips coordinates upside down relative to map graphics. Inverting Y preserves true track shapes.
                return {
                    "x": df_loc['x'].iloc[:min_len].astype(float).values,
                    "y": -df_loc['y'].iloc[:min_len].astype(float).values,
                    "speed": df_car['speed'].iloc[:min_len].astype(float).values
                }
    except Exception:
        pass

    p_dist = np.linspace(0, 2 * np.pi, 600)
    return {
        "x": 1600 * np.sin(p_dist) + 400 * np.sin(2 * p_dist),
        "y": -(900 * np.cos(p_dist) + 200 * np.sin(3 * p_dist)),
        "speed": 180 + 110 * np.sin(2 * p_dist)
    }


# --- 5. STRATEGY ENGINE ---
def get_all_strategies(base_life, track, driver, team, total_laps, start_tyre, sc_active, sc_start, sc_end, rain_active,
                       heavy_rain, rain_start, rain_end):
    def pick_best_tyre(stint, prev=None):
        options = ["SOFT", "MEDIUM", "HARD"]
        res = []
        for t in options:
            c = mappings['Compound'].transform([t])[0]
            p = model_time.predict([[track, driver, team, c, 1, stint + 1]])[0]
            res.append((t, p))
        res.sort(key=lambda x: x[1])
        return res[1][0] if res[0][0] == prev else res[0][0]

    sc_offset = 0
    if sc_active and (sc_start <= base_life <= sc_end):
        sc_offset = sc_start - base_life
    wet_comp = "WET" if heavy_rain else "INTER"
    if rain_active:
        return {
            "Wet Aggressive": {"laps": [rain_start - 1, rain_end], "tyres": [start_tyre, wet_comp, pick_best_tyre(2)],
                               "desc": "Box for wets immediately"},
            "Wet Alternate": {"laps": [rain_start + 2, rain_end + 2],
                              "tyres": [start_tyre, wet_comp, pick_best_tyre(2)], "desc": "Wait for track saturation"}
        }
    else:
        t2_bal = pick_best_tyre(1, start_tyre)
        return {
            "Balanced": {"laps": [base_life + sc_offset, (base_life + sc_offset) + (total_laps - base_life) // 2],
                         "tyres": [start_tyre, t2_bal, "SOFT" if t2_bal != "SOFT" else "MEDIUM"],
                         "desc": "Standard pacing"},
            "Aggressive": {
                "laps": [base_life - 5 + sc_offset, (base_life - 5 + sc_offset) + (total_laps - base_life) // 2],
                "tyres": [start_tyre, t2_bal, "SOFT"], "desc": "Undercut focus"},
            "Alternate": {"laps": [base_life + 8 + sc_offset], "tyres": [start_tyre, "HARD"],
                          "desc": "Long stint overcut"}
        }


def zoom_btn(label, key):
    c = st.columns([0.8, 0.2])
    c[0].subheader(label)
    if st.session_state.zoomed_view == key:
        if c[1].button("Exit", key=f"ex_{key}"):
            st.session_state.zoomed_view = None
            st.rerun()
    else:
        if c[1].button("Zoom", key=f"zm_{key}"):
            st.session_state.zoomed_view = key
            st.rerun()


# --- 6. SIDEBAR & GLOBAL CONTROLS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)
st.sidebar.header("Race Control")

app_mode = st.sidebar.selectbox("Dashboard Mode", ["Editing Mode", "Live Mode"])
sel_track = st.sidebar.selectbox("Grand Prix", sorted(mappings['Track'].classes_))

if app_mode == "Editing Mode":
    openf1_session_key = OPENF1_2025_SESSIONS.get(sel_track, "9495")
else:
    openf1_session_key = "latest"

sel_drv_code = st.sidebar.selectbox("Driver", list(drivers.keys()), format_func=lambda x: f"{x} - {drivers[x]}")
start_tyre = st.sidebar.selectbox("Starting Tyre", ["SOFT", "MEDIUM", "HARD"])

tr_id, dr_id = mappings['Track'].transform([sel_track])[0], mappings['Driver'].transform([sel_drv_code])[0]
tm_id, co_id = mappings['Team'].transform([teams[sel_drv_code]])[0], mappings['Compound'].transform([start_tyre])[0]

try:
    tot_laps = int(full_df[full_df['Track'] == tr_id]['LapNumber'].max())
except:
    tot_laps = 56

st.sidebar.divider()
st.sidebar.subheader("Safety Car & Weather")
sc_active = st.sidebar.checkbox("SC / VSC Active")
sc_start, sc_end = st.sidebar.slider("SC Window", 1, tot_laps, (15, 20)) if sc_active else (-1, -1)
rain_active = st.sidebar.checkbox("Rain in Forecast")
rain_start, rain_end = st.sidebar.slider("Rain Duration", 1, tot_laps, (10, 25)) if rain_active else (-1, -1)
heavy_rain = st.sidebar.checkbox("Heavy Rain (Full Wets)") if rain_active else False

pred_life = int(model_life.predict([[tr_id, dr_id, tm_id, co_id]])[0])
all_strats = get_all_strategies(pred_life, tr_id, dr_id, tm_id, tot_laps, start_tyre, sc_active, sc_start, sc_end,
                                rain_active, heavy_rain, rain_start, rain_end)

# --- 7. MAIN DASHBOARD ---
st.title(f"PIT WALL: {sel_track} 2026")
st.caption(f"Pipeline Mode: **{app_mode}** | Target Session Instance Token Key: `{openf1_session_key}`")

sel_strat_key = st.selectbox("Active Strategy Selection", list(all_strats.keys()))
active_strat = all_strats[sel_strat_key]

m_top1, m_top2, m_top3 = st.columns(3)
with m_top1: st.metric("Driver", f"{sel_drv_code}", teams[sel_drv_code])
with m_top2: st.metric("Predicted Tyre Life", f"{pred_life} Laps")
with m_top3: st.metric("Total Stops", f"{len(active_strat['laps'])}")

st.divider()

if st.session_state.zoomed_view is None:
    r1c1, r1c2 = st.columns(2);
    r2c1, r2c2 = st.columns(2);
    r3c1, r3c2 = st.columns(2)
    with r1c1:
        zoom_btn("Strategy Forecast", "tele")
        if charts:
            st.pyplot(
                charts.plot_telemetry(drivers[sel_drv_code], sel_track, active_strat['laps'], active_strat['tyres'],
                                      tot_laps, sc_start, model_time, tr_id, dr_id, tm_id, mappings))
        else:
            st.info("Charts configuration module standby.")
    with r1c2:
        zoom_btn("Live Timing", "live")
        st.table(pd.DataFrame({"Pos": [1, 2, 3], "Driver": ["VER", "HAM", "LEC"], "Gap": ["--", "+1.2", "+4.5"]}))
    with r2c1:
        zoom_btn("Historical Data", "hist")
        if charts:
            st.pyplot(charts.plot_sawtooth(full_df, sel_track, sel_drv_code, mappings))
        else:
            st.info("Sawtooth telemetry rendering pending.")
    with r2c2:
        zoom_btn("Track Map Monitor", "map")
        st.info(
            "Expand panel via Zoom button to map live GPS array stream configurations from the 2025 archive database.")
    with r3c1:
        zoom_btn("Manual Planner", "plan");
        st.write("Simulate custom pit profiles.")
    with r3c2:
        zoom_btn("Weather Report", "weather");
        st.caption("24°C | Track: 38°C")

else:
    z = st.session_state.zoomed_view

    if z == "plan":
        zoom_btn("Manual Strategy Planner", "plan")
        col_inputs, col_results = st.columns([1, 2])
        with col_inputs:
            st.markdown("### Stint Configuration")
            num_stops = st.number_input("Number of Pit Stops", 1, 3, 1)
            custom_strategy, current_lap = [], 0
            used_compounds = set()
            for i in range(num_stops + 1):
                st.divider()
                remaining = tot_laps - current_lap
                if i == num_stops:
                    s_laps = max(0, remaining)
                    st.info(f"Stint {i + 1} (Final): Fixed to {s_laps} Laps")
                else:
                    max_allowed = max(1, remaining - (num_stops - i))
                    s_laps = st.slider(f"Stint {i + 1} Length", 1, max_allowed, min(15, max_allowed), key=f"s{i}l")
                s_tyre = st.selectbox(f"Stint {i + 1} Tyre", ["SOFT", "MEDIUM", "HARD"], index=1, key=f"s{i}t")
                custom_strategy.append({"laps": s_laps, "tyre": s_tyre})
                used_compounds.add(s_tyre)
                current_lap += s_laps
            regulation_violated = len(used_compounds) < 2

        with col_results:
            if regulation_violated:
                st.error("REGULATION VIOLATION: Use at least two different dry-weather compounds.")
            else:
                st.success(f"Strategy validated for {tot_laps} laps.")
                total_times, total_race_s = [], 0
                fig, ax = plt.subplots(figsize=(10, 5))
                lap_offset = 1
                for idx, stint in enumerate(custom_strategy):
                    stint_laps, stint_times = [], []
                    cid = mappings['Compound'].transform([stint['tyre']])[0]
                    for age in range(1, stint['laps'] + 1):
                        pred = model_time.predict([[tr_id, dr_id, tm_id, cid, age, idx + 1]])[0]
                        stint_times.append(pred);
                        stint_laps.append(lap_offset);
                        lap_offset += 1
                    ax.plot(stint_laps, stint_times, color=TYRE_COLORS[stint['tyre']], linewidth=3,
                            label=f"Stint {idx + 1}: {stint['tyre']}")
                    total_race_s += sum(stint_times)
                    total_times.extend(stint_times)
                total_race_s += num_stops * 22.0
                m_c1, m_c2 = st.columns(2)
                m_c1.metric("Est. Total Race Time", format_f1_time(total_race_s))
                m_c2.metric("Est. Average Lap Time", format_f1_time(np.mean(total_times), is_lap_time=True))
                ax.set_facecolor('#0e1117');
                fig.patch.set_facecolor('#0e1117')
                ax.set_ylabel("Lap Time (s)", color='white');
                ax.set_xlabel("Lap Number", color='white')
                ax.tick_params(colors='white');
                ax.legend(facecolor='#1a1a1a', labelcolor='white');
                ax.grid(alpha=0.1)
                st.pyplot(fig)

    elif z == "map":
        zoom_btn("Track Map Monitor & Telemetry Simulator", "map")
        st.markdown(f"### Spatial GPS Track Simulation & Performance Analysis: {sel_track}")
        st.caption(f"Synthesizing historical telemetry map arrays using Session Reference ID: `{openf1_session_key}`")

        with st.spinner("Downloading high-frequency relative position telemetry files from OpenF1 arrays..."):
            map_data = fetch_openf1_spatial_map(sel_drv_code, openf1_session_key)

        if map_data and len(map_data["x"]) > 0:
            x_coords = map_data["x"]
            y_coords = map_data["y"]
            speeds = map_data["speed"]

            # Form dual-axis plotting canvas
            fig_split, (ax_map, ax_trace) = plt.subplots(1, 2, figsize=(16, 6.5),
                                                         gridspec_kw={'width_ratios': [1.2, 1]})
            fig_split.patch.set_facecolor('#0e1117')

            # SUBPLOT A: GEOGRAPHIC CIRCUIT LAYOUT TRACK
            ax_map.set_facecolor('#0e1117')
            ax_map.plot(x_coords, y_coords, color='white', alpha=0.15, linewidth=2, zorder=1)
            sc = ax_map.scatter(x_coords, y_coords, c=speeds, cmap='plasma', s=6, zorder=2)
            ax_map.scatter(x_coords[0], y_coords[0], color='#00FF00', s=150, marker='^', label='Start / Finish Line',
                           zorder=3)

            # Fixing the track shape distortions
            ax_map.set_aspect('equal')
            ax_map.axis('off')

            # Enforce tight cropping padding to remove blank void margins around the true geometry outline
            ax_map.set_xlim(min(x_coords) - 150, max(x_coords) + 150)
            ax_map.set_ylim(min(y_coords) - 150, max(y_coords) + 150)

            ax_map.set_title("True Circuit Shape Matrix", color='white', fontsize=12, pad=10)
            ax_map.legend(facecolor='#1a1a1a', labelcolor='white', loc='upper right', fontsize=9)

            cbar = fig_split.colorbar(sc, ax=ax_map, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label('Velocity scale mapping (km/h)', color='white', fontsize=9)
            cbar.ax.tick_params(labelsize=8, colors='white')

            # SUBPLOT B: LINEAR TIMELINE PROFILE
            ax_trace.set_facecolor('#14171f')
            distance_axis = np.arange(len(speeds))
            ax_trace.plot(distance_axis, speeds, color='#00FFE0', linewidth=2, label=f"{sel_drv_code} Live Loop")

            ax_trace.set_title("Linear Velocity Profile (Braking vs. Throttle)", color='white', fontsize=12, pad=10)
            ax_trace.set_xlabel("Track Data Samples (Timeline)", color='#aaaaaa', fontsize=9)
            ax_trace.set_ylabel("Telemetry Speed (km/h)", color='#aaaaaa', fontsize=9)
            ax_trace.tick_params(colors='white', labelsize=8)
            ax_trace.grid(color='#2c303d', linestyle=':', alpha=0.6)
            ax_trace.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)
            ax_trace.set_ylim(min(speeds) - 20, max(speeds) + 20)

            st.pyplot(fig_split)
            st.success("Successfully processed historical trace maps with aspect adjustments.")

            # --- NEW ADDITION: LIVE DRIVER POSITION INTERVAL TRACKER GRID ---
            st.divider()
            st.subheader("Live Driver Interval Positioning Stand")
            st.caption("Tracking physical localization steps across the circuit loop.")

            # Pull positioning states across the benchmark track loop array length
            time_slider_step = st.slider("Timeline Tracking Matrix Frame Index", 0, len(speeds) - 1, len(speeds) // 2)

            fig_live_track, ax_live = plt.subplots(figsize=(15, 6))
            fig_live_track.patch.set_facecolor('#0e1117')
            ax_live.set_facecolor('#0e1117')

            # Draw underlying gray layout skeleton line
            ax_live.plot(x_coords, y_coords, color='#2c303d', linewidth=4, alpha=0.8, zorder=1)
            ax_live.scatter(x_coords[0], y_coords[0], color='#00FF00', s=100, marker='^', zorder=2)

            # live grid list
            active_grid = [
                'VER', 'HAD', 'HAM', 'LEC', 'NOR', 'PIA', 'RUS', 'ANT',
                'ALO', 'STR', 'SAI', 'ALB', 'GAS', 'COL', 'OCO', 'BEA',
                'HUL', 'BOR', 'LAW', 'LIN', 'PER', 'BOT'
            ]
            # Map tracking intervals sequentially for every driver in our list
            for idx, d_code in enumerate(active_grid):
                d_team = teams.get(d_code, 'Mercedes')
                d_color = TEAM_COLORS.get(d_team, '#fffff')

                # Disperse drivers across realistic interval delta steps on the circuit path
                offset_idx = (time_slider_step - (idx * 22)) % len(speeds)
                p_x = x_coords[offset_idx]
                p_y = y_coords[offset_idx]

                # Plot position dot
                ax_live.scatter(p_x, p_y, color=d_color, s=120, edgecolors='white', linewidth=1.5, zorder=4)
                # Attach timing badge text labels next to each marker
                ax_live.text(p_x + 35, p_y + 35, d_code, color='white', fontsize=9, fontweight='bold',
                             bbox=dict(facecolor='#1a1a1a', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none'),
                             zorder=5)

            ax_live.set_aspect('equal')
            ax_live.axis('off')
            ax_live.set_xlim(min(x_coords) - 150, max(x_coords) + 150)
            ax_live.set_ylim(min(y_coords) - 150, max(y_coords) + 150)

            st.pyplot(fig_live_track)
        else:
            st.warning("Telemetry file load execution drop.")

    elif z == "live":
        zoom_btn("Live Mission Control Panel", "live")
        st.markdown("### OpenF1 Engine System Output")
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
        with ctrl1:
            main_lap = st.selectbox("Target Lap", range(1, tot_laps + 1), index=min(10, tot_laps - 1))
            compare_on = st.toggle("Enable Rival Comparison", value=True)
        with ctrl2:
            comp_drv = st.selectbox("Compare Against", list(drivers.keys()), index=3) if compare_on else sel_drv_code
            comp_lap = st.selectbox("Rival Lap", range(1, tot_laps + 1),
                                    index=min(10, tot_laps - 1)) if compare_on else None
        with ctrl3:
            view_mode = st.radio("Resolution Split Mode", ["Sectors", "Segments"], horizontal=True)

        p_compound, p_life = start_tyre, main_lap
        c_compound, c_life = "MEDIUM", comp_lap
        p_dist, p_speed, c_dist, c_speed = None, None, None, None
        v1, v2 = [28.45, 33.12, 22.51], [28.61, 32.95, 22.72]

        p_live = fetch_openf1_live_telemetry(sel_drv_code, main_lap, openf1_session_key)
        if p_live:
            p_compound = p_live["compound"];
            p_life = p_live["life"];
            p_dist = p_live["distance"];
            p_speed = p_live["speed"]
            v1 = [p_live["s1"], p_live["s2"], p_live["s3"]]
        if compare_on:
            c_live = fetch_openf1_live_telemetry(comp_drv, comp_lap, openf1_session_key)
            if c_live:
                c_compound = c_live["compound"];
                c_life = c_live["life"];
                c_dist = c_live["distance"];
                c_speed = c_live["speed"]
                v2 = [c_live["s1"], c_live["s2"], c_live["s3"]]

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"**PRIMARY: {sel_drv_code} (Lap {main_lap})**")
            sc1, sc2 = st.columns(2)
            sc1.metric("Tyre Compound", p_compound);
            sc2.metric("Stint Duration Age", f"{int(p_life)} L")
        if compare_on:
            with m_col2:
                st.markdown(f"**RIVAL: {comp_drv} (Lap {comp_lap})**")
                rc1, rc2 = st.columns(2)
                rc1.metric("Tyre Compound", c_compound);
                rc2.metric("Stint Duration Age", f"{int(c_life)} L")

        st.divider()
        col_list, col_graph = st.columns([1.5, 2.5])
        with col_list:
            st.subheader(f"Timing Split Data ({view_mode})")
            if view_mode == "Sectors":
                rows = ["Sector 1", "Sector 2", "Sector 3"]
            else:
                rows = [f"Segment {i}" for i in range(1, 7)]
                np.random.seed(main_lap)
                v1 = [round(np.random.uniform(5.0, 12.0), 2) for _ in range(len(rows))]
                v2 = [round(np.random.uniform(5.0, 12.0), 2) for _ in range(len(rows))]

            timing_data = {"Interval": rows, f"{sel_drv_code}": [f"{v}s" for v in v1]}
            if compare_on:
                timing_data[f"{comp_drv}"] = [f"{v}s" for v in v2]
                deltas = [round(v1[i] - v2[i], 3) for i in range(len(v1))]
                timing_data["Delta"] = [f"+{d}" if d > 0 else f"{d}" for d in deltas]
            st.dataframe(pd.DataFrame(timing_data), use_container_width=True, hide_index=True)

        with col_graph:
            st.subheader("Velocity Trace Profile")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            if p_dist is not None and p_speed is not None:
                max_valid_len = min(len(p_dist), len(p_speed))
                ax.plot(p_dist[:max_valid_len], p_speed[:max_valid_len], color='cyan', label=f"{sel_drv_code}",
                        linewidth=2.5)
                if compare_on and c_dist is not None and c_speed is not None:
                    comp_len = min(max_valid_len, len(c_dist), len(c_speed))
                    ax.plot(c_dist[:comp_len], c_speed[:comp_len], color='#FF3333', label=f"{comp_drv}", linestyle='--',
                            linewidth=1.5)
            track_max_distance = float(p_dist[-1]) if p_dist is not None else 5200.0
            ax.set_xlim(0, track_max_distance)
            ax.set_facecolor('#0e1117');
            fig.patch.set_facecolor('#0e1117')
            ax.set_ylabel("Speed (km/h)", color='white');
            ax.set_xlabel("Distance (m)", color='white')
            ax.tick_params(colors='white');
            ax.legend(facecolor='#1a1a1a', labelcolor='white')
            st.pyplot(fig)

    elif z == "tele":
        zoom_btn("Strategy Telemetry", "tele")
        if charts:
            st.pyplot(
                charts.plot_telemetry(drivers[sel_drv_code], sel_track, active_strat['laps'], active_strat['tyres'],
                                      tot_laps, sc_start, model_time, tr_id, dr_id, tm_id, mappings))
    elif z == "hist":
        zoom_btn("Historical Stint Analysis", "hist")
        if charts: st.pyplot(charts.plot_sawtooth(full_df, sel_track, sel_drv_code, mappings))

    elif z == "weather":
        from streamlit_folium import st_folium
        import folium

        zoom_btn("AerisWeather Real-Time Feed", "weather")
        coords = TRACK_COORDS.get(sel_track, [52.0786, -1.0169])
        lat, lon = coords[0], coords[1]

        AERIS_CLIENT_ID, AERIS_CLIENT_SECRET = "WyOOjxDDoMBKYKJKeoi3E", "Hgw82LbtFF0MsEubwNSWseoMzjzZxhV2x4g43BhC"
        weather_url = f"https://data.api.xweather.com/conditions/{lat},{lon}?filter=minutelyprecip,5min&limit=15&client_id={AERIS_CLIENT_ID}&client_secret={AERIS_CLIENT_SECRET}"

        df_weather = pd.DataFrame()

        with st.spinner("Requesting weather telemetry..."):
            try:
                response = requests.get(weather_url, timeout=5)
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    # --- INSIDE YOUR API FETCH LOOP ---
                    if data.get('success'):
                        periods = data['response'][0]['periods']


                        # 2. Process data directly into the df_weather variable
                        temp_data = [{
                            "Time": pd.to_datetime(p.get('timestamp', 0), unit='s').strftime("%H:%M"),
                            "Air Temp": float(p.get('tempC', 0)),
                            "Track Temp": float(p.get('tempC', 0)) + 12.0,
                            # Use .get('pop', 0) to prevent the KeyError
                            "Rain Prob": int(p.get('pop', 0)),
                            "Humidity": f"{p.get('humidity', 0)}%",
                            "Wind": f"{p.get('windSpeedKPH', 0)} km/h"
                        } for p in periods]

                        df_weather = pd.DataFrame(temp_data)
                    else:
                        st.error("API returned success=false")
                else:
                    st.error(f"API Request failed with status code: {response.status_code}")

            except Exception as e:
                st.error(f"Connection Error: {e}")

        # 3. Now it is safe to check if the dataframe has data
        if not df_weather.empty:
            # Your plotting and display logic here
            st.dataframe(df_weather)
        else:
            st.warning("Forecast data is currently unavailable.")

        tab_radar, tab_forecast, tab_trends = st.tabs(["Live HD Radar", "Real-Time Forecast", "Atmospheric Trends"])
        with tab_radar:
            m = folium.Map(location=coords, zoom_start=15, tiles='CartoDB positron')
            aeris_tile_url = f"https://maps.aerisapi.com/{AERIS_CLIENT_ID}_{AERIS_CLIENT_SECRET}/radar/{{z}}/{{x}}/{{y}}/current.png"
            folium.TileLayer(tiles=aeris_tile_url, attr='AerisWeather', overlay=True, opacity=0.8).add_to(m)
            st_folium(m, width=1200, height=550, key=f"aeris_real_{sel_track}")

        with tab_forecast:
            # 1. Move the function definition here so it is accessible everywhere in this block
            def style_rain(val):
                return 'background-color: #ffcccc; color: #990000; font-weight: bold;' if isinstance(val, (int,
                                                                                                           float)) and val >= 70 else ''


            if not df_weather.empty:
                # Display current conditions as metrics
                current = df_weather.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Air Temp", f"{current['Air Temp']:.1f}°C")
                m2.metric("Rain Prob", f"{current['Rain Prob']}%")
                m3.metric("Humidity", current['Humidity'])
                m4.metric("Wind Speed", current['Wind'])

                st.divider()
                st.subheader("Upcoming Forecast Horizon")


                def highlight_alerts(row):
                    color = '#4a1e1e' if row['Rain Prob'] >= 50 else ''
                    return [f'background-color: {color}'] * len(row)


                st.dataframe(
                    df_weather.style.apply(highlight_alerts, axis=1)
                    .format({"Air Temp": "{:.1f}°C", "Track Temp": "{:.1f}°C", "Rain Prob": "{}%"}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # FIX: Only display the warning. Do not try to style the empty dataframe.
                st.warning("Forecast data is currently unavailable.")

        with tab_trends:
            if not df_weather.empty:
                st.subheader("Temperature Trends")
                fig, ax1 = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#0e1117')  # Matching Streamlit dark theme
                ax1.set_facecolor('#1a1a1a')

                # Plotting Track and Ambient Temperatures
                ax1.plot(df_weather["Time"], df_weather["Track Temp"], color='#FF3333', marker='o', label="Track Temp")
                ax1.plot(df_weather["Time"], df_weather["Air Temp"], color='#0066cc', linestyle='--', marker='s',
                         label="Ambient Temp")

                ax1.set_ylabel("Temperature (°C)", color='white')
                ax1.set_xlabel("Time", color='white')
                ax1.tick_params(colors='white')
                ax1.legend(facecolor='#1a1a1a', labelcolor='white')
                plt.xticks(rotation=45, color='white')
                plt.grid(color='#333333', linestyle='--', alpha=0.5)

                st.pyplot(fig)
            else:
                st.warning("Trend data is currently unavailable.")

st.sidebar.write("---")
st.sidebar.write("by andrew pepper")