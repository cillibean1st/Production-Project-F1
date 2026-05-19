import streamlit as st
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os, warnings
import matplotlib.pyplot as plt

import charts  # Ensure charts.py is in your directory

# --- 1. SETTINGS & THEME ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="2026 F1 Pit Wall", layout="wide", page_icon="🏎️")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #111111;
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCheckbox,
    [data-testid="stSidebar"] .stSlider {
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

if 'zoomed_view' not in st.session_state:
    st.session_state.zoomed_view = None

# --- 2. DATA & MODELS ---
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

teams = {
    'VER': 'Red Bull Racing', 'HAD': 'Red Bull Racing', 'HAM': 'Ferrari', 'LEC': 'Ferrari',
    'NOR': 'McLaren', 'PIA': 'McLaren', 'RUS': 'Mercedes', 'ANT': 'Mercedes',
    'ALO': 'Aston Martin', 'STR': 'Aston Martin', 'SAI': 'Williams', 'ALB': 'Williams',
    'GAS': 'Alpine', 'COL': 'Alpine', 'OCO': 'Haas F1 Team', 'BEA': 'Haas F1 Team',
    'HUL': 'Audi', 'BOR': 'Audi', 'LAW': 'RB', 'LIN': 'RB',
    'PER': 'Cadillac', 'BOT': 'Cadillac'
}


@st.cache_resource
def load_data_and_models():
    f = "f1_stats_main.csv"
    if not os.path.exists(f): return None, None, None, None
    df = pd.read_csv(f)
    mappings = {}
    compounds = ["SOFT", "MEDIUM", "HARD", "INTER", "WET"]
    for col in ['Track', 'Driver', 'Team', 'Compound']:
        le = LabelEncoder()
        if col == 'Compound':
            combined = list(df[col].astype(str).unique()) + compounds
            le.fit(list(set(combined)))
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


# --- 3. STRATEGY ENGINE ---
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
        if c[1].button("🔙 Exit", key=f"ex_{key}"):
            st.session_state.zoomed_view = None
            st.rerun()
    else:
        if c[1].button("🔍 Zoom", key=f"zm_{key}"):
            st.session_state.zoomed_view = key
            st.rerun()


# --- 4. SIDEBAR & GLOBAL CONTROLS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=100)
st.sidebar.header("Race Control")

testing_mode = st.sidebar.toggle("🛠️ Testing Mode", value=True)
sel_track = st.sidebar.selectbox("Grand Prix", sorted(mappings['Track'].classes_))
sel_drv_code = st.sidebar.selectbox("Driver", list(drivers.keys()), format_func=lambda x: f"{x} - {drivers[x]}")
start_tyre = st.sidebar.selectbox("Starting Tyre", ["SOFT", "MEDIUM", "HARD"])

tr_id = mappings['Track'].transform([sel_track])[0]
dr_id = mappings['Driver'].transform([sel_drv_code])[0]
tm_id = mappings['Team'].transform([teams[sel_drv_code]])[0]
co_id = mappings['Compound'].transform([start_tyre])[0]

try:
    tot_laps = int(full_df[full_df['Track'] == tr_id]['LapNumber'].max())
except:
    tot_laps = 56

st.sidebar.divider()
st.sidebar.subheader("Safety Car Period")
sc_active = st.sidebar.checkbox("SC / VSC Active")
sc_start, sc_end = st.sidebar.slider("SC Window", 1, tot_laps, (15, 20)) if sc_active else (-1, -1)

st.sidebar.divider()
st.sidebar.subheader("Weather Window")
rain_active = st.sidebar.checkbox("Rain in Forecast")
rain_start, rain_end = st.sidebar.slider("Rain Duration", 1, tot_laps, (10, 25)) if rain_active else (-1, -1)
heavy_rain = st.sidebar.checkbox("Heavy Rain (Full Wets)") if rain_active else False

pred_life = int(model_life.predict([[tr_id, dr_id, tm_id, co_id]])[0])
all_strats = get_all_strategies(pred_life, tr_id, dr_id, tm_id, tot_laps, start_tyre, sc_active, sc_start, sc_end,
                                rain_active, heavy_rain, rain_start, rain_end)

# --- 5. MAIN DASHBOARD ---
st.title(f"PIT WALL: {sel_track} 2026")
sel_strat_key = st.selectbox("Active Strategy Selection", list(all_strats.keys()))
active_strat = all_strats[sel_strat_key]

m1, m2, m3 = st.columns(3)
with m1: st.metric("Driver", f"{sel_drv_code}", teams[sel_drv_code])
with m2: st.metric("Predicted Tyre Life", f"{pred_life} Laps")
with m3: st.metric("Total Stops", f"{len(active_strat['laps'])}")

st.divider()

if st.session_state.zoomed_view is None:
    r1c1, r1c2 = st.columns(2);
    r2c1, r2c2 = st.columns(2);
    r3c1, r3c2 = st.columns(2)
    with r1c1:
        zoom_btn("Strategy Forecast", "tele")
        st.pyplot(charts.plot_telemetry(drivers[sel_drv_code], sel_track, active_strat['laps'], active_strat['tyres'],
                                        tot_laps, sc_start, model_time, tr_id, dr_id, tm_id, mappings))
    with r1c2:
        zoom_btn("Track Map Monitor", "map")
        st.info("Visual Positioning standby. Zoom to initialize GPS.")
    with r2c1:
        zoom_btn("Manual Planner", "plan")
        st.write("Create and simulate custom pit-stop profiles.")
    with r2c2:
        zoom_btn("Weather Report", "weather")
        st.caption("Current: 24°C | Track: 38°C")
    with r3c1:
        zoom_btn("Historical Data", "hist")
        st.pyplot(charts.plot_sawtooth(full_df, sel_track, sel_drv_code, mappings))
    with r3c2:
        zoom_btn("Live Timing", "live")
        st.table(pd.DataFrame({"Pos": [1, 2, 3], "Driver": ["VER", "HAM", "LEC"], "Gap": ["--", "+1.2", "+4.5"]}))

else:
    z = st.session_state.zoomed_view

    if z == "plan":
        zoom_btn("🛠️ Manual Strategy Planner", "plan")
        col_inputs, col_results = st.columns([1, 2])
        with col_inputs:
            st.markdown("### Stint Configuration")
            num_stops = st.number_input("Number of Pit Stops", 1, 3, 1)
            custom_strategy = []
            current_lap = 0

            s1_laps = st.slider("Stint 1 Length", 1, tot_laps - num_stops, 15)
            s1_tyre = st.selectbox("Stint 1 Tyre", ["SOFT", "MEDIUM", "HARD"], key="s1t")
            custom_strategy.append({"laps": s1_laps, "tyre": s1_tyre})
            current_lap += s1_laps

            for i in range(num_stops):
                remaining = tot_laps - current_lap
                if remaining <= 0: break
                st.divider()
                s_laps = st.slider(f"Stint {i + 2} Length", 1, remaining, remaining)
                s_tyre = st.selectbox(f"Stint {i + 2} Tyre", ["SOFT", "MEDIUM", "HARD"], key=f"s{i + 2}t")
                custom_strategy.append({"laps": s_laps, "tyre": s_tyre})
                current_lap += s_laps

        with col_results:
            st.markdown("### Performance Projection")
            sim_laps, sim_times = [], []
            lap_counter = 1
            for idx, stint in enumerate(custom_strategy):
                cid = mappings['Compound'].transform([stint['tyre']])[0]
                for age in range(1, stint['laps'] + 1):
                    if lap_counter > tot_laps: break
                    pred = model_time.predict([[tr_id, dr_id, tm_id, cid, age, idx + 1]])[0]
                    sim_times.append(pred);
                    sim_laps.append(lap_counter)
                    lap_counter += 1

            pit_loss = num_stops * 22.0
            total_race_s = sum(sim_times) + pit_loss
            avg_lt = np.mean(sim_times)

            m_c1, m_c2 = st.columns(2)
            m_c1.metric("Est. Total Race Time", f"{int(total_race_s // 60)}m {total_race_s % 60:.2f}s")
            m_c2.metric("Est. Average Lap Time", f"{avg_lt:.3f}s")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sim_laps, sim_times, color='gold', linewidth=2, label="Pace Projection")
            slap = 0
            for s in custom_strategy[:-1]:
                slap += s['laps']
                ax.axvline(x=slap, color='red', linestyle='--', alpha=0.4)
            ax.set_facecolor('#0e1117');
            fig.patch.set_facecolor('#0e1117')
            ax.set_ylabel("Lap Time (s)", color='white');
            ax.set_xlabel("Lap Number", color='white')
            ax.tick_params(colors='white');
            ax.legend();
            ax.grid(alpha=0.1)
            st.pyplot(fig)

    elif z == "live":
        zoom_btn("🔴 Live Mission Control", "live")
        st.markdown("### Telemetry Analysis Engine")
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
        with ctrl1:
            main_lap = st.selectbox("Target Lap", range(1, tot_laps + 1), index=min(10, tot_laps - 1))
            compare_on = st.toggle("Enable Rival Comparison")
        with ctrl2:
            comp_drv = st.selectbox("Compare Against", list(drivers.keys()), index=0) if compare_on else sel_drv_code
            comp_lap = st.selectbox("Rival Lap", range(1, tot_laps + 1),
                                    index=min(11, tot_laps - 1)) if compare_on else None
        with ctrl3:
            view_mode = st.radio("Resolution Mode", ["Sectors", "Segments"], horizontal=True)

        p_compound, p_life, p_dist, p_speed = start_tyre, main_lap, None, None
        c_compound, c_life, c_dist, c_speed = "MEDIUM", comp_lap, None, None

        if not testing_mode:
            try:
                session = fastf1.get_session(2026, sel_track, 'R')
                session.load(telemetry=True, laps=True)
                p_lap_obj = session.laps.pick_driver(sel_drv_code).pick_lap(main_lap)
                p_tel = p_lap_obj.get_telemetry().add_distance()
                p_compound, p_life, p_dist, p_speed = p_lap_obj['Compound'], p_lap_obj['TyreLife'], p_tel['Distance'], \
                p_tel['Speed']
                if compare_on:
                    c_lap_obj = session.laps.pick_driver(comp_drv).pick_lap(comp_lap)
                    c_tel = c_lap_obj.get_telemetry().add_distance()
                    c_compound, c_life, c_dist, c_speed = c_lap_obj['Compound'], c_lap_obj['TyreLife'], c_tel[
                        'Distance'], c_tel['Speed']
            except:
                st.error("⚠️ It doesn't seem like there is a race live right now. Please revert to Testing Mode.")
                testing_mode = True

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"**PRIMARY: {sel_drv_code} (Lap {main_lap})**")
            sc1, sc2 = st.columns(2)
            sc1.metric("Tyre", p_compound);
            sc2.metric("Life", f"{int(p_life)} L")
        if compare_on:
            with m_col2:
                st.markdown(f"**RIVAL: {comp_drv} (Lap {comp_lap})**")
                rc1, rc2 = st.columns(2)
                rc1.metric("Tyre", c_compound);
                rc2.metric("Life", f"{int(c_life)} L")

        st.divider()
        col_list, col_graph = st.columns([1.2, 2.3])
        with col_list:
            st.subheader(f"Timing Data ({view_mode})")
            if view_mode == "Sectors":
                rows = ["Sector 1", "Sector 2", "Sector 3"]
                v1, v2 = [28.45, 33.12, 22.51], [28.61, 32.95, 22.72]
            else:
                rows = [f"Segment {i}" for i in range(1, 7)]
                v1 = [round(np.random.uniform(5.0, 12.0), 2) for _ in range(6)]
                v2 = [round(np.random.uniform(5.0, 12.0), 2) for _ in range(6)]
            timing_data = {"Track Point": rows, f"{sel_drv_code}": [f"{v}s" for v in v1]}
            if compare_on:
                timing_data[f"{comp_drv}"] = [f"{v}s" for v in v2]
                deltas = [round(v1[i] - v2[i], 3) for i in range(len(v1))]
                timing_data["Delta"] = [f"+{d}" if d > 0 else f"{d}" for d in deltas]
            st.dataframe(pd.DataFrame(timing_data), use_container_width=True, hide_index=True)

        with col_graph:
            fig, ax = plt.subplots(figsize=(10, 5))
            if testing_mode:
                p_dist = np.linspace(0, 5000, 500)
                p_speed = 200 + 100 * np.sin(p_dist / 400) + 10 * np.cos(p_dist / 150)
                if compare_on:
                    c_dist, c_speed = p_dist, 195 + 105 * np.sin(p_dist / 410) + 8 * np.cos(p_dist / 140)
            ax.plot(p_dist, p_speed, color='cyan', label=f"{sel_drv_code}", linewidth=2.5)
            if compare_on:
                ax.plot(c_dist, c_speed, color='#FF0000', label=f"{comp_drv}", linestyle='--', linewidth=1.5)
                ax.fill_between(p_dist, p_speed, c_speed, where=(p_speed >= c_speed), color='cyan', alpha=0.15)
                ax.fill_between(p_dist, p_speed, c_speed, where=(p_speed < c_speed), color='red', alpha=0.15)
            for d in ([1600, 3300] if view_mode == "Sectors" else np.linspace(500, 4500, 8)):
                ax.axvline(x=d, color='white', alpha=0.2, linestyle=':')
            ax.set_facecolor('#0e1117');
            fig.patch.set_facecolor('#0e1117')
            ax.set_ylabel("Speed (km/h)", color='white');
            ax.set_xlabel("Distance (m)", color='white')
            ax.tick_params(colors='white');
            ax.legend();
            st.pyplot(fig)

    elif z == "tele":
        zoom_btn("Strategy Telemetry", "tele")
        st.pyplot(charts.plot_telemetry(drivers[sel_drv_code], sel_track, active_strat['laps'], active_strat['tyres'],
                                        tot_laps, sc_start, model_time, tr_id, dr_id, tm_id, mappings))
    elif z == "hist":
        zoom_btn("Historical Stint Analysis", "hist")
        st.pyplot(charts.plot_sawtooth(full_df, sel_track, sel_drv_code, mappings))

st.sidebar.write("---")
st.sidebar.write("by andrew pepper")