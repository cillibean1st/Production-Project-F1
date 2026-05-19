import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# makes the graph dark mode
plt.style.use('dark_background')


# helper to make the y-axis look like actual lap times
#accounts for fuel
# changes time in to HH:MM:SS:ss
def _time_fmt(x, _):
    m = int(x // 60)
    s = x % 60
    if m > 0: return f"{m}:{s:05.2f}"
    return f"{s:.2f}"


def plot_telemetry(drv_name, track_name, pit_laps, compounds, total_laps, sc_lap, m_time, track_v, driver_v, team_v, encs):
    f, ax = plt.subplots(figsize=(10, 4))

    # f1 tyre colours
    compound_map = {'SOFT': 'Red', 'MEDIUM': 'Yellow', 'HARD': 'White', 'INTER': 'Green', 'WET': 'Blue'}

    cur_lap = 1
    # loop thru the stints to draw the deg curves
    for i, comp in enumerate(compounds):
        end_lap = pit_laps[i] if i < len(pit_laps) else total_laps
        if cur_lap > total_laps: break
        if end_lap > total_laps: end_lap = total_laps

        laps_arr = np.arange(cur_lap, end_lap + 1)

        # rough base pace
        base_time = 85.0
        if comp == 'SOFT':
            base_time = 84.0
        elif comp == 'HARD':
            base_time = 86.0
        elif comp == 'INTER':
            base_time = 92.0

        # rough deg factor
        deg = 0.08 if comp == 'SOFT' else (0.05 if comp == 'MEDIUM' else 0.03)
        if comp in ['INTER', 'WET']: deg = 0.06

        times = []
        wear = 0.0

        # we need the encoded compound for the ML model
        c_v = encs['Compound'].transform([comp])[0]

        for lap in laps_arr:
            is_sc = (sc_lap > 0 and sc_lap <= lap <= sc_lap + 3)
            wear += (deg * 0.2 if is_sc else deg)

            # fuel burn correction (cars get lighter as race goes on)
            # matching the 0.01 factor used in plot_sawtooth
            fuel_effect = lap * -0.01

            # if this is the pit lap, use the ML model to predict the massive time loss
            if lap == end_lap and i < len(pit_laps):
                # predict using: Track, Driver, Team, Compound, TyreLife, Stint
                tyre_age = lap - cur_lap + 1
                pit_time = m_time.predict([[track_v, driver_v, team_v, c_v, tyre_age, i + 1]])[0]
                times.append(pit_time)
            else:
                # include fuel_effect in the calculation
                times.append(base_time + wear + fuel_effect + (20.0 if is_sc else 0.0))

        ax.plot(laps_arr, times, label=f"Stint {i + 1}: {comp}", color=compound_map.get(comp, 'white'), lw=2, marker='.')

        # draw a line where the pit stop is
        if i < len(pit_laps):
            ax.axvline(x=end_lap, color='gray', linestyle=':', alpha=0.5)

        cur_lap = end_lap + 1

    ax.set_title(f"Predicted Pace - {drv_name}: {track_name}")
    if sc_lap > 0:
        ax.axvspan(sc_lap, sc_lap + 3, color='yellow', alpha=0.15, label='Safety Car Window')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.yaxis.set_major_formatter(FuncFormatter(_time_fmt))
    f.tight_layout()
    return f


def plot_sawtooth(df, track_name, drv_code, mappings):
    f, ax = plt.subplots(figsize=(10, 4))

    # Convert the string names to the encoded integers used in the dataframe
    try:
        track = mappings['Track'].transform([track_name])[0]
        driver = mappings['Driver'].transform([drv_code])[0]
    except Exception as e:
        ax.text(0.5, 0.5, f"Mapping Error: {e}", ha='center')
        return f

    # Filter using the encoded integers
    sub = df[(df['Track'] == track) & (df['Driver'] == driver)].copy()

    if sub.empty or len(sub) < 5:
        ax.text(0.5, 0.5, "Insufficient historical data for this selection", ha='center')
        return f

    # Grab the most recent year
    y_max = sub['Year'].max()
    sub = sub[sub['Year'] == y_max]

    # Fuel burn correction
    sub['Adj_Time'] = sub['Secs'] + (sub['LapNumber'] * 0.0045)

    # Decode compounds for the legend/colors
    # We create a mapping back to strings for the plotting logic
    compound_map = {'SOFT': 'Red', 'MEDIUM': 'Yellow', 'HARD': 'White', 'INTER': 'Green', 'WET': 'Blue'}
    #c_val = compound value
    for c_val in sub['Compound'].unique():
        # Inverse transform to get 'SOFT', 'MEDIUM', etc.
        c_text = mappings['Compound'].inverse_transform([int(c_val)])[0]
        c_data = sub[sub['Compound'] == c_val]
        ax.scatter(c_data['LapNumber'], c_data['Adj_Time'], label=c_text, color=compound_map.get(c_text, 'white'), alpha=0.6)

    # Trend lines per stint
    for s in sub['Stint'].unique():
        s_data = sub[sub['Stint'] == s]
        if len(s_data) > 3:
            c_val = s_data['Compound'].iloc[0]
            c_text = mappings['Compound'].inverse_transform([int(c_val)])[0]
            z = np.polyfit(s_data['LapNumber'], s_data['Adj_Time'], 1)
            p = np.poly1d(z)
            ax.plot(s_data['LapNumber'], p(s_data['LapNumber']), color=compound_map.get(c_text, 'white'), ls='--')

    ax.set_title(f"Historical Deg ({y_max}): {drv_code} @ {track_name}")
    ax.yaxis.set_major_formatter(FuncFormatter(_time_fmt))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    f.tight_layout()
    return f