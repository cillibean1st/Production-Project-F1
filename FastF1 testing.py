#https://docs.fastf1.dev/examples/index.html
#pip install matplotlib, fastf1

from matplotlib import pyplot as plt
import fastf1
import fastf1.plotting

fastf1.plotting.setup_mpl(color_scheme='fastf1')

session = fastf1.get_session(2025, 'BAKU', 'R')

session.load()
laps_piastri= session.laps.pick_drivers('PIA').pick_laps(range(1,20))
pia_car_data = laps_piastri.get_car_data()
t = pia_car_data['Time']
vCar = pia_car_data['Speed']

# The rest is just plotting
fig, ax = plt.subplots()
ax.plot(t, vCar, label='Fast')
ax.set_xlabel('Time')
ax.set_ylabel('Speed [Km/h]')
ax.set_title('Piastri is')
ax.legend()
if pia_car_data['COMPOUND'] == "MEDIUM":
    plt.plot(color ='yellow')
elif pia_car_data['COMPOUND'] == "SOFT":
    plt.plot(color ='red')
elif pia_car_data['COMPOUND'] == "Hard":
    plt.plot(color ='yellow')



plt.show()

#shows the schedule of the F1
schedule = fastf1.get_event_schedule(2025)
print(schedule)

event =fastf1.get_event(2025,17)
print(event)
