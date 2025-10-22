
"""
Created on Sat May 10 19:23:11 2025

@author: liampiper
"""

"""
Mars Communications Simulation (Probe-Orbiter-Earth)

Features:
- UHF (433 MHz) and LoRa (915 MHz) links from Mars surface to 250 km orbiter
- X-band Earth relay from orbiter using 200W and 70m DSN dish
- Static 10-minute pass + dynamic elevation-arc data volume estimation
- Command-line tables + plots

Note LoRa Dynamic Profile:
- 10–30°: 1.2 kbps @ 3.0 W
- 30–60°: 5.47 kbps @ 2.0 W
- 60–90°: 10.4 kbps @ 1.0 W

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Constants
# -------------------------------
c = 3e8
mars_radius_km = 3390
orbiter_altitude_km = 250
system_losses_dB = 2
latitudes = [-90, -45, 0, 45, 90]
freqs = {'433 MHz': 433e6, '915 MHz': 915e6}
tx_power_dBm = {'433 MHz': 10 * np.log10(3 * 1000), '915 MHz': 10 * np.log10(1 * 1000), 'Orbiter': 10 * np.log10(200 * 1000)}
rx_sensitivity_dBm = {'433 MHz': -120, '915 MHz': -130}
probe_gain_dBi = 3
orbiter_gain_dBi = 7

bitrate_static = 5470
pass_duration_static_sec = 10 * 60

# -------------------------------
# Elevation Profile Setup
# -------------------------------
time_steps = 200
pass_duration_total_sec = 20 * 60
times = np.linspace(0, pass_duration_total_sec, time_steps)
elevations_deg = 90 * np.sin(np.pi * times / pass_duration_total_sec)

def get_dynamic_profile(elev_deg):
    if elev_deg < 10:
        return 0, 0
    elif elev_deg < 30:
        return 1200, 3.0
    elif elev_deg < 60:
        return 5470, 2.0
    else:
        return 10400, 1.0

# -------------------------------
# Link Budget Calculations
# -------------------------------
def slant_range_km(alt_km):
    return np.sqrt((mars_radius_km + alt_km)**2 - mars_radius_km**2)

def fspl_dB(freq_hz, dist_m):
    return 20 * np.log10(dist_m) + 20 * np.log10(freq_hz) - 147.55

range_km = round(slant_range_km(orbiter_altitude_km), 1)
range_m = range_km * 1000
link_results = []
uplink_plot = []
downlink_plot = []

for lat in latitudes:
    for freq_label, freq_hz in freqs.items():
        fspl = fspl_dB(freq_hz, range_m)
        rx_uplink_dBm = tx_power_dBm[freq_label] + probe_gain_dBi + orbiter_gain_dBi - fspl - system_losses_dB
        rx_downlink_dBm = tx_power_dBm['Orbiter'] + orbiter_gain_dBi + probe_gain_dBi - fspl - system_losses_dB
        margin_uplink = rx_uplink_dBm - rx_sensitivity_dBm[freq_label]
        margin_downlink = rx_downlink_dBm - rx_sensitivity_dBm[freq_label]
        uplink_ok = margin_uplink > 0
        downlink_ok = margin_downlink > 0

        if freq_label == '915 MHz':
            data_static_MB = (bitrate_static * pass_duration_static_sec) / 8 / 1e6
        else:
            data_static_MB = 0

        total_bits = 0
        total_energy = 0
        valid_time = 0
        for i in range(len(elevations_deg) - 1):
            elev = elevations_deg[i]
            dt = times[i+1] - times[i]
            bitrate, power = get_dynamic_profile(elev)
            if bitrate > 0:
                total_bits += bitrate * dt
                total_energy += power * dt
                valid_time += dt

        if freq_label == '915 MHz':
            data_dynamic_MB = total_bits / 8 / 1e6
            avg_tx_power = total_energy / valid_time if valid_time > 0 else 0
        else:
            data_dynamic_MB = 0
            avg_tx_power = 0

        link_results.append({
            'Latitude (deg)': lat,
            'Radio': freq_label,
            'Range (km)': range_km,
            'Uplink Margin (dB)': round(margin_uplink, 2),
            'Downlink Margin (dB)': round(margin_downlink, 2),
            'Uplink OK': uplink_ok,
            'Downlink OK': downlink_ok,
            'Static Data (MB)': round(data_static_MB, 3),
            'Dynamic Data (MB)': round(data_dynamic_MB, 3),
            'Dynamic TX Power (W)': round(avg_tx_power, 2),
            'Dynamic Pass Duration (s)': round(valid_time, 1)
        })

        if freq_label == '433 MHz':
            uplink_plot.append((lat, margin_uplink))
            downlink_plot.append((lat, margin_downlink))

# -------------------------------
# Earth X-Band Link (to be modeled better soon)
# -------------------------------
k = 1.38e-23
freq_xband = 8.4e9
wavelength = c / freq_xband
antenna_eff = 0.55
dsn_eff = 0.55
tx_power_watts = 200
tx_power_dBm = 10 * np.log10(tx_power_watts * 1000)
dsn_diameter = 70
receiver_threshold_dBm = -150
T_sys = 300
bitrate = 5470
distance_cases_km = {
    "Minimum (Opposition)": 56e6,
    "Average": 225e6,
    "Maximum (Conjunction)": 401e6
}
dish_sizes = np.linspace(0.5, 5.0, 100)

def parabolic_gain(d_m, eff, wavelength_m):
    gain = eff * (np.pi * d_m / wavelength_m)**2
    return 10 * np.log10(gain)

dsn_gain_dBi = parabolic_gain(dsn_diameter, dsn_eff, wavelength)
deep_space_results = {}

for label, d_km in distance_cases_km.items():
    d_m = d_km * 1000
    fspl = 20 * np.log10(d_m) + 20 * np.log10(freq_xband) - 147.55
    link_margins = []
    ebn0_margins = []
    for D in dish_sizes:
        orbiter_gain_dBi = parabolic_gain(D, antenna_eff, wavelength)
        rx_power_dBm = tx_power_dBm + orbiter_gain_dBi + dsn_gain_dBi - fspl - system_losses_dB
        link_margin = rx_power_dBm - receiver_threshold_dBm
        link_margins.append(link_margin)

        rx_power_watts = 10 ** ((rx_power_dBm - 30) / 10)
        noise_watts = k * T_sys * bitrate
        snr_linear = rx_power_watts / noise_watts
        ebn0_dB = 10 * np.log10(snr_linear)
        ebn0_margin = ebn0_dB - 9.6
        ebn0_margins.append(ebn0_margin)

    deep_space_results[label] = {
        "dish_sizes": dish_sizes,
        "link_margin": link_margins,
        "ebn0_margin": ebn0_margins
    }

# -------------------------------
# Radio Occultation Inversion
# -------------------------------
refractivity_surface = 300
scale_height_km = 10
altitudes_km = np.linspace(0, 100, 101)
refractivity_profile = refractivity_surface * np.exp(-altitudes_km / scale_height_km)
bending_angle_mrad = np.gradient(refractivity_profile, altitudes_km) * 1e-3

occultation_results = pd.DataFrame({
    "Altitude (km)": altitudes_km,
    "Refractivity (N-units)": refractivity_profile,
    "Bending Angle (mrad)": bending_angle_mrad
})

# -------------------------------
# Terminal Output
# -------------------------------
print("\nPROBE ↔ ORBITER LINK BUDGET (UHF + LoRa)")
print(f"{'Latitude (deg)':>14} {'Radio':>9} {'Range (km)':>12} {'Uplink Margin (dB)':>21} {'Downlink Margin (dB)':>23} {'Uplink OK':>11} {'Downlink OK':>13}")
for row in link_results:
    print(f"{row['Latitude (deg)']:>14} {row['Radio']:>9} {row['Range (km)']:>12.1f} {row['Uplink Margin (dB)']:>21.2f} {row['Downlink Margin (dB)']:>23.2f} {str(row['Uplink OK']):>11} {str(row['Downlink OK']):>13}")

print("\nSTATIC + DYNAMIC PASS DATA VOLUME AND POWER (LoRa only)")
print(f"{'Latitude':>9} {'Static MB':>12} {'Dynamic MB':>14} {'Dyn. Avg Power (W)':>20} {'Pass Time (s)':>16}")
for row in link_results:
    if row['Radio'] == '915 MHz':
        print(f"{row['Latitude (deg)']:>9} {row['Static Data (MB)']:>12.3f} {row['Dynamic Data (MB)']:>14.3f} {row['Dynamic TX Power (W)']:>20.2f} {row['Dynamic Pass Duration (s)']:>16.1f}")

print("\nEARTH X-BAND LINK — REQUIRED ORBITER DISH SIZES")
print(f"{'Distance':<25} {'Dish for Link Margin ≥0 dB':<30} {'Dish for Eb/N0 ≥0 dB':<30}")
print("-" * 85)
for label, data in deep_space_results.items():
    lm_array = np.array(data['link_margin'])
    eb_array = np.array(data['ebn0_margin'])
    try:
        lm_dish = dish_sizes[lm_array >= 0][0]
    except IndexError:
        lm_dish = 'Not achievable'
    try:
        eb_dish = dish_sizes[eb_array >= 0][0]
    except IndexError:
        eb_dish = 'Not achievable'
    print(f"{label:<25} {str(round(lm_dish,2)) + ' m' if isinstance(lm_dish, float) else lm_dish:<30} {str(round(eb_dish,2)) + ' m' if isinstance(eb_dish, float) else eb_dish:<30}")

print("\nRADIO OCCULTATION INVERSION SAMPLE:")
print(occultation_results.head(5).to_string(index=False))

# -------------------------------
# Plotting
# -------------------------------
uplink_plot.sort()
downlink_plot.sort()
lats, uplink_vals = zip(*uplink_plot)
_, downlink_vals = zip(*downlink_plot)

plt.figure()
plt.plot(lats, uplink_vals, label="Uplink (433 MHz)")
plt.plot(lats, downlink_vals, label="Downlink (433 MHz)")
plt.axhline(0, color='gray', linestyle='--')
plt.title("433 MHz Link Margins vs. Latitude")
plt.xlabel("Latitude (deg)")
plt.ylabel("Link Margin (dB)")
plt.grid(True)
plt.legend()

for label in deep_space_results:
    data = deep_space_results[label]
    plt.figure()
    plt.plot(data["dish_sizes"], data["link_margin"])
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"X-Band Link Margin vs. Dish Size\n{label}")
    plt.xlabel("Orbiter Dish Diameter (m)")
    plt.ylabel("Link Margin (dB)")
    plt.grid(True)

    plt.figure()
    plt.plot(data["dish_sizes"], data["ebn0_margin"])
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"X-Band Eb/N₀ Margin vs. Dish Size\n{label}")
    plt.xlabel("Orbiter Dish Diameter (m)")
    plt.ylabel("Eb/N₀ Margin (dB)")
    plt.grid(True)

# Radio occultation plots
plt.figure()
plt.plot(refractivity_profile, altitudes_km)
plt.gca().invert_yaxis()
plt.title("Refractivity Profile (Mars Atmosphere)")
plt.xlabel("Refractivity (N-units)")
plt.ylabel("Altitude (km)")
plt.grid(True)

plt.figure()
plt.plot(bending_angle_mrad, altitudes_km)
plt.gca().invert_yaxis()
plt.title("Bending Angle Profile (Radio Occultation)")
plt.xlabel("Bending Angle (mrad)")
plt.ylabel("Altitude (km)")
plt.grid(True)

plt.show()
