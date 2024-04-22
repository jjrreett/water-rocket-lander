# assume pressure is constant
# then thrust is constant
# and q is constant 
# acceleration = (f / (m_dry + m_water)) - g
# m_water = m_water_0 - q * t
# acceleration = (f / (m_dry + m_water_0 - q * t)) - g
# delta_v = -f/q * ln(m_dry + m_water_0 - q * t) - g*t

# plug in 0 = m_water_0 - q * t -> t = m_water_0 / q
# t_fire = m_water / q
# delta_v =  f/q * ln( (m_dry + m_water_0) / m_dry ) - g * m_water_0/q)


# v(t) = v0 + f/q * ln(m0 / (m0 - q * t)) - g*t
# h(t) = h0 + v0*t - g/2*t**2 + f/q * ((m0 - q*t)/q * ln((m0 - q*t)/m0) + t)


import numpy as np 

g = 9.81 

# units
mm = 0.001
psi = 6890
liter = 0.001  # m3
kg = 1
m = 1
m2 = m * m
m3 = m2 * m
s = 1
s2 = s * s

# constants
g = 9.81 * m / s2  # acceleration due to gravity, m/s^2
gamma = 1.4  # adiabatic constant for air
rho_water = 1000 * kg / m3

# constant params
area_nozzle = 3.14 * (6 * mm) ** 2 / 4

# varying params
pressure_init = 100 * psi
vol_water_0 = 1 * liter
vol_air_0 = 1 * liter
m_dry = 1 * kg

pressure = 100 * psi
area_nozzle = 3.14 * (6 * mm) ** 2 / 4
dm_water = rho_water * area_nozzle * np.sqrt(2 * pressure / rho_water)
thrust = dm_water * np.sqrt(2 * pressure / rho_water)

m_water = rho_water * vol_water_0
delta_v = thrust/dm_water * np.log( (m_dry + m_water) / m_dry ) - g * m_water/dm_water

def how_far_will_i_fall(thrust, q, v0, m_water, m_dry):
    m0 = m_water + m_dry 
    t = m_water / q 
    return v0*t - g/2*t**2 + thrust/q * ((m0 - q*t)/q * np.log((m0 - q*t)/m0) + t)

print(delta_v)
print(how_far_will_i_fall(thrust, dm_water, -delta_v, rho_water * vol_water_0, m_dry))