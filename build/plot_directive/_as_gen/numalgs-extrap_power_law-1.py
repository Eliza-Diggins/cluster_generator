from cluster_generator.radial_profiles import vikhlinin_temperature_profile
from cluster_generator.numalgs import extrap_power_law
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1)
T = vikhlinin_temperature_profile(3.61,0.12,5.00,10.0,1420,0.9747,57,3.88)
x = np.geomspace(1,5000,5000)
y = T(x)

ax.semilogx(x,y,"k:")

xn,yn = extrap_power_law(700,800,-2,x=x,y=y)

ax.semilogx(xn,yn,"r--")

plt.show()