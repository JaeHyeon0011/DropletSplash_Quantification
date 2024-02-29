import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

import Side_Quantify as quantify
import Calculate as cal

syringeFlowrate = (np.pi/4)*(0.02**2)*(12.05*1e-3)
sf = syringeFlowrate
supply = 3.14159
# %% ImageJ csv file path setting
'''[0]:id [1]:start frame [2]:initial x [3]:initial y [4]: total velocity 
[5]: velocity angle [6]:radius [7]:x velocity [8]:z velocity [9]: y velocity 
정면에 촬영했으므로 가로: x방향, 세로 : z방향 카메라로 다가오는 방향 : y 방향
[7],[8],[9]는 set up에 따라 바뀔 수 있음.

02.26 각도값 계산 안하고, 0으로 통일
'''
ca_0 = "./240216data/ca/0216_front_10mm_ca0_1.csv"
ca_90 = "./240216data/ca/0216_front_10mm_ca90_2.csv"
ca_135 = "./240216data/ca/0216_front_10mm_ca135_2.csv"

ela_1000 = "./240216data/elastic/0216_front_1mm_ca90_1.csv"  # 1 mm ca:90
ela_500 = "./240216data/elastic/0216_front_05mm_ca90_1.csv"  # 0.5 mm ca:90


rough_micro_ver = "./240216data/rough/0216_front_micro_ca90_2.csv"
rough_micro_hor = "./240216data/rough/0226_micro_horizontal_ca90_1.csv"

rough_macro_ver = "./240216data/rough/0216_front_macro_ca90_1.csv"
rough_macro_hor = "./240216data/rough/0226_macro_horizontal_ca90_1.csv"

# [x,y,remain, emit] (g)
ca0_info = [64.231, 48.718, 0.2014, 2.6488]
ca90_info = [62.358, 48.725, 0.1209, 2.7289]
ca135_info = [62.280, 49.736, 0, 2.6688]

ela1000_info = [59.044, 48.366, 0.1423, 2.8]
ela500_info = [57.150, 51.114, 0.0303, 2.962]

microVer_info = [50.590, 49.313, 0.1456, 2.8457]
microHor_info = [55.890, 52.506, 0.23748, 1.5833]

macroVer_info = [57.195, 48.841, 0.07295, 2.8279]
macroHor_info = [56.351, 49.494, 0.5118, 2.2299]

surface_num = 9
# %% Preprocess 1
supply = 3.14159

ca0_arr = quantify.preprocess_intensity(ca_0, ca0_info)
ca90_arr = quantify.preprocess_intensity(ca_90, ca90_info)
ca135_arr = quantify.preprocess_intensity(ca_135, ca135_info)

ela1000_arr = quantify.preprocess_intensity(ela_1000, ela1000_info)
ela500_arr = quantify.preprocess_intensity(ela_500, ela500_info)

microVer_arr = quantify.preprocess_intensity(rough_micro_ver, microVer_info)
microHor_arr = quantify.preprocess_intensity(rough_micro_hor, microHor_info)

macroVer_arr = quantify.preprocess_intensity(rough_macro_ver, macroVer_info)
macroHor_arr = quantify.preprocess_intensity(rough_macro_hor, macroHor_info)
# %% Preprocess 2 ratio calculate
#ca135_info = [62.280, 49.736, 0, 0]

syringeFlowrate = (np.pi/4)*(0.02**2)*(12.05*1e-3)
sf = syringeFlowrate

ca0_list = cal.CalculateRatio(ca0_arr, ca0_info, sf, supplyMass=supply)
ca90_list = cal.CalculateRatio(ca90_arr, ca90_info, sf, supplyMass=supply)
ca135_list = cal.CalculateRatio(ca135_arr, ca135_info, sf, supplyMass=supply)

ela1000_list = cal.CalculateRatio(ela1000_arr, ela1000_info, sf, time_step=0.1)
ela500_list = cal.CalculateRatio(ela500_arr, ela500_info, sf, time_step=0.1)

microVer_list = cal.CalculateRatio(
    microVer_arr, microVer_info, syringeFlowrate=syringeFlowrate, time_step=0.1)
microHor_list = cal.CalculateRatio(
    microHor_arr, microHor_info, syringeFlowrate=syringeFlowrate, time_step=0.1)

macroVer_list = cal.CalculateRatio(
    macroVer_arr, macroVer_info, syringeFlowrate=syringeFlowrate, time_step=0.1)
macroHor_list = cal.CalculateRatio(
    macroHor_arr, macroHor_info, syringeFlowrate=syringeFlowrate, time_step=0.1)
# %%
print(ca0_list[0])
print(ca90_list)
print(ca135_list[0])

print(ela1000_list[0])
print(ela500_list[0])

print(microVer_list)
print(microHor_list)

print(macroVer_list)
print(macroHor_list)
# %%
ca135_info_2 =  [62.280, 49.736, 0, 2.6688]

ca_135_2 = "./240216data/ca/0216_front_10mm_ca135_1.csv"
ca135_arr_2 = quantify.preprocess_intensity(ca_135_2, ca135_info_2)
#%%
ca135_list_2 = cal.CalculateRatio(ca135_arr_2, ca135_info_2, sf, supplyMass=supply)
print(ca135_list_2)

ca_135_3 = "./240216data/ca/0216_front_10mm_ca135_3.csv"
ca135_arr_3= quantify.preprocess_intensity(ca_135_3, ca135_info_2)
#%%
ca135_list_3 = cal.CalculateRatio(ca135_arr_3, ca135_info_2, sf, supplyMass=supply)
print(ca135_list_3)
# %%
print(ca135_list_2[0])
print(ca135_list[0])
# %%
print("\n")

print(ca0_list[4])
print(ca90_list[4])
print(ca135_list[4])
print(ela1000_list[4])
print(ela500_list[4])
print(macroVer_list[4])
print(microVer_list[4])
#%% Plot process 1. diamter-velocity / 2. diameter - angle (x,y) [7],[9]
def Add_totalVelAndAngle(data):
    selected_columns = data[:, 7:10]
    diameter_col = data[:, 6]*2  # unit : mm radius -> diameter
    total_velocity = np.sqrt(np.sum(selected_columns ** 2, axis=1))
    # [9]:y velocity  [7] : x velocity
    angle_xy = np.degrees(np.arctan(np.abs(data[:, 9]) / (data[:, 7])))
    angle_ = np.where(angle_xy > 0, angle_xy, angle_xy + 180)
    angle_xy = angle_
    plot_arr = np.hstack((diameter_col.reshape(-1, 1),
                         total_velocity.reshape(-1, 1), angle_xy.reshape(-1, 1)))

    return plot_arr
# %%
plot_ca0 = Add_totalVelAndAngle(ca0_arr)
plot_ca90 = Add_totalVelAndAngle(ca90_arr)
plot_ca135 = Add_totalVelAndAngle(ca135_arr)

plot_ela1000 = Add_totalVelAndAngle(ela1000_arr)
plot_ela500 = Add_totalVelAndAngle(ela500_arr)

plot_microVer = Add_totalVelAndAngle(microVer_arr)
plot_microHor = Add_totalVelAndAngle(microHor_arr)

plot_macroVer = Add_totalVelAndAngle(macroVer_arr)
plot_macroHor = Add_totalVelAndAngle(macroHor_arr)
# %% Plot Diameter - velocity
plt.style.use(['science', 'notebook', 'grid'])
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.family'] = 'Arial'  # Arial 폰트 사용

'''
plt.figure(figsize=(12,8))
plt.scatter(plot_ca0[:,0], plot_ca0[:,1], label = "$\\theta_{contact}=0\degree$")
plt.scatter(plot_ca90[:,0], plot_ca90[:,1],label = "$\\theta_{contact}=90\degree$")
plt.scatter(plot_ca135[:,0], plot_ca135[:,1],s=20,label = "$\\theta_{contact}=135\degree$")

'''
plt.scatter(plot_ela1000[:, 0], plot_ela1000[:, 1], marker='v', label="Type 4")
plt.scatter(plot_ela500[:, 0], plot_ela500[:, 1], marker='^', label="Type 5")
plt.scatter(plot_microVer[:, 0], plot_microVer[:, 1],marker='s', label="Type 6")
plt.scatter(plot_macroVer[:, 0], plot_macroVer[:, 1],s=20, marker='o', label="Type 7")


plt.xlabel('Droplet Diameter [mm]', fontsize=25, fontweight='bold')
#plt.ylabel('Total Velocity [m/s]',fontsize=25, fontweight = 'bold')
plt.ylabel('Splash Angle [$\degree$]', fontsize=25, fontweight='bold')
plt.xlim(0.1, 1.5)
plt.legend(loc='best', fontsize=20)

'''plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)'''
plt.show()
# %%
rows = 2
cols = 2
plt.rcParams['font.family'] = 'Arial'  # Arial 폰트 사용
fig, axes = plt.subplots(rows, cols, figsize=(14, 12), dpi=500)
colors = ['blue', 'orange', 'green', 'red', 'darkviolet', 'lightgreen', 'pink']

axes[0, 0].scatter(plot_ca0[:, 0], plot_ca0[:, 1], s=40, marker='o',
                   label='Type 1')  # Diameter - velocity
axes[0, 0].scatter(plot_ca90[:, 0], plot_ca90[:, 1], s=25,
                   marker='v', label="Type 2")  # Diameter - velocity
axes[0, 0].scatter(plot_ca135[:, 0], plot_ca135[:, 1], s=5,
                   marker='s', label="Type 3")# Diameter - velocity
axes[0, 0].scatter(plot_ela1000[:, 0], plot_ela1000[:, 1],
                   s=3, marker='v', label="Type 4")
axes[0, 0].scatter(plot_ela500[:, 0], plot_ela500[:, 1], s=1,
                   marker='^', label="Type 5")

axes[0, 0].legend(loc='best', fontsize=15, markerscale=3)


axes[0, 1].scatter(plot_microVer[:, 0], plot_microVer[:, 1],
                   s=4, marker='s', label="Type 6")
axes[0, 1].scatter(plot_microHor[:, 0], plot_microHor[:, 1],
                   s=4, marker='s', label="Type 7")

axes[0, 1].scatter(plot_macroVer[:, 0], plot_macroVer[:, 1],
                   s=4, marker='o', label="Type 8")
axes[0, 1].scatter(plot_macroHor[:, 0], plot_macroHor[:, 1],
                   s=4, marker='o', label="Type 9")
axes[0, 1].legend(loc='lower right', fontsize=15, markerscale=5)

# row 1
axes[1, 0].scatter(plot_ca0[:, 0], plot_ca0[:, 2], s=4, marker='o',
                   label='Type 1')  # Diameter - velocity
axes[1, 0].scatter(plot_ca90[:, 0], plot_ca90[:, 2], s=4,
                   marker='v', label="Type 2")  # Diameter - velocity
axes[1, 0].scatter(plot_ca135[:, 0], plot_ca135[:, 2], s=4,
                   marker='s', label="Type 3")# Diameter - velocity
axes[1, 0].scatter(plot_ela1000[:, 0], plot_ela1000[:, 2],
                   s=8, marker='v', label="Type 4")
axes[1, 0].scatter(plot_ela500[:, 0], plot_ela500[:, 2], s=8,
                   marker='^', label="Type 5")
#axes[0, 0].legend(loc='best', fontsize=15, markerscale=5)


axes[1, 1].scatter(plot_microVer[:, 0], plot_microVer[:, 2],
                   s=4, marker='s', label="Type 6")
axes[1, 1].scatter(plot_microHor[:, 0], plot_microHor[:, 2],
                   s=4, marker='s', label="Type 7")
axes[1, 1].scatter(plot_macroVer[:, 0], plot_macroVer[:, 2],
                   s=4, marker='o', label="Type 8")
axes[1, 1].scatter(plot_macroHor[:, 0], plot_macroHor[:, 2],
                   s=4, marker='o', label="Type 9")
#axes[1, 1].legend(loc='best', fontsize=15, markerscale=5)

# plt.subplots_adjust(wspace=0.4)
axes[0, 0].set_ylabel("Droplet Velocity [m/s]", fontsize=18, fontweight='bold')
axes[0, 1].set_ylabel("Droplet Velocity [m/s]", fontsize=18, fontweight='bold')
axes[1, 0].set_ylabel("Splash Angle [$\degree$]",
                      fontsize=18, fontweight='bold')
axes[1, 1].set_ylabel("Splash Angle [$\degree$]",
                      fontsize=18, fontweight='bold')

axes[0, 0].set_xlabel("Droplet Diameter [mm]", fontsize=18, fontweight='bold')
axes[1, 0].set_xlabel("Droplet Diameter [mm]", fontsize=18, fontweight='bold')
axes[1, 1].set_xlabel("Droplet Diameter [mm]", fontsize=18, fontweight='bold')
axes[0, 1].set_xlabel("Droplet Diameter [mm]", fontsize=18, fontweight='bold')

plt.xlim(0.1, 1.5)

'''plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)'''
plt.show()
#%%
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.family'] = 'Arial'

species = (
    f"Type 1\n{ca0_list[0]:.1f}",f"Type 2\n{ca90_list[0]:.1f}",f"Type 3\n{ca135_list[0]:.1f}",
    f"Type 4\n{ela1000_list[0]:.1f}",f"Type 5\n{ela500_list[0]:.1f}",
    f"Type 6\n{microVer_list[0]:.1f}", f"Type 7\n{microHor_list[0]:.1f}",
    f"Type 8\n{macroVer_list[0]:.1f}",f"Type 9\n{macroHor_list[0]:.1f}"
)


ratio = {
    "Splash Ratio": np.array([ca0_list[1], ca90_list[1],ca135_list[1],ela1000_list[1],ela500_list[1],
                              microVer_list[1],microHor_list[1], macroVer_list[1],macroHor_list[1]]),
    
    "Remain Ratio":  np.array([ca0_list[2], ca90_list[2],ca135_list[2],ela1000_list[2],ela500_list[2],
                              microVer_list[2],microHor_list[2], macroVer_list[2],macroHor_list[2]]),
    
    "emit Ratio":  np.array([ca0_list[3], ca90_list[3],ca135_list[3],ela1000_list[3],ela500_list[3],
                              microVer_list[3],microHor_list[3], macroVer_list[3],macroHor_list[3]])
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(9)

for boolean, weight_count in ratio.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

ax.set_xlabel('Surface Type and Sum of Ratio',fontsize=12, fontweight = 'bold')
ax.set_ylabel('Ratio [-]',fontsize=12, fontweight='bold')
ax.legend(loc="upper right",fontsize=10)
#ax.text(-0.23, 0.7, r'$\varepsilon_{2500}$=1.046', fontsize=25)
#ax.text(0.78, 0.7, r'$\varepsilon_{5000}$=1.068', fontsize=25)
#plt.title("Ratio bar graph",fontsize=25)
plt.grid(False)
plt.show()
#%%
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.family'] = 'Arial'

species = (
    f"Type 1\n{ca0_list[0]:.1f}",f"Type 2\n{ca90_list[0]:.1f}",f"Type 3\n{ca135_list[0]:.1f}",
    f"Type 4\n{ela1000_list[0]:.1f}",f"Type 5\n{ela500_list[0]:.1f}",
    f"Type 6\n{microVer_list[0]:.1f}", f"Type 7\n{microHor_list[0]:.1f}",
    f"Type 8\n{macroVer_list[0]:.1f}",f"Type 9\n{macroHor_list[0]:.1f}"
)

ratio = {
    "Splash Ratio": np.array([ca0_list[4], ca90_list[4],ca135_list[4],ela1000_list[4],ela500_list[4],
                              microVer_list[4],microHor_list[4], macroVer_list[4],macroHor_list[4]]),
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(9)

for boolean, weight_count in ratio.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

#ax.set_title("Number of penguins with above average body mass")
ax.set_ylabel('Ratio [-]',fontsize=15)
ax.legend(loc="upper right",fontsize=10)
#ax.text(-0.23, 0.7, r'$\varepsilon_{2500}$=1.046', fontsize=25)
#ax.text(0.78, 0.7, r'$\varepsilon_{5000}$=1.068', fontsize=25)
#plt.title("Ratio bar graph",fontsize=25)
plt.grid(False)
plt.show()