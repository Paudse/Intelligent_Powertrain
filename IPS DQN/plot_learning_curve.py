import matplotlib.pyplot as plt
import scipy.io 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import io

# fig setup
fig1 = plt.figure(figsize=(6,5.5), dpi=300)

# Load DQN result
data_dqn = scipy.io.loadmat('iteration_1_CYC_UDDSHDV_reward-90.47')
sim_t_dqn = data_dqn['sim_t_dqn'][0]
sim_W_ice_dqn = data_dqn['sim_W_ice_dqn'][0]
sim_T_ice_dqn = data_dqn['sim_T_ice_dqn'][0]
sim_P_ice_dqn = data_dqn['sim_P_ice_dqn'][0]
sim_W_mg2_dqn = data_dqn['sim_W_mg2_dqn'][0]
sim_T_mg2_dqn = data_dqn['sim_T_mg2_dqn'][0]
sim_P_mg2_dqn = data_dqn['sim_P_mg2_dqn'][0]
car_kph_dqn = data_dqn['sim_car_kph_dqn'][0]
sim_T_brake_dqn = data_dqn['sim_T_brake_dqn'][0]
sim_gear_dqn = data_dqn['sim_gear_dqn'][0]
sim_fuel_instant_dqn = data_dqn['sim_fuel_instant_dqn'][0]
sim_fuel_accumulate_dqn = data_dqn['sim_fuel_accumulate_dqn'][0]
sim_battery_charge_dqn = data_dqn['sim_battery_charge_dqn'][0]
sim_battery_soc_dqn = data_dqn['sim_battery_soc_dqn'][0]
sim_reward_dqn = data_dqn['sim_reward_dqn'][0]
sim_acum_reward_dqn = data_dqn['sim_acum_reward_dqn'][0]
cycle_iteration_dqn = data_dqn['cycle_iteration_dqn'][0]
ave_cyc_reward_dqn = data_dqn['ave_cyc_reward_dqn'][0]
cyc_fuel_accumulate_dqn = data_dqn['cyc_fuel_accumulate_dqn'][0]
ave_cyc_speed_difference_kmh_dqn = data_dqn['ave_cyc_speed_difference_kmh_dqn'][0]
sim_action_dqn = data_dqn['sim_action_dqn']
# speed_difference_kmh_dqn = data_dqn['speed_difference_kmh_dqn'][0]

#### Fig1####
plt.subplots_adjust(wspace =1, hspace =0.5)
ax = plt.subplot(3,1,1)
plt.plot(cycle_iteration_dqn,ave_cyc_reward_dqn, label='DQN')
plt.title('(a) Reward', loc='left')
plt.ylabel('')
# plt.xlabel('cycle iteration')
# plt.xlabel('(a)')
plt.xlim(0, max(cycle_iteration_dqn)+1)
# plt.xlim(0, 1000)
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)

# plt.legend(loc='upper left')

ax = plt.subplot(3,1,2)
plt.plot(cycle_iteration_dqn,ave_cyc_speed_difference_kmh_dqn)
plt.title('(b) Speed Deviation', loc='left')
plt.ylabel('km/h')
# plt.xlabel('cycle iteration')
# plt.xlabel('(b)')
plt.xlim(0, max(cycle_iteration_dqn)+1)
# plt.xlim(0, 1000)
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)

ax = plt.subplot(3,1,3)
plt.plot(cycle_iteration_dqn,cyc_fuel_accumulate_dqn)
plt.title('(c) Fuel Consumption', loc='left')
plt.ylabel('g')
plt.xlabel('cycle iteration')
plt.xlim(0, max(cycle_iteration_dqn)+1)
# plt.xlim(0, 1000)
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)

# Save the image in memory in PNG format
png1 = io.BytesIO()
fig1.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
png2.save("save.tiff")
png1.close()

################################################################################################
cycle_name = 'CYC_UDDSHDV'
dir_name = 'advisor_2003_data/drive_cycle/' # import powertrain and cycle data
data = scipy.io.loadmat(dir_name+cycle_name+'.mat')
cyc_mph = data['cyc_mph']
cyc_kph = []
cyc_kph.append(cyc_mph[:,0])
cyc_kph.append(cyc_mph[:,1] * 1.609344)
cycle_speed = cyc_kph[1]
cycle_time = cyc_kph[0]
cyecle_end_time = cycle_time[-1]
print('cycle name: ' + cycle_name)

ice_name = 'FC_PRIUS_JPN'
##ice_name = 'FC_SI95'
##ice_name = 'FC_CI88'
dir_name = 'advisor_2003_data/fuel_converter/'
data = scipy.io.loadmat(dir_name+ice_name+'.mat')
fc_map_spd = data['fc_map_spd'][0] # (rad/s)
fc_map_trq = data['fc_map_trq'][0] # (N*m)
fc_max_trq = data['fc_max_trq'][0] # (N*m)
fc_fuel_map = data['fc_fuel_map'] # (g/s)
fc_co_map = data['fc_co_map'] # (g/s)
fc_hc_map = data['fc_hc_map'] # (g/s)
fc_nox_map = data['fc_nox_map'] # (g/s)

mg2_name = 'MC_PRIUS_JPN'
##mg2_name = 'MC_INSIGHT_draft'
##mg2_name = 'MC_PM49'
dir_name = 'advisor_2003_data/motor/'
data = scipy.io.loadmat(dir_name+mg2_name+'.mat')
mg2_map_spd = data['mc_map_spd'][0] # (rad/s)
mg2_map_trq = data['mc_map_trq'][0] # (N*m)
mg2_max_trq = data['mc_max_trq'][0] # (N*m)
mg2_max_gen_trq = data['mc_max_gen_trq'][0] # (N*m)
mg2_eff_map = data['mc_eff_map'] # (g/s)

fig2 = plt.figure(figsize=(6,4), dpi=300)

plt.subplots_adjust(wspace =0.5, hspace =0.5)

plt.subplot(2,2,1)
plt.title('(a) ICE BSFC (g/kWh)', loc='left')
plt.ylabel('Nm')
plt.xlabel('rpm')

pi = 3.14159
rads2rpm = float(1/(2*pi/60))

X,Y = np.meshgrid(fc_map_spd*rads2rpm, fc_map_trq)
kW_map = np.multiply(X,Y)/1000
# C_bsfc = self.bsfc.contour(X, Y, kW_map, alpha=.75, cmap=plt.cm.hot)
C_bsfc = plt.contour(X, Y, fc_fuel_map.T/kW_map,30, alpha=.75, cmap=plt.cm.hot)
plt.plot([i * rads2rpm for i in sim_W_ice_dqn],sim_T_ice_dqn,"o",color='C2',markersize=2 )
plt.plot(fc_map_spd*rads2rpm,fc_max_trq,'--',color='red',linewidth=3)
# plt.set(title='BSFC (g/kWh)', ylabel='Torque (Nm)', xlabel='Speed (rpm)')
# plt.set_title('BSFC (g/kWh)',fontsize=10,style='italic')
plt.clabel(C_bsfc, inline=False, fontsize=10)
plt.xlim(min(fc_map_spd*rads2rpm), max(fc_map_spd*rads2rpm))


###
plt.subplot(2,2,2)
plt.title('(b) MG Efficiency', loc='left')
plt.ylabel('Nm')
plt.xlabel('rpm')
X,Y = np.meshgrid(mg2_map_spd*rads2rpm, mg2_map_trq)
C_eff = plt.contour(X, Y, mg2_eff_map.T, 10, alpha=.75, cmap=plt.cm.hot)
for ii in range(len(sim_T_mg2_dqn)):
    if sim_T_mg2_dqn[ii] < 0:
        plt.plot(sim_W_mg2_dqn[ii]*rads2rpm,sim_T_mg2_dqn[ii],"o",color='purple',markersize=2 )
    else:
        plt.plot(sim_W_mg2_dqn[ii]*rads2rpm,sim_T_mg2_dqn[ii],"o",color='C2',markersize=2 )
plt.plot(mg2_map_spd*rads2rpm, mg2_max_trq,'--',color='red',linewidth=3)
plt.plot(mg2_map_spd*rads2rpm, mg2_max_gen_trq,'--',color='red',linewidth=3)

# plt.show()

# Save the image in memory in PNG format
png1 = io.BytesIO()
fig2.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
png2.save("save2.tiff")
png1.close()


#### Fig3 ################################################################################################################################
fig3 = plt.figure(figsize=(6,5.5), dpi=300)
plt.subplots_adjust(wspace =1, hspace =0.5)

plt.subplot(3,1,1)
plt.plot(cycle_time,cycle_speed,c = 'steelblue', linewidth=6, label='Target Speed')
plt.plot(sim_t_dqn,car_kph_dqn, label='Vehicle Speed',c = 'C1', linewidth=3)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.7),ncol=3, fancybox=True, shadow=False)
plt.title('(a) Vehicle Speed', loc='left')
plt.ylabel('km/h')
plt.xlim(0, max(sim_t_dqn))
# plt.xlim(0, 1000)
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)

# plt.legend(loc='upper left')

# plt.subplot(3,1,2)
# plt.plot(sim_t_dqn,sim_P_ice_dqn,"o",color='C1',markersize=3 )
# plt.title('(b) ICE Power', loc='left')
# plt.ylabel('kW')
# plt.xlim(0, max(sim_t_dqn))
# # plt.xlim(0, 1000)
# plt.grid(color='gray', linestyle='-', linewidth=0.2)
# ax.spines['top'].set_linewidth(0)
# ax.spines['right'].set_linewidth(0)

plt.subplot(3,1,3)
# plt.plot(sim_t_dqn,sim_P_mg2_dqn, label='DQN')
plt.plot(sim_t_dqn,np.maximum(sim_P_mg2_dqn, 0),"o",color='C1',markersize=3, label='Motor' )
plt.plot(sim_t_dqn,np.minimum(sim_P_mg2_dqn, 0),"o",color='purple',markersize=3, label='Generator' )
# plt.plot(sim_t_dqn,np.maximum(sim_P_mg2_dqn, 0),"o",color='C1',markersize=3, label='Motor' )
plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1.7),ncol=3, fancybox=True, shadow=False)
plt.title('(c) MG Power', loc='left')
plt.ylabel('kW')
plt.xlabel('time (s)')
plt.xlim(0, max(sim_t_dqn))
# plt.xlim(0, 1000)
plt.grid(color='gray', linestyle='-', linewidth=0.2)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
# plt.show()

# Save the image in memory in PNG format
png1 = io.BytesIO()
fig3.savefig(png1, format="png")

# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
png2.save("save3.tiff")
png1.close()