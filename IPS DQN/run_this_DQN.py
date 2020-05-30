from RL_brain import DeepQNetwork
import numpy as np
import scipy.io
from matplotlib import pyplot as plt  
from scipy import interpolate
import os

################# Simulation Setting #################
cycle_target_speed = 0
time = 0
cycle_dt = 0.5
number_show = 20000
cycle_round = 0
end_cycle_round = 2000
print('total cycle round:', end_cycle_round)

################# Learning Setting #################
set_learning_rate = 0.001 # 0.01 ~ 0.0001
set_reward_decay = 0.9# 0.9
set_e_greedy = 0.9# 0.9
set_replace_target_iter = 50
set_memory_size = 2000

################# Select Powertrain #################
# vehicle_name = 'NV_ICE_5AT'
# vehicle_name = 'NV_ICE_CVT'
# vehicle_name = 'EV_MG2'
# vehicle_name = 'EV_2MG_MG1_MG2'
# vehicle_name = 'EV_MG2_2AT'
vehicle_name = 'Hybrid_ICE_5AT_MG2'
# vehicle_name = 'Hybrid_ICE_CVT_MG2'
# vehicle_name = 'Hybrid_ICE_MG2_5AT'
# vehicle_name = 'Hybrid_ICE_MG2_CVT'
# vehicle_name = 'Hybrid_Power_Split_ICE_MG1_MG2'

vehicle_type = 'car_env_'+ vehicle_name

################# Select Drive Cycle #################
# cycle_name = 'CYC_COMMUTER'
# cycle_name = 'CYC_ECE'
# cycle_name = 'CYC_EUDC'
# cycle_name = 'CYC_1015_6PRIUS'
# cycle_name = 'CYC_ARTERIAL'
# cycle_name = 'cyc_cleveland'

# cycle_name = 'CYC_HWFET'
# cycle_name = 'CYC_US06_HWY'
# cycle_name = 'CYC_Cruise3'

cycle_name = 'CYC_UDDSHDV'
# cycle_name = 'CYC_UDDS'
# cycle_name = 'CYC_LA92'
# cycle_name = 'CYC_INDIA_URBAN_SAMPLE'
# cycle_name = 'CYC_CSHVR_Vehicle'

# cycle_name = 'Holomatic'
if cycle_name == 'Holomatic':
	# holomatic_data_name = '2019-03-11-15-22-39_4'
	# holomatic_data_name = '2019-03-11-15-52-39_5'
  # holomatic_data_name = '2019-03-11-16-22-39_6'
  # holomatic_data_name = '2019-02-25-09-45-31_0'
  # holomatic_data_name = '2019-02-25-10-15-32_1'
  # holomatic_data_name = '2019-02-25-10-45-32_2'
  holomatic_data_name = '2019-02-14-13-54-02'
  # holomatic_data_name = ''

  from numpy import genfromtxt
  ms2kmh = 60*60/1000
  holomatic_data = genfromtxt('holomatic_data/vehicle_info_'+holomatic_data_name+'.csv', delimiter=',')
  cycle_speed = holomatic_data[1::50,8]*ms2kmh
  cycle_speed = [round(cycle_speed[i],0) for i in range(0,len(cycle_speed))]
  cycle_time = holomatic_data[1::50,0]-holomatic_data[1,0]
  cycle_time = [round(cycle_time[i],0) for i in range(0,len(cycle_time))]
  print('cycle name: ' + cycle_name+'_'+holomatic_data_name)
else:
	dir_name = 'advisor_2003_data/drive_cycle/'
	data = scipy.io.loadmat(dir_name+cycle_name+'.mat')
	cyc_mph = data['cyc_mph']
	cyc_kph = []
	cyc_kph.append(cyc_mph[:,0])
	cyc_kph.append(cyc_mph[:,1] * 1.609344)
	cycle_speed = cyc_kph[1]
	cycle_time = cyc_kph[0]
	print('cycle name: ' + cycle_name)

cycle_data_dt = round(cycle_time[2]-cycle_time[1],2)
cycle_end_time = cycle_time[-1]

# plt.plot(cycle_time, cycle_speed)
# plt.title(cycle_name)
# plt.ylabel('speed (km/h)')
# plt.xlabel('time (s)')
# plt.show()

cycle_total_step_number = ((len(cycle_speed)-1)*cycle_data_dt/cycle_dt)+1 # total number of steps in one cycle
save_step = cycle_total_step_number * number_show # the step to save the NN weights and bias

################# Select ICE #################
ice_name = 'FC_PRIUS_JPN'
# ice_name = 'FC_SI95'
# ice_name = 'FC_CI88'
dir_name = 'advisor_2003_data/fuel_converter/'
data = scipy.io.loadmat(dir_name+ice_name+'.mat')
fc_map_spd = data['fc_map_spd'][0] # (rad/s)
fc_map_trq = data['fc_map_trq'][0] # (N*m)
fc_max_trq = data['fc_max_trq'][0] # (N*m)
fc_fuel_map = data['fc_fuel_map'] # (g/s)
fc_co_map = data['fc_co_map'] # (g/s)
fc_hc_map = data['fc_hc_map'] # (g/s)
fc_nox_map = data['fc_nox_map'] # (g/s)
min_fuel_consum_line = []
min_fuel_consum_line_spd = []
min_fuel_consum_line_trq = []

# Calculate most efficient line
for ice_kw in range(0,int(fc_map_spd[-1]*fc_map_trq[-1]/1000),2):
  if ice_kw == 0:
    pass
  else:
    min_fuel_consum_line.append(min_fuel_consum)
  for ice_spd in range(20,int(fc_map_spd[-1]),5):
    ice_trq = ice_kw*1000/ice_spd
    fuel_map = interpolate.interp2d(fc_map_trq,fc_map_spd,fc_fuel_map, kind='cubic')
    ice_fuel_consum = fuel_map(ice_trq, ice_spd)[0]
    if ice_spd == 20:
      min_fuel_consum = [ice_kw,ice_spd,ice_trq,999]
    else:
      if ice_fuel_consum < min_fuel_consum[-1] and ice_trq<fc_map_trq[-1]:
        min_fuel_consum = [ice_kw,ice_spd,ice_trq,ice_fuel_consum]

for i in range(0,len(min_fuel_consum_line)):
  min_fuel_consum_line_spd.append(min_fuel_consum_line[i][1])
  min_fuel_consum_line_trq.append(min_fuel_consum_line[i][2])


# pi = 3.14159
# rads2rpm = float(1/(2*pi/60))
# set_title_fontsize = 10
# set_label_fontsize = 9
# set_tick_fontsize = 8
# X,Y = np.meshgrid(fc_map_spd*rads2rpm, fc_map_trq)
# C_fuel = plt.contour(X, Y, fc_fuel_map.T, 10, alpha=.75, cmap=plt.cm.hot)
# # plt.plot([i * rads2rpm for i in self.W_ice_array],self.T_ice_array,"o",color='C2',markersize=2 )
# plt.plot(fc_map_spd*rads2rpm,fc_max_trq,'--',color='red',linewidth=3)
# plt.plot([i * rads2rpm for i in min_fuel_consum_line_spd],min_fuel_consum_line_trq,'-',color='green',linewidth=3)
# plt.clabel(C_fuel, inline=False, fontsize=set_label_fontsize)
# plt.show()

# X,Y = np.meshgrid(fc_map_spd*rads2rpm, fc_map_trq)
# kW_map = np.multiply(X,Y)/1000
# C_bsfc = plt.contour(X, Y, fc_fuel_map.T/kW_map,30, alpha=.75, cmap=plt.cm.hot)
# plt.plot(fc_map_spd*rads2rpm,fc_max_trq,'--',color='red',linewidth=3)
# plt.plot([i * rads2rpm for i in min_fuel_consum_line_spd],min_fuel_consum_line_trq,'-',color='green',linewidth=3)
# plt.clabel(C_bsfc, inline=False, fontsize=set_label_fontsize)
# plt.show()


################# Select MG1 #################
##mg1_name = 'GC_PRIUS_JPN'
mg1_name = 'MC_PRIUS_JPN'
##mg1_name = 'MC_INSIGHT_draft'
##mg1_name = 'MC_PM49'
if vehicle_name == 'Hybrid_Power_Split_ICE_MG1_MG2':
    mg1_name = 'GC_PRIUS_JPN'
if mg1_name == 'GC_PRIUS_JPN':
    dir_name = 'advisor_2003_data/generator/'
    data = scipy.io.loadmat(dir_name+mg1_name+'.mat')
    mg1_map_spd = data['gc_map_spd'][0] # (rad/s)
    mg1_map_trq = data['gc_map_trq'][0] # (N*m)
    mg1_max_trq = data['gc_max_trq'][0] # (N*m)
    mg1_max_gen_trq = -1*data['gc_max_trq'][0] # (N*m)
    mg1_eff_map = data['gc_eff_map'] # (g/s)
else:
    dir_name = 'advisor_2003_data/motor/'
    data = scipy.io.loadmat(dir_name+mg1_name+'.mat')
    mg1_map_spd = data['mc_map_spd'][0] # (rad/s)
    mg1_map_trq = data['mc_map_trq'][0] # (N*m)
    mg1_max_trq = data['mc_max_trq'][0] # (N*m)
    mg1_max_gen_trq = data['mc_max_gen_trq'][0] # (N*m)
    mg1_eff_map = data['mc_eff_map'] # (g/s)

################# Select MG2 #################
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

################# Create Dictionary #################
folder = os.path.exists('check_point/'+vehicle_name+'/')
if not folder:
    os.makedirs('check_point/'+vehicle_name+'/')
folder = os.path.exists('result_figure/'+vehicle_name+'/')
if not folder:
    os.makedirs('result_figure/'+vehicle_name+'/')
folder = os.path.exists('result_data/'+vehicle_name+'/')
if not folder:
    os.makedirs('result_data/'+vehicle_name+'/')

###################################################
###################################################
###################################################
def run_training():
    step_RL = 0
    cycle_round = 0
    last_time_iteration=RL.return_last_iteration()

        
##    de = 1/cycle_dt
    for step in range(999999999999999):

        k = step % cycle_total_step_number # number of steps in one cycle
        time = k*cycle_dt

        # load cycle target speed
        current_cycle_target_speed = np.interp(time, cycle_time, cycle_speed)
        next_cycle_target_speed = np.interp(time+cycle_dt, cycle_time, cycle_speed)
        next2_cycle_target_speed = np.interp(time+2*cycle_dt, cycle_time, cycle_speed)

        #send_cycle_speed(cycle_target_speed)
        # initial observation
        observation = env.reset(current_cycle_target_speed,next_cycle_target_speed)

        while True:
                        
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
##            observation_, reward, done = env.step(action)
            observation_, reward, last_iteration, latest_reward  = env.vehicle_powertrain(action,time,cycle_end_time,next_cycle_target_speed,next2_cycle_target_speed)

            RL.store_transition(observation, action, reward, observation_)

            if (step_RL > 10) and (step_RL % 5 == 0):
                RL.learn(save_step,step)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            break

        if time == cycle_time[-1]: # clean figure for next cycle 

          if cycle_round== 0:
              max_cycle_reward = 0
              # max_cycle_reward = latest_reward[-1]
              # print('Iteration:', cycle_round+last_time_iteration, 'max_cycle_reward = ',max_cycle_reward)
          else:
            if latest_reward[-1]>max_cycle_reward:
                max_cycle_reward = latest_reward[-1]
                print('Iteration:', cycle_round+last_time_iteration, 'max_cycle_reward = ',max_cycle_reward)
                RL.save_learning(cycle_round,max_cycle_reward,cycle_name)
                env.clear_figure()
                env.cycle_plot(cycle_time,cycle_speed,cycle_name,cycle_round,last_time_iteration,max_cycle_reward)

          if (cycle_round+last_time_iteration+1) % number_show == 1:
              env.clear_figure()
              print('Iteration:', cycle_round+last_time_iteration, 'plot')
              env.cycle_plot(cycle_time,cycle_speed,cycle_name,cycle_round,last_time_iteration,max_cycle_reward)
          cycle_round += 1
          env.cycle_reset()
        step_RL += 1
        
        if cycle_round == end_cycle_round:
            break

        # print(cycle_round)

    # end of game
    print('cycle learning over')
##    env.destroy()


if __name__ == "__main__":
    # maze game
    if vehicle_type == 'car_env_NV_ICE_5AT':
        from car_env_NV_ICE_5AT import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  fc_map_spd,fc_map_trq,fc_fuel_map,fc_max_trq,min_fuel_consum_line_spd,min_fuel_consum_line_trq)
    elif vehicle_type == 'car_env_NV_ICE_CVT':
        from car_env_NV_ICE_CVT import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  fc_map_spd,fc_map_trq,fc_fuel_map,fc_max_trq,min_fuel_consum_line_spd,min_fuel_consum_line_trq)
    elif vehicle_type == 'car_env_EV_MG2':
        from car_env_EV_MG2 import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  mg2_map_spd,mg2_map_trq,mg2_eff_map,mg2_max_trq,mg2_max_gen_trq)
    elif vehicle_type == 'car_env_EV_MG2_2AT':
        from car_env_EV_MG2_2AT import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  mg2_map_spd,mg2_map_trq,mg2_eff_map,mg2_max_trq,mg2_max_gen_trq)
    elif vehicle_type == 'car_env_EV_2MG_MG1_MG2':
        from car_env_EV_2MG_MG1_MG2 import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  mg1_map_spd,mg1_map_trq,mg1_eff_map,mg1_max_trq,mg1_max_gen_trq,
                  mg2_map_spd,mg2_map_trq,mg2_eff_map,mg2_max_trq,mg2_max_gen_trq)
    elif vehicle_type == 'car_env_Hybrid_ICE_5AT_MG2':
        from car_env_Hybrid_ICE_5AT_MG2 import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  fc_map_spd,fc_map_trq,fc_fuel_map,fc_max_trq,min_fuel_consum_line_spd,min_fuel_consum_line_trq,
                  mg2_map_spd,mg2_map_trq,mg2_eff_map,mg2_max_trq,mg2_max_gen_trq)
    elif vehicle_type == 'car_env_Hybrid_ICE_CVT_MG2':
        from car_env_Hybrid_ICE_CVT_MG2 import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  fc_map_spd,fc_map_trq,fc_fuel_map,fc_max_trq,min_fuel_consum_line_spd,min_fuel_consum_line_trq,
                  mg2_map_spd,mg2_map_trq,mg2_eff_map,mg2_max_trq,mg2_max_gen_trq)
    elif vehicle_type == 'car_env_Hybrid_ICE_MG2_5AT':
        from car_env_Hybrid_ICE_MG2_5AT import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  fc_map_spd,fc_map_trq,fc_fuel_map,fc_max_trq,min_fuel_consum_line_spd,min_fuel_consum_line_trq,
                  mg1_map_spd,mg1_map_trq,mg1_eff_map,mg1_max_trq,mg1_max_gen_trq)
    elif vehicle_type == 'car_env_Hybrid_ICE_MG2_CVT':
        from car_env_Hybrid_ICE_MG2_CVT import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  fc_map_spd,fc_map_trq,fc_fuel_map,fc_max_trq,min_fuel_consum_line_spd,min_fuel_consum_line_trq,
                  mg1_map_spd,mg1_map_trq,mg1_eff_map,mg1_max_trq,mg1_max_gen_trq)
    elif vehicle_type == 'car_env_Hybrid_Power_Split_ICE_MG1_MG2':
        from car_env_Hybrid_Power_Split_ICE_MG1_MG2 import Car
        env = Car(time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
                  fc_map_spd,fc_map_trq,fc_fuel_map,fc_max_trq,min_fuel_consum_line_spd,min_fuel_consum_line_trq,
                  mg1_map_spd,mg1_map_trq,mg1_eff_map,mg1_max_trq,mg1_max_gen_trq,
                  mg2_map_spd,mg2_map_trq,mg2_eff_map,mg2_max_trq,mg2_max_gen_trq)

    RL = DeepQNetwork(env.n_actions, env.n_features, vehicle_name,
                      learning_rate=set_learning_rate, # 0.01 ~ 0.0001
                      reward_decay=set_reward_decay,
                      e_greedy=set_e_greedy,
                      replace_target_iter=set_replace_target_iter,
                      memory_size=set_memory_size,
                      # output_graph=True
                      )
    env.after(100, run_training)
    env.mainloop()
    RL.plot_cost()
