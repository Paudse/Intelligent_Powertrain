import numpy as np
import scipy.io
from scipy import interpolate
import matplotlib.pyplot as plt
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import scipy.io

pi = 3.14159
rads2rpm = float(1/(2*pi/60))

class Car(tk.Tk, object):
    def __init__(self,time,cycle_time,cycle_speed,cycle_round,cycle_dt,cycle_total_step_number,vehicle_name,
        fc_map_spd,fc_map_trq,fc_fuel_map,fc_max_trq,min_fuel_consum_line_spd,min_fuel_consum_line_trq,
    	mg2_map_spd,mg2_map_trq,mg2_eff_map,mg2_max_trq,mg2_max_gen_trq):
        super(Car, self).__init__()
        self.vehicle_name = vehicle_name
        print('vehicle type: '+self.vehicle_name)
        self.time = time
        self.cycle_time = cycle_time
        self.cycle_speed = cycle_speed
        self.cycle_round = cycle_round
        self.cycle_dt = cycle_dt
        self.cycle_total_step_number = cycle_total_step_number
        self.fc_map_spd = fc_map_spd
        self.fc_map_trq = fc_map_trq
        self.fc_fuel_map = fc_fuel_map
        self.fc_max_trq = fc_max_trq
        self.mg2_map_spd = mg2_map_spd
        self.mg2_map_trq = mg2_map_trq
        self.mg2_eff_map = mg2_eff_map
        self.mg2_max_trq = mg2_max_trq
        self.mg2_max_gen_trq = mg2_max_gen_trq
        self.min_fuel_consum_line_spd = min_fuel_consum_line_spd
        self.min_fuel_consum_line_trq = min_fuel_consum_line_trq
        self.n_actions = 11
        self.n_features = 10
        self.i_gear = 1
        self.i_gear_max = 5
        self.i_gear_min = 1
        self.W_ice = 0
        self.T_ice = 0
        self.kW_ice = 0
        self.W_mg2 = 0
        self.T_mg2 = 0
        self.kW_mg2 = 0
        self.T_brake = 0
        self.fuel_accumulate = 0
        self.total_resis_torque = 0
        self.W_wheel = 0
        self.v_vehicle = 0
        self.v_vehicle_kmh = 0
        self.cycle_iteration = 0
        self.initial_soc = 0.6
        self.soc = self.initial_soc
        self.reward_acum = 0
        self.i_f_gear_ratio = 3.21 
        self.i_gear_ratio_map = np.array([3.46, 1.75, 1.10, 0.86, 0.71]) # 2008 Saab
        self.i_gear_ratio = self.i_gear_ratio_map[self.i_gear-1]

        self.d_T_brake = 50
        self.T_brake_max = 2000

        self.m_vehicle_kg = 1400 # vehicle mass kg
        self.m_equ_vehicle_kg = 1600 # vehicle equivalent mass kg (effective mass of the vehicle)
        self.r_wheel = 0.28 # m

        self.batt_kWh = 1.3 # battery capacity watt hour Toyota Prius III: 1.3 kWh
        self.batt_J = self.batt_kWh*1000*60*60 
        self.soc_max = 0.8
        self.soc_min = 0.2

        self.time_array = []
        self.action_array = []
        self.W_ice_array = []
        self.T_ice_array = []
        self.kW_ice_array = []
        self.W_mg2_array = []
        self.T_mg2_array = []
        self.kW_mg2_array = []
        self.v_vehicle_array = []
        self.v_vehicle_kmh_array = []
        self.T_brake_array = []
        self.i_gear_array = []
        self.fuel_instant_array = []
        self.fuel_accumulate_array = [] 
        self.battery_charge_array = []
        self.battery_soc_array = [] 
        self.reward_array = []
        self.reward_acum_array = []
        self.cycle_iteration_array = []
        self.ave_cyc_reward_array = []
        self.speed_difference_kmh_array = []
        self.cyc_fuel_accumulate_array = []
        self.ave_cyc_speed_difference_kmh_array = []
        
        # self.geometry('1200x700+1950-175')
        self.title('IPS DQN')
        self._build_plot()

    def _build_plot(self):

        # Creat figure
        # self.f = Figure(figsize=(200,10), dpi=100)
        self.f = plt.figure(figsize=(25,6), dpi=100)
        plt.suptitle(self.vehicle_name, fontsize=10)
        self.f.subplots_adjust(hspace=0.8, wspace=0.2)
        self.canvas2 = FigureCanvasTkAgg(self.f, self)
        self.canvas2.show()
        self.canvas2._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Cycle 
        self.cycle = plt.subplot2grid((5, 5), (0, 0))
        # Gear 
        self.gear = plt.subplot2grid((5, 5), (1, 0))
        # Brake 
        self.brake = plt.subplot2grid((5, 5), (2, 0))
        # Action 
        self.action = plt.subplot2grid((5, 5), (3, 0), rowspan=2)  

        # ICE Fuel Consumption Map
        self.fuel = plt.subplot2grid((5, 5), (0, 1)) 
        # BSFC 
        self.bsfc = plt.subplot2grid((5, 5), (0, 1))
        # W_ice figure
        self.w_ice = plt.subplot2grid((5, 5), (1, 1))
        # T_ice figure
        self.t_ice = plt.subplot2grid((5, 5), (2, 1))
        # kW_ice figure
        self.fig_kW_ice = plt.subplot2grid((5, 5), (3, 1))  
        # Accumulated Fuel Consumption
        self.fig_accumulated_fuel = plt.subplot2grid((5, 5), (4, 1))
        # Instance Fuel Consumption
        self.fig_instant_fuel = plt.subplot2grid((5, 5), (4, 1))  
        
        # MG2 Map 
        self.mg2_map = plt.subplot2grid((5, 5), (0, 2))
        # W_mg2 
        self.w_mg2 = plt.subplot2grid((5, 5), (1, 2))
        # T_mg2 
        self.t_mg2 = plt.subplot2grid((5, 5), (2, 2))
        # kW_mg2 
        self.fig_kW_mg2 = plt.subplot2grid((5, 5), (3, 2))
        # Battery Charge
        self.fig_battery_charge = plt.subplot2grid((5, 5), (4, 2))

        # Battery SOC
        self.fig_battery_soc = plt.subplot2grid((5, 5), (4, 2))
        
        # Instant Reward
        self.fig_reward = plt.subplot2grid((5, 5), (0, 3))

        # Learning Curve (Average Reward / Iteration)
        self.fig_learning_curve = plt.subplot2grid((5, 5), (0, 4))
        # Speed Error / Iteration
        self.d_speed = plt.subplot2grid((5, 5), (1, 4), rowspan=1)
        # Fuel Consumption / Iteration
        self.cyc_fuel = plt.subplot2grid((5, 5), (2, 4), rowspan=1)
   
    def cycle_reset(self):
        self.T_ice = 0
        self.W_ice = 0
        self.T_mg2 = 0
        self.W_mg2 = 0
        self.W_wheel = 0
        self.T_brake = 0
        self.total_resis_torque = 0
        self.v_vehicle = 0
        self.v_vehicle_kmh = 0
        self.i_gear = 1
        self.fuel_accumulate = 0
        self.soc = self.initial_soc
        self.reward_acum = 0

        self.time_array = []
        self.action_array = []
        self.W_ice_array = []
        self.T_ice_array = []
        self.kW_ice_array = []
        self.W_mg2_array = []
        self.T_mg2_array = []
        self.kW_mg2_array = []
        self.v_vehicle_array = []
        self.v_vehicle_kmh_array = []
        self.T_brake_array = []
        self.i_gear_array = []
        self.fuel_instant_array = []
        self.fuel_accumulate_array = [] 
        self.battery_charge_array = []
        self.battery_soc_array = [] 
        self.reward_array = []
        self.reward_acum_array = []
        self.speed_difference_kmh_array = []
        

    def reset(self,current_cycle_target_speed,next_cycle_target_speed):
        self.update()
        self.current_cycle_target_speed = current_cycle_target_speed
        self.next_cycle_target_speed = next_cycle_target_speed 

        s = np.array([self.W_ice,self.T_ice,self.W_mg2,self.T_mg2,self.i_gear,self.T_brake,self.soc,
            self.v_vehicle_kmh,current_cycle_target_speed,next_cycle_target_speed])

        # return observation
        return s

    def cycle_plot(self,cycle_time,cycle_speed,cycle_name,cycle_round,last_time_iteration,max_cycle_reward):
        set_title_fontsize = 10
        set_label_fontsize = 9
        set_tick_fontsize = 8

        #################################################################################
        #################################################################################
        ########################### 
        self.cycle.plot(cycle_time,cycle_speed,'b-')      
        self.cycle.plot(self.time_array,self.v_vehicle_kmh_array,"o",color='C1',markersize=2 )

        self.cycle.set_title(cycle_name,fontsize=set_title_fontsize)
        self.cycle.set_xlabel('',fontsize=set_label_fontsize)
        self.cycle.set_ylabel('km/h',fontsize=set_label_fontsize)
        self.cycle.tick_params(labelsize=set_tick_fontsize)
        self.cycle.set(xlim=[0, max(cycle_time)])
        self.cycle.grid(color='gray', linestyle='-', linewidth=0.5)
        
        ###########################
        self.gear.plot([0, max(cycle_time)], [1, 1],'c--')
        self.gear.plot([0, max(cycle_time)], [2, 2],'c--')
        self.gear.plot([0, max(cycle_time)], [3, 3],'c--')
        self.gear.plot([0, max(cycle_time)], [4, 4],'c--')
        self.gear.plot([0, max(cycle_time)], [5, 5],'c--')
        self.gear.plot(self.time_array,self.i_gear_array,"o",color='C1',markersize=2 )

        self.gear.set_title('Gear',fontsize=set_title_fontsize)
        self.gear.set_xlabel('',fontsize=set_label_fontsize)
        self.gear.set_ylabel('',fontsize=set_label_fontsize)
        self.gear.tick_params(labelsize=set_tick_fontsize)
        self.gear.set(xlim=[0, max(cycle_time)],ylim=[0, self.i_gear_max+1])
        self.gear.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.brake.plot(self.time_array,self.T_brake_array,"o",color='C1',markersize=2 )

        self.brake.set_title('Brake Torque',fontsize=set_title_fontsize)
        self.brake.set_xlabel('',fontsize=set_label_fontsize)
        self.brake.set_ylabel('Nm',fontsize=set_label_fontsize)
        self.brake.tick_params(labelsize=set_tick_fontsize)
        self.brake.set(xlim=[0, max(cycle_time)])
        self.brake.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################

        # for i in range(0,self.n_actions):
        #     self.action.plot([0, max(cycle_time)], [i, i],'c-')     
        # self.action.plot(self.time_array,self.action_array,"o",color='C1',markersize=2 )

        # self.action.set_title('Action',fontsize=set_title_fontsize)
        # self.action.set_xlabel('time (s)',fontsize=set_label_fontsize)
        # self.action.set_ylabel('',fontsize=set_label_fontsize)
        # self.action.tick_params(labelsize=set_tick_fontsize)
        # self.action.set(xlim=[0, max(cycle_time)])
        # self.action.grid(color='gray', linestyle='-', linewidth=0.5)

        #################################################################################
        #################################################################################
        ###########################
        X,Y = np.meshgrid(self.fc_map_spd*rads2rpm, self.fc_map_trq)
        C_fuel = self.fuel.contour(X, Y, self.fc_fuel_map.T, 10, alpha=.75, cmap=plt.cm.hot)
        self.fuel.plot([i * rads2rpm for i in self.W_ice_array],self.T_ice_array,"o",color='C2',markersize=2 )
        self.fuel.plot([i * rads2rpm for i in self.min_fuel_consum_line_spd],self.min_fuel_consum_line_trq,'-',color='gold',linewidth=3)
        self.fuel.plot(self.fc_map_spd*rads2rpm,self.fc_max_trq,'--',color='red',linewidth=3)
        self.fuel.clabel(C_fuel, inline=False, fontsize=set_label_fontsize)

        self.fuel.set_title('Fuel Comsumption (g/s)',fontsize=set_title_fontsize)
        self.fuel.set_xlabel('rpm',fontsize=set_label_fontsize)
        self.fuel.set_ylabel('Nm',fontsize=set_label_fontsize)
        self.fuel.tick_params(labelsize=set_tick_fontsize)
        self.fuel.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        X,Y = np.meshgrid(self.fc_map_spd*rads2rpm, self.fc_map_trq)
        kW_map = np.multiply(X,Y)/1000
        C_bsfc = self.bsfc.contour(X, Y, self.fc_fuel_map.T/kW_map,30, alpha=.75, cmap=plt.cm.hot)
        self.bsfc.plot([i * rads2rpm for i in self.W_ice_array],self.T_ice_array,"o",color='C2',markersize=2 )
        self.bsfc.plot([i * rads2rpm for i in self.min_fuel_consum_line_spd],self.min_fuel_consum_line_trq,'-',color='gold',linewidth=3)
        self.bsfc.plot(self.fc_map_spd*rads2rpm,self.fc_max_trq,'--',color='red',linewidth=3)
        self.bsfc.clabel(C_bsfc, inline=False, fontsize=1)

        self.bsfc.set_title('BSFC (g/kWh)',fontsize=set_title_fontsize)
        self.bsfc.set_xlabel('rpm ',fontsize=set_label_fontsize)
        self.bsfc.set_ylabel('Nm',fontsize=set_label_fontsize)
        self.bsfc.tick_params(labelsize=set_tick_fontsize)
        self.bsfc.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.w_ice.plot(self.time_array,[i * rads2rpm for i in self.W_ice_array],"o",color='C1',markersize=2 )

        self.w_ice.set_title('ICE Speed',fontsize=set_title_fontsize)
        self.w_ice.set_xlabel('',fontsize=set_label_fontsize)
        self.w_ice.set_ylabel('rpm',fontsize=set_label_fontsize)
        self.w_ice.tick_params(labelsize=set_tick_fontsize)
        self.w_ice.set(xlim=[0, max(cycle_time)])
        self.w_ice.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.t_ice.grid(color='gray', linestyle='-', linewidth=0.5)
        self.t_ice.set(xlim=[0, max(cycle_time)],  ylabel='ICE Torque (Nm)', xlabel='')
        self.t_ice.plot(self.time_array,self.T_ice_array,"o",color='C1',markersize=2 )

        self.t_ice.set_title('ICE Torque',fontsize=set_title_fontsize)
        self.t_ice.set_xlabel('',fontsize=set_label_fontsize)
        self.t_ice.set_ylabel('Nm',fontsize=set_label_fontsize)
        self.t_ice.tick_params(labelsize=set_tick_fontsize)
        self.t_ice.set(xlim=[0, max(cycle_time)])
        self.t_ice.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.fig_kW_ice.plot(self.time_array,self.kW_ice_array,"o",color='C1',markersize=2 )

        self.fig_kW_ice.set_title('ICE Power',fontsize=set_title_fontsize)
        self.fig_kW_ice.set_xlabel('',fontsize=set_label_fontsize)
        self.fig_kW_ice.set_ylabel('kW',fontsize=set_label_fontsize)
        self.fig_kW_ice.tick_params(labelsize=set_tick_fontsize)
        self.fig_kW_ice.set(xlim=[0, max(cycle_time)])
        self.fig_kW_ice.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.fig_instant_fuel.plot(self.time_array,self.fuel_instant_array,"o",color='C1',markersize=2 )

        self.fig_instant_fuel.set_title('Instant Fuel Consumption',fontsize=set_title_fontsize)
        self.fig_instant_fuel.set_xlabel('time (s)',fontsize=set_label_fontsize)
        self.fig_instant_fuel.set_ylabel('g/s',fontsize=set_label_fontsize)
        self.fig_instant_fuel.tick_params(labelsize=set_tick_fontsize)
        self.fig_instant_fuel.set(xlim=[0, max(cycle_time)])
        self.fig_instant_fuel.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.fig_accumulated_fuel.plot(self.time_array,self.fuel_accumulate_array,"o",color='C1',markersize=2 )

        self.fig_accumulated_fuel.set_title('Accumulated Fuel Consumption',fontsize=set_title_fontsize)
        self.fig_accumulated_fuel.set_xlabel('time (s)',fontsize=set_label_fontsize)
        self.fig_accumulated_fuel.set_ylabel('g',fontsize=set_label_fontsize)
        self.fig_accumulated_fuel.tick_params(labelsize=set_tick_fontsize)
        self.fig_accumulated_fuel.set(xlim=[0, max(cycle_time)])
        self.fig_accumulated_fuel.grid(color='gray', linestyle='-', linewidth=0.5)

        #################################################################################
        #################################################################################
        ###########################
        X,Y = np.meshgrid(self.mg2_map_spd*rads2rpm, self.mg2_map_trq)
        C_eff = self.mg2_map.contour(X, Y, self.mg2_eff_map.T, 10, alpha=.75, cmap=plt.cm.hot)
        for ii in range(len(self.T_mg2_array)):
            if self.T_mg2_array[ii] < 0:
                self.mg2_map.plot(self.W_mg2_array[ii]*rads2rpm,self.T_mg2_array[ii],"o",color='purple',markersize=2 )
            else:
                self.mg2_map.plot(self.W_mg2_array[ii]*rads2rpm,self.T_mg2_array[ii],"o",color='C2',markersize=2 )
        self.mg2_map.plot(self.mg2_map_spd*rads2rpm, self.mg2_max_trq,'--',color='red',linewidth=3)
        self.mg2_map.plot(self.mg2_map_spd*rads2rpm, self.mg2_max_gen_trq,'--',color='red',linewidth=3)
        self.mg2_map.clabel(C_eff, inline=False, fontsize=set_tick_fontsize) 

        self.mg2_map.set_title('MG2 Efficiency',fontsize=set_title_fontsize)
        self.mg2_map.set_xlabel('rpm',fontsize=set_label_fontsize)
        self.mg2_map.set_ylabel('Nm',fontsize=set_label_fontsize)
        self.mg2_map.tick_params(labelsize=set_tick_fontsize)
        self.mg2_map.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.w_mg2.plot(self.time_array,[i * rads2rpm for i in self.W_mg2_array],"o",color='C1',markersize=2 )

        self.w_mg2.set_title('MG2 Speed',fontsize=set_title_fontsize)
        self.w_mg2.set_xlabel('',fontsize=set_label_fontsize)
        self.w_mg2.set_ylabel('rpm',fontsize=set_label_fontsize)
        self.w_mg2.tick_params(labelsize=set_tick_fontsize)
        self.w_mg2.set(xlim=[0, max(cycle_time)])
        self.w_mg2.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.t_mg2.plot(self.time_array,np.minimum(self.T_mg2_array, 0),"o",color='purple',markersize=2 )
        self.t_mg2.plot(self.time_array,np.maximum(self.T_mg2_array, 0),"o",color='C1',markersize=2 )
        self.t_mg2.plot([0, max(cycle_time)], [0, 0],'black',linewidth=2)

        self.t_mg2.set_title('MG2 Torque',fontsize=set_title_fontsize)
        self.t_mg2.set_xlabel('',fontsize=set_label_fontsize)
        self.t_mg2.set_ylabel('Nm',fontsize=set_label_fontsize)
        self.t_mg2.tick_params(labelsize=set_tick_fontsize)
        self.t_mg2.set(xlim=[0, max(cycle_time)])
        self.t_mg2.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.fig_kW_mg2.plot(self.time_array,np.minimum(self.kW_mg2_array, 0),"o",color='purple',markersize=2 )
        self.fig_kW_mg2.plot(self.time_array,np.maximum(self.kW_mg2_array, 0),"o",color='C1',markersize=2 )
        self.fig_kW_mg2.plot([0, max(cycle_time)], [0, 0],'black',linewidth=2)

        self.fig_kW_mg2.set_title('MG2 Power',fontsize=set_title_fontsize)
        self.fig_kW_mg2.set_xlabel('',fontsize=set_label_fontsize)
        self.fig_kW_mg2.set_ylabel('kW',fontsize=set_label_fontsize)
        self.fig_kW_mg2.tick_params(labelsize=set_tick_fontsize)
        self.fig_kW_mg2.set(xlim=[0, max(cycle_time)])
        self.fig_kW_mg2.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.fig_battery_charge.plot(self.time_array,np.minimum(self.battery_charge_array, 0),"o",color='purple',markersize=2 )
        self.fig_battery_charge.plot(self.time_array,np.maximum(self.battery_charge_array, 0),"o",color='C1',markersize=2 )

        self.fig_battery_charge.set_title('Battery Charge',fontsize=set_title_fontsize)
        self.fig_battery_charge.set_xlabel('',fontsize=set_label_fontsize)
        self.fig_battery_charge.set_ylabel('kW',fontsize=set_label_fontsize)
        self.fig_battery_charge.tick_params(labelsize=set_tick_fontsize)
        self.fig_battery_charge.set(xlim=[0, max(cycle_time)])
        self.fig_battery_charge.grid(color='gray', linestyle='-', linewidth=0.5)

        ############################
        self.fig_battery_soc.plot(self.time_array,self.battery_soc_array,"o",color='C1',markersize=2 )

        self.fig_battery_soc.set_title('Battery SOC',fontsize=set_title_fontsize)
        self.fig_battery_soc.set_xlabel('time (s)',fontsize=set_label_fontsize)
        self.fig_battery_soc.set_ylabel('',fontsize=set_label_fontsize)
        self.fig_battery_soc.tick_params(labelsize=set_tick_fontsize)
        self.fig_battery_soc.set(xlim=[0, max(cycle_time)],ylim=[self.soc_min, self.soc_max])
        self.fig_battery_soc.grid(color='gray', linestyle='-', linewidth=0.5)

        #################################################################################
        #################################################################################
        ###########################
        self.fig_reward.plot(self.time_array,self.reward_array,"o",color='C1',markersize=2 )    
        
        self.fig_reward.set_title('Instant Reward',fontsize=set_title_fontsize)
        self.fig_reward.set_xlabel('time (s)',fontsize=set_label_fontsize)
        self.fig_reward.set_ylabel('',fontsize=set_label_fontsize)
        self.fig_reward.tick_params(labelsize=set_tick_fontsize)
        self.fig_reward.set(xlim=[0, max(cycle_time)],ylim=[0, 100])
        self.fig_reward.grid(color='gray', linestyle='-', linewidth=0.5)

        
        #################################################################################
        #################################################################################
        ###########################
        self.fig_learning_curve.plot([i+last_time_iteration-1 for i in self.cycle_iteration_array],self.ave_cyc_reward_array,"-",color='C2',markersize=2 )

        self.fig_learning_curve.set_title('Learning Curve',fontsize=set_title_fontsize)
        self.fig_learning_curve.set_xlabel('',fontsize=set_label_fontsize)
        self.fig_learning_curve.set_ylabel('Average Reward',fontsize=set_label_fontsize)
        self.fig_learning_curve.tick_params(labelsize=set_tick_fontsize)
        self.fig_learning_curve.set(xlim=[last_time_iteration, max(self.cycle_iteration_array)+last_time_iteration-1],ylim=[0, 100])
        self.fig_learning_curve.grid(color='gray', linestyle='-', linewidth=0.5)

        # print('average cycle reward:', self.reward_acum_array[-1]/self.cycle_total_step_number)

        ###########################
        self.d_speed.plot([i+last_time_iteration-1 for i in self.cycle_iteration_array],self.ave_cyc_speed_difference_kmh_array,"-",color='C2',markersize=2 )

        self.d_speed.set_title('Average Speed Error',fontsize=set_title_fontsize)
        self.d_speed.set_xlabel('',fontsize=set_label_fontsize)
        self.d_speed.set_ylabel('km/h',fontsize=set_label_fontsize)
        self.d_speed.tick_params(labelsize=set_tick_fontsize)
        self.d_speed.set(xlim=[last_time_iteration, max(self.cycle_iteration_array)+last_time_iteration-1])
        self.d_speed.grid(color='gray', linestyle='-', linewidth=0.5)

        ###########################
        self.cyc_fuel.plot([i+last_time_iteration-1 for i in self.cycle_iteration_array],self.cyc_fuel_accumulate_array,"-",color='C2',markersize=2 )

        self.cyc_fuel.set_title('Average Fuel Consumption',fontsize=set_title_fontsize)
        self.cyc_fuel.set_xlabel('iteration',fontsize=set_label_fontsize)
        self.cyc_fuel.set_ylabel('g/s',fontsize=set_label_fontsize)
        self.cyc_fuel.tick_params(labelsize=set_tick_fontsize)
        self.cyc_fuel.set(xlim=[last_time_iteration, max(self.cycle_iteration_array)+last_time_iteration-1])
        self.cyc_fuel.grid(color='gray', linestyle='-', linewidth=0.5)

        #################################################################################
        #################################################################################
        #################################################################################
        self.canvas2.draw()
        self.f.savefig('result_figure/'+self.vehicle_name+'/iteration_'+str(last_time_iteration+cycle_round)+'_'+str(cycle_name)+'_reward-'+str(round(max_cycle_reward,2))+'.png', format="png", dpi=100)
        scipy.io.savemat('result_data/'+self.vehicle_name+'/iteration_'+str(last_time_iteration+cycle_round)+'_'+str(cycle_name)+'_reward-'+str(round(max_cycle_reward,2))+'.mat', {'sim_t_dqn':self.time_array,
            'sim_action_dqn':self.action_array,
            'sim_W_ice_dqn':self.W_ice_array,
            'sim_T_ice_dqn':self.T_ice_array,
            'sim_P_ice_dqn':self.kW_ice_array,
            'sim_W_mg2_dqn':self.W_mg2_array,
            'sim_T_mg2_dqn':self.T_mg2_array,
            'sim_P_mg2_dqn':self.kW_mg2_array,
            'sim_car_kph_dqn':self.v_vehicle_kmh_array,
            'sim_T_brake_dqn':self.T_brake_array,
            'sim_gear_dqn':self.i_gear_array,
            'sim_fuel_accumulate_dqn':self.fuel_accumulate_array,
            'sim_fuel_instant_dqn':self.fuel_instant_array,
            'sim_battery_charge_dqn':self.battery_charge_array,
            'sim_battery_soc_dqn':self.battery_soc_array,
            'sim_reward_dqn':self.reward_array,
            'sim_acum_reward_dqn':self.reward_acum_array,
            'cycle_iteration_dqn':self.cycle_iteration_array,
            'ave_cyc_reward_dqn':self.ave_cyc_reward_array,
            'cyc_fuel_accumulate_dqn':self.cyc_fuel_accumulate_array,
            'ave_cyc_speed_difference_kmh_dqn':self.ave_cyc_speed_difference_kmh_array,
            'speed_difference_kmh_dqn':self.speed_difference_kmh_array
            })

    def clear_figure(self):
        self.cycle.cla()
        self.gear.cla()
        self.brake.cla()
        self.action.cla()
        self.bsfc.cla()
        self.fuel.cla()
        self.w_ice.cla()
        self.t_ice.cla()
        self.fig_kW_ice.cla()
        self.fig_instant_fuel.cla()
        self.fig_accumulated_fuel.cla()
        self.mg2_map.cla()
        self.w_mg2.cla()
        self.t_mg2.cla()
        self.fig_kW_mg2.cla()
        self.fig_battery_charge.cla()
        self.fig_battery_soc.cla()
        self.fig_reward.cla()
        self.d_speed.cla()

    def vehicle_powertrain(self,T_demand,time,cycle_end_time,next_cycle_target_speed,next2_cycle_target_speed):
        self.T_ice_max = np.interp(self.W_ice, self.fc_map_spd, self.fc_max_trq)
        self.T_ice_min = 0
        self.W_ice_idle_spd_rs = 73.4  # % idle speed is 701 rpm = 73.4 rad/s
        self.I_ice = 0.18

        self.T_mg2_max = np.interp(self.W_mg2, self.mg2_map_spd, self.mg2_max_trq)
        self.T_mg2_min = np.interp(self.W_mg2, self.mg2_map_spd, self.mg2_max_gen_trq)
        self.I_mg2 = 0.2

        gravity = 9.81
        rho_air = 1.2 # Air density (kg/m^3)
        a_front_m2 = 2.3 # frontal area (m^2)
        c_d = 0.3 # Aerodynamic drag coefficient
        c_r = 0.009 # Rolling resistance coefficient

        self.T_demand = T_demand


        # Control Rule
        # Gear shifting rule
        if self.W_ice > max(self.fc_map_spd)*1/3: # shift up one gear 
            if self.i_gear+1 > self.i_gear_max:
                self.i_gear = self.i_gear_max
            else: 
                self.i_gear = self.i_gear+1
        if self.W_ice < self.W_ice_idle_spd_rs: 
            if self.i_gear-1 < self.i_gear_min:
                self.i_gear = self.i_gear_min
            else: 
                self.i_gear = self.i_gear-1

        if self.T_demand>0:
            self.T_mg2 = self.T_demand/self.i_f_gear_ratio
            if self.T_mg2 > self.T_mg2_max:
                self.T_mg2 = self.T_mg2_max
            if self.soc<0.4:
                self.T_mg2 = self.T_mg2/2
            if self.soc<0.3:
                self.T_mg2 = self.T_mg2/3
            if self.soc<self.soc_min:
                self.T_mg2 = 0
            self.T_ice = (self.T_demand-self.T_mg2*self.i_f_gear_ratio)/self.i_gear_ratio/self.i_f_gear_ratio
            self.T_brake = 0

        if self.T_demand<0:
            self.T_mg2 = self.T_demand/self.i_f_gear_ratio
            if abs(self.T_mg2) > self.T_mg2_max:
                self.T_mg2 = -self.T_mg2_max
                self.T_brake = -self.T_demand+self.T_mg2_max*self.i_f_gear_ratio
            else: 
                self.T_brake = 0
        
        ### Constraints & Punishment ###
        punishment = 0

        if self.T_ice > self.T_ice_max:
            self.T_ice = self.T_ice_max
            punishment = punishment + 10
        if self.T_ice < self.T_ice_min:
            self.T_ice = 0
            punishment = punishment + 10            
        if self.W_ice > max(self.fc_map_spd):
            self.T_ice = 0
            punishment = punishment + 100

        if self.T_mg2 > self.T_mg2_max:
            self.T_mg2 = self.T_mg2_max
            punishment = punishment + 10
        if self.T_mg2 < self.T_mg2_min:
            self.T_mg2 = 0
            punishment = punishment + 10            
        if self.W_mg2 > max(self.mg2_map_spd):
            self.T_mg2 = 0
            punishment = punishment + 10
        if self.W_mg2 < 0:
            self.T_mg2 = 0
            punishment = punishment + 10
        if self.soc < self.soc_min and self.T_mg2 > 0:
            self.T_mg2 = 0
            punishment = punishment + 10
        if self.soc > self.soc_max and self.T_mg2 < 0:
            self.T_mg2 = 0
            punishment = punishment + 10

        if self.T_brake > self.T_brake_max:
            self.T_brake = self.T_brake_max
            punishment = punishment + 10
        if self.T_brake < 0:
            self.T_brake = 0
            punishment = punishment + 10   

        if self.soc < self.soc_min:
            punishment = punishment + 100
        if self.soc > self.soc_max:
            punishment = punishment + 100

        # look-up transmission gear ratio
        self.i_gear_ratio = self.i_gear_ratio_map[self.i_gear-1]

        # calculate resistance
        if self.v_vehicle < 0:
            self.total_resis_torque = 0
        else:
            self.resis_roll = self.m_vehicle_kg*gravity*c_r
            self.resis_aero = 1/2*c_d*rho_air*a_front_m2*self.v_vehicle**2
            self.total_resis = self.resis_roll + self.resis_aero 
            self.total_resis_torque = self.total_resis*self.r_wheel

        ### Powertrain Dynamics ###
        self.dW_wheel = \
        (self.T_ice*self.i_gear_ratio*self.i_f_gear_ratio \
        +self.T_mg2*self.i_f_gear_ratio \
        -self.T_brake \
        -self.total_resis_torque) \
        /((self.r_wheel**2*self.m_equ_vehicle_kg) \
        +self.I_mg2*(self.i_f_gear_ratio**2) \
        +self.I_ice*(self.i_gear_ratio**2)*(self.i_f_gear_ratio**2))

        # calculate speed
        self.W_wheel = self.W_wheel + self.dW_wheel # rad/s
        self.W_mg2 = self.W_wheel*self.i_f_gear_ratio
        self.W_ice = self.W_wheel*self.i_gear_ratio*self.i_f_gear_ratio
        if self.W_ice < self.W_ice_idle_spd_rs and self.W_ice > 0:
            self.W_ice = self.W_ice_idle_spd_rs
        if self.W_ice < self.W_ice_idle_spd_rs and self.T_ice > 0:
            self.W_ice = self.W_ice_idle_spd_rs
            
        self.v_vehicle = self.W_wheel*self.r_wheel # m/s
        self.v_vehicle_kmh = self.v_vehicle*60*60/1000 #km/h   

        # speed constraints to avoid negative speed caused by numerical issue
        if self.W_wheel<0 or self.W_mg2<0 or self.W_ice<0 or self.v_vehicle<0 or self.v_vehicle_kmh<0: 
            self.W_wheel = 0
            self.W_mg2 = 0
            self.W_ice = 0
            self.v_vehicle = 0
            self.v_vehicle_kmh = 0
            punishment = punishment + 1000

        self.kW_ice = self.W_ice * self.T_ice/1000
        self.kW_mg2 = self.W_mg2 * self.T_mg2/1000

        # calculate ice fuel consumption
        fuel_map = interpolate.interp2d(self.fc_map_trq,self.fc_map_spd,self.fc_fuel_map, kind='cubic')
        self.ICE_fuel_consum = fuel_map(self.T_ice, self.W_ice)[0]
        if self.W_ice < self.W_ice_idle_spd_rs:
            self.ICE_fuel_consum = 0
        self.fuel_instant = self.ICE_fuel_consum*self.cycle_dt
        self.fuel_accumulate = self.fuel_accumulate + self.fuel_instant

        # calculate mg2 efficiency
        mg2_map = interpolate.interp2d(self.mg2_map_trq,self.mg2_map_spd,self.mg2_eff_map,  kind='cubic')
        self.mg2_eff = mg2_map(self.T_mg2, self.W_mg2)[0]

        # calculate battery soc
        self.W_batt_c = self.kW_mg2/self.mg2_eff*1000 # +: discharge; -: charge 
        self.J_batt_c = self.W_batt_c * self.cycle_dt # J
        self.dsoc = -self.J_batt_c/self.batt_J
        self.soc = (self.soc + self.dsoc)

        reward = (1/1)*(70*0.95**(abs(self.current_cycle_target_speed-self.v_vehicle_kmh))  \
        +10*0.5**(self.ICE_fuel_consum) \
        +10*0.95**(100*abs(self.soc-0.6)) \
        +10*0.97**(punishment))

        if time == 0:
            self.reward_acum = 0
        else:
            self.reward_acum = self.reward_acum + reward

        self.time_array.append(time)
        # self.action_array.append(action)
        self.W_ice_array.append(self.W_ice)
        self.T_ice_array.append(self.T_ice)
        self.kW_ice_array.append(self.kW_ice)
        self.W_mg2_array.append(self.W_mg2)
        self.T_mg2_array.append(self.T_mg2)
        self.kW_mg2_array.append(self.kW_mg2)
        self.v_vehicle_array.append(self.v_vehicle)
        self.v_vehicle_kmh_array.append(self.v_vehicle_kmh)
        self.T_brake_array.append(self.T_brake)
        self.i_gear_array.append(self.i_gear)
        self.fuel_instant_array.append(self.fuel_instant)
        self.fuel_accumulate_array.append(self.fuel_accumulate)
        self.battery_charge_array.append(self.W_batt_c)
        self.battery_soc_array.append(self.soc)
        self.reward_array.append(reward)
        self.reward_acum_array.append(self.reward_acum)
        self.speed_difference_kmh_array.append(abs(self.current_cycle_target_speed-self.v_vehicle_kmh))
        if time == cycle_end_time:
            self.cycle_iteration = self.cycle_iteration + 1
            self.cycle_iteration_array.append(self.cycle_iteration)
            self.ave_cyc_reward_array.append(self.reward_acum_array[-1]/self.cycle_total_step_number) # average cycle reward
            self.cyc_fuel_accumulate_array.append(self.fuel_accumulate_array[-1])
            self.ave_cyc_speed_difference_kmh_array.append(sum(self.speed_difference_kmh_array)/self.cycle_total_step_number)
            self.speed_difference_kmh_array = []

        s_ = np.array([self.W_ice,self.T_ice,self.W_mg2,self.T_mg2,self.i_gear,self.T_brake,self.soc,
            self.v_vehicle_kmh,next_cycle_target_speed,next2_cycle_target_speed])

        return s_, reward, self.cycle_iteration_array, self.ave_cyc_reward_array

    def render(self):
        self.update()

