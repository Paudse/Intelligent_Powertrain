#The recipe gives simple implementation of a Discrete Proportional-Integral-Derivative (PID) controller. PID controller gives output value for error between desired reference input and measurement feedback to minimize error value.
#More information: http://en.wikipedia.org/wiki/PID_controller
#
#cnr437@gmail.com
#
#######	Example	#########
#
#p=PID(3.0,0.4,1.2)
#p.setPoint(5.0)
#while True:
#     pid = p.update(measurement_value)
#
#


class PID_Controller:
	"""
	Discrete PID control
	"""

	def __init__(self, P=100.0, I=1.0, D=1.0, Derivator=0.0, Integrator=0, Integrator_max=500, Integrator_min=-500):

		self.Kp=P
		self.Ki=I
		self.Kd=D
		self.Derivator=Derivator
		self.Integrator=Integrator
		self.Integrator_max=Integrator_max
		self.Integrator_min=Integrator_min

		self.set_point=0.0
		self.error=0.0

		self.T_ice = 0
		self.T_mg2 = 0
		self.T_brake = 0
		self.i_gear = 0

	def PID_update(self,current_value):
		"""
		Calculate PID output value for given reference input and feedback
		"""

		self.error = self.set_point - current_value

		self.P_value = self.Kp * self.error
		self.D_value = self.Kd * ( self.error - self.Derivator)
		self.Derivator = self.error

		self.Integrator = self.Integrator + self.error

		if self.Integrator > self.Integrator_max:
			self.Integrator = self.Integrator_max
		elif self.Integrator < self.Integrator_min:
			self.Integrator = self.Integrator_min

		self.I_value = self.Integrator * self.Ki

		PID = self.P_value + self.I_value + self.D_value

		return PID

	def setPoint(self,set_point):
		self.set_point = set_point

	def reset(self):
		self.set_point = 0
		self.Integrator = 0
		self.Derivator = 0
		self.T_ice = 0
		self.T_mg2 = 0
		self.T_brake = 0
		self.i_gear = 0


	def getIntegrator(self):
		return self.Integrator

	def getDerivator(self):
		return self.Derivator

	def control_command(self,current_cycle_target_speed,current_v_vehicle_kmh):
		self.setPoint(current_cycle_target_speed)
		self.T_demand = self.PID_update(current_v_vehicle_kmh)
		return self.T_demand
