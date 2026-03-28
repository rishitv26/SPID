import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm
from math import *

# ==========================================
# Abstract Base Class For Controllers
# ==========================================
class Controller(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def compute(self, error, dt) -> float:
        pass

# ==========================================
# Standard PID Controller
# ==========================================
class PID(Controller):
    def __init__(self, kp, ki, kd, output_limits=(-100, 100)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limits = output_limits
        self.reset()

    def reset(self):
        self.integral = 0
        self.last_error = 0

    def compute(self, error, dt):
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Anti-windup: reset integral if saturated
        if output == self.output_limits[0] or output == self.output_limits[1]:
            self.integral -= error * dt
        
        self.last_error = error
        return output

# ==========================================
# SmartPID Controller
# ==========================================
class SmartPID(Controller):
    def __init__(self, correction_constant=1.0, windup=0.2,
                 learning_constant=0.001, max_value=127.0):
        """
        SmartPID - Self-tuning PID controller
        
        Args:
            actual_output_func: Function that returns the actual control output applied to system
            correction_constant: Affects shape of fitting curve (default: 1.0)
            windup: Integral max/min value (default: 0.2)
            learning_constant: Affects learning rate (default: 0.001)
            max_value: Maximum value control_output can be (default: 127.0)
        """
        # PID gains
        self.kP = 1.0
        self.kI = 1.0
        self.kD = 1.0
        
        # PID components
        self.P = 0.0
        self.I = 0.0
        self.D = 0.0
        self.prev_val = 0.0
        
        # SPID parameters
        self.correction_constant = abs(correction_constant)
        self.windup = abs(windup)
        self.learning_constant = abs(learning_constant)
        self.CKp = 0.0
        self.CKi = 0.0
        self.CKd = 0.0
        self.largest_measured_error = -np.inf
        self.max_value = abs(max_value)
        
        # Actual output function
        self.control_output = 0.0
        
        # History tracking
        self.history = {
            'P': [],
            'I': [],
            'D': [],
            'expected': [],
            'constant': [],
            'kP': [],
            'kI': [],
            'kD': []
        }

    def reset(self):
        """Reset PID components but not gains"""
        self.P = 0.0
        self.I = 0.0
        self.D = 0.0
        self.prev_val = 0.0
        # Don't reset history - we want to keep it for plotting

    def compute(self, error, dt):
        """
        Calculate the new PID value given the error
        
        Args:
            error: The error from target
            
        Returns:
            Control output value
        """
        # Normalize error from -1 to 1:
        if error > self.largest_measured_error:
            self.largest_measured_error = error
        
        error /= self.largest_measured_error
        
        # SPID Magic
        self._update_components(error, dt)
        self._update_constants(error)
        
        # Track values for plotting
        self.history['P'].append(self.P)
        self.history['I'].append(self.I)
        self.history['D'].append(self.D)
        self.history['expected'].append(self._get_expected(error))
        self.history['kP'].append(self.kP)
        self.history['kI'].append(self.kI)
        self.history['kD'].append(self.kD)
        
        # Return output
        return self._get_control_output() * self.max_value

    def _update_components(self, e, dt):
        """Update the proportion, integral, and derivative components"""
        # Proportional
        self.P = e
        
        self.D = (e - self.prev_val) / dt
        self.prev_val = e
        
        # Integral with windup protection
        if abs(e) < self.windup:
            self.I += e * dt
        
        if abs(self.I) >= self.windup:
            self.I = 0.0

    def _update_constants(self, e):
        """Update the PID constants based on calculated components"""
        
        gamma = self.learning_constant * (self._get_control_output() - self._get_expected(e))
        
        self.CKp = gamma * self.P
        self.CKi = gamma * self.I
        self.CKd = gamma * self.D
        
        self.kP -= self.CKp
        self.kI -= self.CKi
        self.kD -= self.CKd
        
        # safety net for kP, kI, kD
        if self.kP < 0 or isinf(self.kP) or isnan(self.kP):
            self.kP = 0
        if self.kI < 0 or isinf(self.kI) or isnan(self.kI):
            self.kI = 0
        if self.kD < 0 or isinf(self.kD) or isnan(self.kD):
            self.kD = 0
        
        self.history['constant'].append(self.learning_constant)

    def _get_expected(self, e):
        """Get the expected value for the given error"""
        # Using tanh with cubic error
        # return np.tanh(3.3 * e * e * e) # causes overfit-underfit oscillation
        # Using exp with cubic error
        if e >= 0:
            return np.exp(0.7 * e * e * e) - 1.0
        else:
            return -np.exp(-0.7 * e * e * e) + 1.0
        

    def _get_control_output(self):
        return self.kP * self.P + self.kI * self.I + self.kD * self.D