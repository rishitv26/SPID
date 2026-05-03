import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm
from math import isnan, isinf

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
        self.alpha = 1.0
        self.max_length = 1.0
        
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
        
        self.normalized_error = error / self.largest_measured_error
        self.alpha = 1 - abs(self.normalized_error)
        
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
        return self._get_control_output() #* self.max_value

    def _update_components(self, e, dt):
        """Update the proportion, integral, and derivative components"""
        # Proportional
        self.P = e
        
        if abs(e) < 1:
            self.D = (e - self.prev_val) / dt
            self.prev_val = e
        
        # Integral with windup protection
        if abs(e) < self.windup:
            self.I += e * dt
        
        if abs(self.I) >= self.windup:
            self.I = 0.0

    def _update_constants(self, e):
        """Update the PID constants based on calculated components"""
        # calculate learn rate:
        learn_rate = self.learning_constant * self.alpha
        
        # put clips on difference:
        difference = self._get_control_output() - self._get_expected(e)
        if abs(difference) > self.max_length: difference = np.sign(difference) * self.max_length

        gamma = learn_rate * difference
            
        self.CKp = gamma * self.P
        self.CKi = gamma * self.I
        self.CKd = gamma * self.D
        
        # clip to prevent overflow (Bottou et al.)
        clip = 1.0
        self.CKp = np.clip(self.CKp, -clip, clip)
        self.CKi = np.clip(self.CKi, -clip, clip)
        self.CKd = np.clip(self.CKd, -clip, clip)
            
        self.kP -= self.CKp
        self.kI -= self.CKi
        self.kD -= self.CKd
            
        # safety net for kP, kI, kD
        self.kP = max(0.01, self.kP)
        self.kI = max(0.01, self.kI)
        self.kD = max(0.01, self.kD)
        
        self.history['constant'].append(learn_rate)

    def _get_expected(self, e):
        """Get the expected value for the given error"""
        # Using tanh with cubic error
        return self.alpha * self.max_value / 2 * np.tanh(0.01 * e * e * e) 
        

    def _get_control_output(self):
        return self.kP * self.P + self.kI * self.I + self.kD * self.D