import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm

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
    def __init__(self, actual_output_func, correction_constant=1.0, windup=10.0,
                 learning_constant=0.001, max_value=127.0):
        """
        SmartPID - Self-tuning PID controller
        
        Args:
            actual_output_func: Function that returns the actual control output applied to system
            correction_constant: Affects shape of fitting curve (default: 1.0)
            windup: Integral max/min value (default: 10.0)
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
        self.alpha = 0.0
        
        # SPID parameters
        self.correction_constant = abs(correction_constant)
        self.windup = abs(windup)
        self.learning_constant = abs(learning_constant)
        self.max_value = abs(max_value)
        self.maximum_error = -np.inf
        self.minimum_error = np.inf
        self.CKp = 0.0
        self.CKi = 0.0
        self.CKd = 0.0
        self.batch = {}
        self.N = 5
        self.skip_train = False
        
        # Actual output function
        self.actual_output_func = actual_output_func
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
            dt: Time step (not used in C++ version)
            
        Returns:
            Control output value
        """
        self._update_components(error)
        self._update_constants(error)
        
        growth = 0.001
        self.alpha += growth if self.alpha <= 1.0 else 0.0
        if abs(error - self.prev_val) < 0.1:
            self.alpha += growth + 2*error
        
        # Track values for plotting
        self.history['P'].append(self.P)
        self.history['I'].append(self.I)
        self.history['D'].append(self.D)
        self.history['expected'].append(self._get_expected(error))
        self.history['kP'].append(self.kP)
        self.history['kI'].append(self.kI)
        self.history['kD'].append(self.kD)
        
        return (self.kP * self.P + self.kI * self.I + self.kD * self.D)

    def _update_components(self, e):
        """Update the proportion, integral, and derivative components"""
        if e > self.maximum_error:
            self.maximum_error = e
            self.skip_train = True
        if abs(e) < self.minimum_error:
            self.minimum_error = e
        
        # Proportional
        self.P = e
        
        # Derivative (no dt scaling in C++ version)
        self.D = e - self.prev_val
        self.prev_val = e
        
        # Integral with windup protection
        if abs(e) < self.windup:
            self.I += e
        
        if abs(self.I) >= self.windup:
            self.I = 0.0

    def _update_constants(self, e):
        """Update the PID constants based on calculated components"""
        # Track the constant value (average from batch)
        avg_constant = 0.0
        
        if not self.skip_train:
            Y = self._get_expected(e)
            self.batch[Y] = [self.actual_output_func(), self.P, self.I, self.D]
            
            if len(self.batch) >= self.N:
                # Save past gradients:
                CKp_old = self.CKp / self.N
                CKi_old = self.CKi / self.N
                CKd_old = self.CKd / self.N
                
                # Calculate gradients
                self.CKp, self.CKi, self.CKd = 0.0, 0.0, 0.0
                
                for Ybatch, data in self.batch.items():
                    constant = self.learning_constant * 3 * (data[0] - Ybatch)**2 * np.sign(data[0] - Ybatch)
                    avg_constant += abs(constant)  # Track average absolute constant
                    self.CKp += constant * data[1]
                    self.CKi += constant * data[2]
                    self.CKd += constant * data[3]
                
                avg_constant /= len(self.batch)
                    
                # Update constants as per gradient
                weight = 0.5
                self.kP -= (((1-weight)*(self.CKp / self.N) + (weight)*(CKp_old))) * self._learn_factor(self.minimum_error)
                self.kI -= (((1-weight)*(self.CKi / self.N) + (weight)*(CKi_old))) * self._learn_factor(self.minimum_error)
                self.kD -= (((1-weight)*(self.CKd / self.N) + (weight)*(CKd_old))) * self._learn_factor(self.minimum_error)
                
                self.batch.clear()
        else:
            self.skip_train = False
        
        self.history['constant'].append(avg_constant)

    def _get_expected(self, e):
        """Get the expected value for the given error"""
        # Using tanh with cubic error
        return self.alpha * self.max_value * np.tanh(self.correction_constant * e * e * e)
        # return e

    def _learn_factor(self, x):
        """Special sigmoid function for learning"""
        # Make it twice as strong to stop learning near sub-0 error values
        # return 1.0 / (1.0 + np.exp(-2 * self.correction_constant * x)) - 0.5
        # return 1.0 / (1.0+self.correction_constant*x**2)
        return norm.pdf(x, loc=2, scale=300.0)
    #i messed up your code heheehehehe
    