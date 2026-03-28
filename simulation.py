import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from spid import *
import sys

# Open a file for logging
log_file = open("pid_simulation_log.txt", "w")

# Redirect all prints to the file
sys.stdout = log_file

# ==========================================
# Physical Plant with Friction and Momentum
# ==========================================
class PhysicalPlant:
    def __init__(self, mass=1.0, friction_static=0.5, friction_kinetic=0.3, 
                 friction_viscous=0.1, disturbance_magnitude=0.0):
        """
        Simulates a physical system with:
        - mass: inertia of the system
        - friction_static: static friction coefficient (must overcome to start moving)
        - friction_kinetic: kinetic friction coefficient (opposes motion)
        - friction_viscous: viscous friction (proportional to velocity)
        - disturbance_magnitude: external disturbances
        """
        self.mass = mass
        self.friction_static = friction_static
        self.friction_kinetic = friction_kinetic
        self.friction_viscous = friction_viscous
        self.disturbance_magnitude = disturbance_magnitude
        self.reset()

    def reset(self):
        self.position = 0.0      # Current position
        self.velocity = 0.0      # Current velocity
        self.acceleration = 0.0  # Current acceleration
        self.applied_force = 0.0 # Last applied control force

    def update(self, control_force, dt):
        """Update plant state with physics simulation"""
        
        # Store the commanded control force BEFORE any modifications
        self.applied_force = control_force
        
        # Add random disturbance
        disturbance = np.random.uniform(-self.disturbance_magnitude, 
                                       self.disturbance_magnitude)
        
        # Calculate friction force
        if abs(self.velocity) < 1e-6:
            # Static friction - object at rest
            if abs(control_force) <= self.friction_static:
                friction_force = -control_force  # Exactly balances applied force
            else:
                # Overcome static friction
                friction_force = -np.sign(control_force) * self.friction_static
        else:
            # Kinetic + viscous friction - object moving
            friction_force = (-np.sign(self.velocity) * self.friction_kinetic - 
                            self.friction_viscous * self.velocity)
        
        # Calculate net force
        net_force = control_force + disturbance + friction_force
        
        # F = ma -> a = F/m
        self.acceleration = net_force / self.mass
        
        # Integrate velocity and position
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        return self.position

# ==========================================
# Sensor with Noise
# ==========================================
class NoisySensor:
    def __init__(self, noise_std=0.01, bias=0.0, delay_samples=0):
        """
        Simulates a noisy sensor
        - noise_std: standard deviation of measurement noise
        - bias: constant measurement bias
        - delay_samples: measurement delay in samples
        """
        self.noise_std = noise_std
        self.bias = bias
        self.delay_samples = delay_samples
        self.measurement_buffer = []

    def measure(self, true_value):
        """Add noise and delay to measurement"""
        # Add noise and bias
        noisy_value = true_value + np.random.normal(0, self.noise_std) + self.bias
        
        # Add delay
        self.measurement_buffer.append(noisy_value)
        if len(self.measurement_buffer) > self.delay_samples:
            return self.measurement_buffer.pop(0)
        else:
            return noisy_value

# ==========================================
# Simulation function
# ==========================================
def simulate(controller, plant, sensor, setpoint=10.0, sim_time=20.0, dt=0.01, 
             power_limit=None):
    """Run closed-loop simulation
    
    Args:
        controller: Controller object
        plant: Plant object
        sensor: Sensor object
        setpoint: Target position
        sim_time: Total simulation time
        dt: Time step
        power_limit: Tuple of (min, max) power limits, or None for no limits
    """
    controller.reset()
    plant.reset()

    t_vals = np.arange(0, sim_time, dt)
    position_vals = []
    velocity_vals = []
    error_vals = []
    control_vals = []
    measured_vals = []
    actual_force_vals = []

    for t in t_vals:
        # Measure current position (with noise)
        measured_position = sensor.measure(plant.position)
        
        # Calculate error
        error = setpoint - measured_position
        
        # Compute control signal
        control_signal = controller.compute(error, dt)
        
        # Apply power limits if specified
        if power_limit is not None:
            control_signal = max(power_limit[0], min(power_limit[1], control_signal))
        
        # Update plant (plant applies the control signal and stores it in applied_force)
        plant.update(control_signal, dt)

        # Store data
        position_vals.append(plant.position)
        velocity_vals.append(plant.velocity)
        error_vals.append(error)
        control_vals.append(control_signal)
        measured_vals.append(measured_position)
        actual_force_vals.append(plant.applied_force)

    return (t_vals, np.array(position_vals), np.array(velocity_vals), 
            np.array(error_vals), np.array(control_vals), np.array(measured_vals),
            np.array(actual_force_vals))

# ==========================================
# Performance metrics
# ==========================================
def analyze_performance(t, x, setpoint=10.0):
    """Calculate standard control system metrics"""
    final_value = x[-1]
    error = np.abs(setpoint - x)

    # Steady-state error
    ss_error = abs(setpoint - final_value)

    # Overshoot
    overshoot = max(0, (np.max(x) - setpoint) / setpoint * 100)

    # Rise time (10% to 90%)
    try:
        idx_10 = np.where(x >= 0.1 * setpoint)[0]
        idx_90 = np.where(x >= 0.9 * setpoint)[0]
        if len(idx_10) > 0 and len(idx_90) > 0:
            t_10 = t[idx_10[0]]
            t_90 = t[idx_90[0]]
            rise_time = t_90 - t_10
        else:
            rise_time = np.nan
    except:
        rise_time = np.nan

    # Settling time (within 2%)
    try:
        settled = np.abs(x - setpoint) <= 0.02 * abs(setpoint)
        idx = np.where(settled)[0]
        if len(idx) > 0:
            # Find last time it left the band
            for i in range(len(settled)-1, 0, -1):
                if not settled[i-1]:
                    settling_time = t[i]
                    break
            else:
                settling_time = t[idx[0]]
        else:
            settling_time = np.nan
    except:
        settling_time = np.nan

    return {
        "steady_state_error": ss_error,
        "overshoot_%": overshoot,
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
    }

# ==========================================
# Main execution
# ==========================================
if __name__ == "__main__":
    # Setpoint
    setpoint = 1.0
    sim_time = 40.0
    dt = 0.01
    
    # ========== Test 1: Standard PID ==========
    plant1 = PhysicalPlant(
        mass=2.0,
        friction_static=1.5,
        friction_kinetic=1.0,
        friction_viscous=0.2,
        disturbance_magnitude=0.5
    )
    
    sensor1 = NoisySensor(noise_std=0.05, bias=0.0, delay_samples=2)
    
    pid = PID(
        kp=15.0,
        ki=5.0,
        kd=8.0,
        output_limits=(-50, 50)
    )
    
    print("Running Standard PID simulation...")
    t_pid, pos_pid, vel_pid, err_pid, ctrl_pid, meas_pid, actual_pid = simulate(
        pid, plant1, sensor1, setpoint, sim_time, dt, power_limit=(-50, 50)
    )
    
    metrics_pid = analyze_performance(t_pid, pos_pid, setpoint)
    
    print("\n===== Standard PID Performance =====")
    for key, value in metrics_pid.items():
        print(f"{key}: {value:.4f}")
    print(f"Final Position: {pos_pid[-1]:.4f}")
    print(f"Final Velocity: {vel_pid[-1]:.4f}")
    
    # ========== Test 2: SmartPID ==========
    plant2 = PhysicalPlant(
        mass=2.0,
        friction_static=1.5,
        friction_kinetic=1.0,
        friction_viscous=0.2,
        disturbance_magnitude=0.5
    )
    
    sensor2 = NoisySensor(noise_std=0.05, bias=0.0, delay_samples=2)
    
    
    spid = SmartPID(
        correction_constant=0.002,
        windup=1.0,
        learning_constant=0.00005,
        max_value=50.0
    )
    
    print("\n\nRunning SmartPID simulation...")
    t_spid, pos_spid, vel_spid, err_spid, ctrl_spid, meas_spid, actual_spid = simulate(
        spid, plant2, sensor2, setpoint, sim_time, dt, power_limit=(-50, 50)
    )
    
    metrics_spid = analyze_performance(t_spid, pos_spid, setpoint)
    
    print("\n===== SmartPID Performance =====")
    for key, value in metrics_spid.items():
        print(f"{key}: {value:.4f}")
    print(f"Final Position: {pos_spid[-1]:.4f}")
    print(f"Final Velocity: {vel_spid[-1]:.4f}")
    print(f"Final Gains - kP: {spid.kP:.4f}, kI: {spid.kI:.4f}, kD: {spid.kD:.4f}")
    
    log_file.close()
    
    # ==========================================
    # Plotting - Comparison
    # ==========================================
    
    # Extract history data (already recorded during simulation)
    # Convert to numpy arrays for easier plotting
    P_history = np.array(spid.history['P'])
    I_history = np.array(spid.history['I'])
    D_history = np.array(spid.history['D'])
    expected_history = np.array(spid.history['expected'])
    constant_history = np.array(spid.history['constant'])
    kP_history = np.array(spid.history['kP'])
    kI_history = np.array(spid.history['kI'])
    kD_history = np.array(spid.history['kD'])
    
    # Plot 1: Position comparison
    plt.figure(figsize=(14, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(t_pid, pos_pid, label='Standard PID', linewidth=2, alpha=0.8)
    plt.plot(t_spid, pos_spid, label='SmartPID', linewidth=2, alpha=0.8)
    plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint', linewidth=1.5)
    plt.title('Position Tracking Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Velocity comparison
    plt.subplot(4, 1, 2)
    plt.plot(t_pid, vel_pid, label='Standard PID', linewidth=2, alpha=0.8)
    plt.plot(t_spid, vel_spid, label='SmartPID', linewidth=2, alpha=0.8)
    plt.axhline(0, color='k', linestyle='-', linewidth=0.5)
    plt.title('Velocity Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Control signal comparison
    plt.subplot(4, 1, 3)
    plt.plot(t_pid, ctrl_pid, label='Standard PID Control', linewidth=2, alpha=0.8)
    plt.plot(t_spid, ctrl_spid, label='SmartPID Control', linewidth=2, alpha=0.8)
    plt.plot(t_pid, actual_pid, label='PID Actual Applied', linewidth=1, alpha=0.5, linestyle='--')
    plt.plot(t_spid, actual_spid, label='SPID Actual Applied', linewidth=1, alpha=0.5, linestyle='--')
    plt.axhline(50, color='r', linestyle=':', linewidth=1, alpha=0.7, label='Power Limit')
    plt.axhline(-50, color='r', linestyle=':', linewidth=1, alpha=0.7)
    plt.title('Control Signal Comparison (Commanded vs Actual)', fontsize=14, fontweight='bold')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: SmartPID Internal Components Over Time
    plt.subplot(4, 1, 4)
    plt.plot(t_spid, P_history, label='P Component', linewidth=2, alpha=0.8)
    plt.plot(t_spid, I_history, label='I Component', linewidth=2, alpha=0.8)
    plt.plot(t_spid, D_history, label='D Component', linewidth=2, alpha=0.8)
    plt.title('SmartPID: P, I, D Components Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Component Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 5: SmartPID Learning Dynamics
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(t_spid, expected_history, linewidth=2, alpha=0.8, color='orange')
    plt.title('Expected Output Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Expected Value')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(t_spid, constant_history, linewidth=2, alpha=0.8, color='green')
    plt.title('Learning Constant Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Avg |Constant|')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 3)
    plt.plot(t_spid, kP_history, linewidth=2, alpha=0.8, label='kP')
    plt.title('kP Gain Evolution', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('kP Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 2, 4)
    plt.plot(t_spid, kI_history, linewidth=2, alpha=0.8, label='kI', color='orange')
    plt.title('kI Gain Evolution', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('kI Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 2, 5)
    plt.plot(t_spid, kD_history, linewidth=2, alpha=0.8, label='kD', color='green')
    plt.title('kD Gain Evolution', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('kD Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 2, 6)
    plt.plot(t_spid, P_history, label='P', linewidth=1.5, alpha=0.8)
    plt.plot(t_spid, I_history, label='I', linewidth=1.5, alpha=0.8)
    plt.plot(t_spid, D_history, label='D', linewidth=1.5, alpha=0.8)
    plt.title('All PID Components', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 6: Expected vs Actual Velocity for SmartPID
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_spid, expected_history, label='Expected (from history)', linewidth=2, alpha=0.8)
    plt.plot(t_spid, vel_spid, label='Actual Plant Velocity', linewidth=2, alpha=0.8)
    plt.title('SmartPID: Expected vs Actual Velocity', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(expected_history, vel_spid, alpha=0.5, s=10)
    plt.plot([-50, 50], [-50, 50], 'r--', linewidth=2, label='Perfect Match')
    plt.title('Expected vs Actual (Scatter)', fontsize=14, fontweight='bold')
    plt.xlabel('Expected Velocity')
    plt.ylabel('Actual Plant Velocity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 7: Phase portraits
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pos_pid, vel_pid, linewidth=2, alpha=0.8)
    plt.scatter([setpoint], [0], color='red', s=100, marker='*', 
                label='Target', zorder=5)
    plt.title('Standard PID Phase Portrait', fontsize=12, fontweight='bold')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(pos_spid, vel_spid, linewidth=2, alpha=0.8, color='orange')
    plt.scatter([setpoint], [0], color='red', s=100, marker='*', 
                label='Target', zorder=5)
    plt.title('SmartPID Phase Portrait', fontsize=12, fontweight='bold')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()