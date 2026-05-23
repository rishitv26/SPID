# SPID

SPID (Self-tuning PID) is a motion control algorithm that treats PID as a single-layer neural network and uses gradient descent to adapt its gains online during operation. It is designed to outperform generic PID in environments where system dynamics are variable or unknown ahead of time, especially in non-linear environments where PID tends to perform poorly.

A pre-print documenting the design, benchmarks, and analysis is awaiting publication by May, 2026.

## Motivation

Standard PID requires manual gain tuning, which is environment-specific and does not adapt at runtime. Fully AI-based controllers (e.g., reinforcement learning) are very energy consuming and thus require much more energy than what can be available in some applications. SPID tries to act as the solution in the middle by retaining the structure and interpretability of PID while using backpropagation to update P, I, and D gains simultaneuosly.

## How It Works

PID can be interpreted as a linear neural network with no activation function, where the three gains (kP, kI, kD) are the learnable weights. In SPID, a cost function is defined relative to error, which is then using in conjuction with backpropogation to converge to the 'best' P, I, and D gains within the plant's environment. Rigourous notation and testing results shall be released with the pre-print

## Files

| File | Description |
|---|---|
| `spid.py` | Core implementation: `SmartPID` and baseline `PID` classes |
| `simulation.py` | Simulation environment for benchmarking |
| `descend.py` | Gradient descent and gain update logic |
| `pid_simulation_log.txt` | Sample simulation output |
| `requirements.txt` | Python dependencies |

## Usage

```python
from spid import SmartPID

controller = SmartPID(
    correction_constant=1.0,
    windup=0.2,
    learning_constant=0.001,
    max_value=127.0
)

# In your control loop:
output = controller.compute(error, dt)
```

The controller requires no pre-tuning. Gains initialize to a placeholder value and adapt from the first timestep.

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `correction_constant` | Scales the target curve shape | `1.0` |
| `windup` | Integral anti-windup threshold | `0.2` |
| `learning_constant` | Base learning rate | `0.001` |
| `max_value` | Maximum control output | `127.0` |

## Benchmarks (Planned)

SPID will be evaluated against:
- PID tuned via Ziegler-Nichols
- PID tuned via Particle Swarm Optimization (PSO)
- Model Reference Adaptive Control (MRAC)

Metrics: RMSE, settling time, overshoot percentage across small and large setpoint ranges.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

## Progress thus far

SPID works well within an ideal environment with no external forces. A processing simulation was written to demonstrate this. More testing and iterating has to be done for unideal environments.

https://github.com/user-attachments/assets/7d376503-ee4c-4a0f-9093-4ee74eeaa01e

> Note that because of an ideal environment, kP, kI, and kD may not reflect what a real system might exhibit. Learning rate was slowed down intentionally to enhance proof-of-concept

## License

MIT
