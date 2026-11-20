# Project Proposal: Vision Guided Search Robot

**Affiliation:** Department of Innovative Technologies, The University of Applied Sciences and Arts of Southern Switzerland (SUPSI), Lugano, Switzerland

## Introduction

We propose the development of an autonomous two-wheel differential-drive mobile robot capable of exploring a bounded environment, avoiding obstacles, and locating a predefined target using visual information. The project will focus primarily on developing and evaluating the robot's control and navigation algorithms within the Swift simulation environment, with subsequent extensions to Gazebo and a physical prototype.

The main objective is to implement a complete mobile robot simulation that integrates differential-drive kinematics, feedback control, autonomous navigation, and target-search behaviour. We will progressively increase the complexity of the simulation, starting from ideal conditions and introducing measurement uncertainty and limited visual perception. This incremental approach will allow us to evaluate individual components before combining them into a complete autonomous system.

## Development Methodology

The primary deliverable will be a functional autonomous mobile robot simulation implemented in Swift. We will develop a simplified robot model and a bounded indoor environment containing obstacles and a predefined target.

### Robot Modelling and Kinematics

We will model the robot as a differential-drive platform with two independently actuated wheels. Its motion will be described using the differential-drive kinematic model, including the non-holonomic constraint that prevents lateral motion.

The robot's position and orientation will be represented in a two-dimensional coordinate system using an $\mathrm{SE}(2)$ transformation. We will implement discrete-time motion updates using Euler numerical integration, allowing the robot's trajectory to be simulated from its linear and angular velocity commands.

The initial model will assume ideal actuation and perfect knowledge of the robot's state. Subsequently, we will introduce uncertainties to investigate their influence on navigation accuracy and control performance.

### Motion Control and Autonomous Navigation

We will implement a hierarchical control strategy that combines low-level motion commands with high-level navigation objectives. The robot will initially use open-loop movement commands, followed by closed-loop feedback control based on its estimated position and orientation.

Navigation will be implemented using proportional go-to-point controllers, angle wrapping, and turn-drive-turn manoeuvres. These methods will allow the robot to reach specified waypoints while maintaining the non-holonomic constraints of its motion.

To enable autonomous exploration, we will define a bounded environment in which the robot must navigate without colliding with obstacles. We will initially assume that obstacle positions are known, allowing us to focus on the navigation and control algorithms independently of perception.

### Target Search and Tracking

We will extend the navigation system with a target-search behaviour. The robot will explore the environment until it identifies a predefined target, after which it will navigate towards it.

Initially, the target position will be directly accessible to the robot. We will then introduce measurement noise and uncertainty to investigate the robustness of the search strategy.

As a further extension within Swift, we will implement target detection based on a simulated monocular camera. Since Swift does not provide a native camera sensor, we will approximate its functionality through a custom virtual camera model. At each simulation step, the camera will be represented by a field of view defined relative to the robot's position and orientation. We will determine which objects fall within this field of view using their relative positions, angular visibility limits, and distance constraints.

This approach will provide a camera-like observation interface without requiring actual image rendering. Objects outside the field of view will remain unknown to the robot, while visible objects will be available to the detection and navigation algorithms. We will additionally introduce detection uncertainty and false negatives to approximate imperfect visual perception.

This abstraction will allow us to develop and evaluate perception-dependent control strategies within Swift while keeping the implementation compatible with the simulator's capabilities.

### Simulation Evaluation and Analysis

We will evaluate the system through progressively more challenging simulation scenarios:

1. **Ideal conditions:** Known robot state, obstacle positions, and target location.

2. **Uncertain information:** Introduction of measurement noise and imperfect state information.

3. **Limited perception:** Replacement of direct environmental knowledge with observations from the virtual camera.

We will analyse navigation accuracy, trajectory efficiency, collision frequency, target-search success rate, and convergence time. We will also generate plots illustrating robot trajectories, orientation changes, control inputs, and relevant performance metrics.

These experiments will allow us to compare the different implementations and assess how uncertainty and limited perception affect autonomous behaviour.

## Extensions

The following stages are considered optional extensions beyond the primary Swift deliverable. Their purpose is to investigate the transferability of the developed control architecture to more realistic simulation and physical hardware.

### Gazebo and ROS Integration

We will investigate transferring the robot model and control architecture to Gazebo [1], using ROS2 [2] for communication between the simulated robot and its control algorithms.

Unlike the simplified Swift implementation, Gazebo will allow us to introduce more realistic physical dynamics and native sensor simulations. In particular, we will replace the virtual camera abstraction with an actual simulated camera sensor, providing image data for visual processing. We will also incorporate more realistic motor behaviour, sensor noise, and physical interactions.

This stage will allow us to evaluate the robustness of the navigation system under more realistic conditions and prepare the software architecture for hardware deployment.

### Physical Robot Implementation

As a final extension, we will construct a small differential-drive robot using an ESP32-S3 controller, two N20 geared motors with encoders, and a TB6612FNG motor driver. The controller will also provide access to a GC2145 monocular camera, enabling visual observations on the physical platform. Rechargeable batteries will provide onboard power, while a small speaker will support optional audio feedback [3].

We will implement low-level motor control, encoder-based motion feedback, and camera interfacing on the microcontroller. The high-level navigation and target-search algorithms developed in simulation will then be adapted to the computational and hardware constraints of the physical platform.

## Expected Outcomes

The primary outcome will be a documented Swift simulation of an autonomous differential-drive robot capable of navigating a bounded environment, avoiding obstacles, exploring its surroundings, and locating a specified target. The implementation will demonstrate the application of mobile robot kinematics, feedback control, numerical integration, and perception-dependent navigation.

We will provide the source code, simulation scenarios, experimental results, and visualisations, accompanied by an analysis of the system's performance and limitations.

The optional Gazebo and physical implementations will further demonstrate the potential for transferring the developed algorithms from an abstract simulation to increasingly realistic environments.

## References

[1] C. Aguero et al., “Inside the Virtual Robotics Challenge: Simulating Real-Time Robotic Disaster Response,” *Automation Science and Engineering, IEEE Transactions on*, vol. 12, no. 2, pp. 494–506, Apr. 2015, doi: 10.1109/TASE.2014.2368997.

[2] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, “Robot Operating System 2: Design, architecture, and uses in the wild,” *Science Robotics*, vol. 7, no. 66, p. eabm6074, 2022, doi: 10.1126/scirobotics.abm6074.

[3] J. Wang and D. Herath, “What Makes Robots? Sensors, Actuators, and Algorithms,” *Foundations of Robotics*, pp. 177–203, 2022, doi: 10.1007/978-981-19-1983-1_7.