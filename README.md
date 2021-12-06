# Trajectory Optimization with Optimization-Based Dynamics 

This repository contains the implementation and examples from our paper: [Trajectory Optimization with Optimization-Based Dynamics](https://arxiv.org/abs/2109.04928).

## Installation
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add https://github.com/thowell/optimization_dynamics
```
This will install and build the package. 

[Notebooks](examples/README.md) can be run for the following examples:

## planar push
<img src="animations/planar_push_rotate.gif" alt="drawing" width="400"/> 

## acrobot with joint limits
<img src="animations/acrobot_joint_limits.gif" alt="drawing" width="400"/>

## cart-pole with joint friction
<img src="animations/cartpole_friction_35.gif" alt="drawing" width="400"/>

## hopper gait
<img src="animations/hopper_gait_1.gif" alt="drawing" width="400"/>

## rocket with thrust limits
<img src="animations/starship_bellyflop_landing.gif" alt="drawing" width="400"/>

Additional comparisons with [MuJoCo](examples/comparisons/acrobot) and [contact-implicit trajectory optimization](examples/comparisons/hopper.jl) are available.

