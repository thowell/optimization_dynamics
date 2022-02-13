module OptimizationDynamics

using LinearAlgebra 
using BenchmarkTools
using Symbolics 
using IfElse
using JLD2
import JLD2: load 
using DirectTrajectoryOptimization
using IterativeLQR
using MeshCat 
using Colors
using CoordinateTransformations 
using GeometryBasics
using Rotations 
using RoboDojo 
import RoboDojo: LinearSolver, LUSolver, Model, ResidualMethods, Space, Disturbances, IndicesZ, InteriorPoint, EmptySolver, Policy, Trajectory, GradientTrajectory, InteriorPointOptions, IndicesOptimization, interior_point, interior_point_solve!, bilinear_violation, residual_violation, general_correction_term!, r!, rz!, rθ!, linear_solve!, lu_solver, empty_policy, empty_disturbances, friction_coefficients, SimulatorStatistics, SimulatorOptions, indices_θ, num_data, initialize_z!, initialize_θ!, indices_z, indices_θ, simulate!, policy, process!, Simulator, cone_product, lagrangian_derivatives, Indicesθ
using Scratch 

export LinearSolver, LUSolver, Model, ResidualMethods, Space, Disturbances, IndicesZ, InteriorPoint, EmptySolver, Policy, Trajectory, GradientTrajectory, InteriorPointOptions, IndicesOptimization, interior_point, interior_point_solve!, bilinear_violation, residual_violation, general_correction_term!, r!, rz!, rθ!, linear_solve!, lu_solver, empty_policy, empty_disturbances, friction_coefficients, SimulatorStatistics, SimulatorOptions, indices_θ, num_data, initialize_z!, initialize_θ!, indices_z, indices_θ, simulate!, policy, process!, Simulator, cone_product, lagrangian_derivatives, Indicesθ

export 
    load

export 
    Visualizer, render, open, visualize!  

include("dynamics.jl")
include("ls.jl")
include("gradient_bundle.jl")

export 
    ImplicitDynamics, f, fx, fu, state_to_configuration,
    f_gb, fx_gb, fu_gb

# acrobot 
include("../src/models/acrobot/model.jl")
include("../src/models/acrobot/simulator_impact.jl") 
include("../src/models/acrobot/simulator_nominal.jl") 
include("../src/models/acrobot/visuals.jl") 
path_acrobot = @get_scratch!("acrobot")
@load joinpath(path_acrobot, "impact.jld2") r_acrobot_impact_func rz_acrobot_impact_func rθ_acrobot_impact_func rz_acrobot_impact_array rθ_acrobot_impact_array
@load joinpath(path_acrobot, "nominal.jld2") r_acrobot_nominal_func rz_acrobot_nominal_func rθ_acrobot_nominal_func rz_acrobot_nominal_array rθ_acrobot_nominal_array

# cartpole
include("../src/models/cartpole/model.jl")
include("../src/models/cartpole/simulator_friction.jl")
include("../src/models/cartpole/simulator_frictionless.jl")
include("../src/models/cartpole/visuals.jl")
path_cartpole = @get_scratch!("cartpole")
@load joinpath(path_cartpole, "friction.jld2") r_cartpole_friction_func rz_cartpole_friction_func rθ_cartpole_friction_func rz_cartpole_friction_array rθ_cartpole_friction_array
@load joinpath(path_cartpole, "frictionless.jld2") r_cartpole_frictionless_func rz_cartpole_frictionless_func rθ_cartpole_frictionless_func rz_cartpole_frictionless_array rθ_cartpole_frictionless_array


# hopper from RoboDojo.jl 

# planar push 
include("../src/models/planar_push/model.jl")
include("../src/models/planar_push/simulator.jl")
include("../src/models/planar_push/visuals.jl")
path_planarpush = @get_scratch!("planarpush")
@load joinpath(path_planarpush, "residual.jld2") r_pp_func rz_pp_func rθ_pp_func rz_pp_array rθ_pp_array

# rocket
include("../src/models/rocket/model.jl")
include("../src/models/rocket/simulator.jl")
include("../src/models/rocket/dynamics.jl")
include("../src/models/rocket/visuals.jl")
path_rocket = @get_scratch!("rocket")
@load joinpath(path_rocket, "residual.jld2") r_rocket_func rz_rocket_func rθ_rocket_func rz_rocket_array rθ_rocket_array
@load joinpath(path_rocket, "projection.jld2") r_proj_func rz_proj_func rθ_proj_func rz_proj_array rθ_proj_array

include("../src/models/visualize.jl")

export 
    acrobot_impact, acrobot_nominal,
    cartpole_friction, cartpole_frictionless, 
    planarpush, 
    rocket, RocketInfo, f_rocket_proj, fx_rocket_proj, fu_rocket_proj, f_rocket, fx_rocket, fu_rocket

export 
    r_acrobot_impact_func, rz_acrobot_impact_func, rθ_acrobot_impact_func, rz_acrobot_impact_array, rθ_acrobot_impact_array,
    r_acrobot_nominal_func, rz_acrobot_nominal_func, rθ_acrobot_nominal_func, rz_acrobot_nominal_array, rθ_acrobot_nominal_array,
    r_cartpole_friction_func, rz_cartpole_friction_func, rθ_cartpole_friction_func, rz_cartpole_friction_array, rθ_cartpole_friction_array,
    r_cartpole_frictionless_func, rz_cartpole_frictionless_func, rθ_cartpole_frictionless_func, rz_cartpole_frictionless_array, rθ_cartpole_frictionless_array,
    r_pp_func, rz_pp_func, rθ_pp_func, rz_pp_array, rθ_pp_array,
    r_rocket_func, rz_rocket_func, rθ_rocket_func, rz_rocket_array, rθ_rocket_array,
    r_proj_func, rz_proj_func, rθ_proj_func, rz_proj_array, rθ_proj_array

end # module
