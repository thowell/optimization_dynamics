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
    Visualizer, 
    render, 
    open

include("dynamics.jl")
include("ls.jl")
include("gradient_bundle.jl")

# acrobot 
include("../src/models/acrobot/model.jl")
# include("../src/models/acrobot/simulator_impact.jl") 
# include("../src/models/acrobot/simulator_no_impact.jl") 
include("../src/models/acrobot/visuals.jl") 
path_acrobot = @get_scratch!("acrobot")

# cartpole
include("../src/models/cartpole/model.jl")
# include("../src/models/cartpole/simulator_friction.jl")
# include("../src/models/cartpole/simulator_no_friction.jl")
include("../src/models/cartpole/visuals.jl")
path_cartpole = @get_scratch!("cartpole")

# hopper from RoboDojo.jl 

# planar push 
include("../src/models/planar_push/model.jl")
include("../src/models/planar_push/simulator.jl")
include("../src/models/planar_push/visuals.jl")
path_planarpush = @get_scratch!("rocket")

# rocket
include("../src/models/rocket/model.jl")
include("../src/models/rocket/simulator.jl")
include("../src/models/rocket/visuals.jl")
path_rocket = @get_scratch!("rocket")


include("../src/models/visualize.jl")

export 
    path_acrobot, 
    path_cartpole,
    path_planarpush,
    path_rocket

end # module
