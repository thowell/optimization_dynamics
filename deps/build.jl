append!(empty!(LOAD_PATH), Base.DEFAULT_LOAD_PATH)
using Pkg

################################################################################
# Generate notebooks
################################################################################
exampledir = joinpath(@__DIR__, "..", "examples")
Pkg.activate(exampledir)
Pkg.instantiate()
include(joinpath(exampledir, "generate_notebooks.jl"))

################################################################################
# Build simulation environments
################################################################################
pkgdir = joinpath(@__DIR__, "..")
Pkg.activate(pkgdir)

using JLD2 
using Symbolics
using LinearAlgebra
using Scratch 
using RoboDojo 
using Rotations
import RoboDojo: Model, lagrangian_derivatives, IndicesZ, cone_product

# acrobot 
include("../src/models/acrobot/model.jl")
include("../src/models/acrobot/simulator_impact.jl")
include("../src/models/acrobot/simulator_nominal.jl")
include("../src/models/acrobot/codegen.jl") 

# cartpole
include("../src/models/cartpole/model.jl")
include("../src/models/cartpole/simulator_friction.jl")
include("../src/models/cartpole/simulator_frictionless.jl")
include("../src/models/cartpole/codegen.jl")

# hopper from RoboDojo.jl 

# planar push 
include("../src/models/planar_push/model.jl")
include("../src/models/planar_push/simulator.jl")
include("../src/models/planar_push/codegen.jl")

# rocket
include("../src/models/rocket/model.jl")
include("../src/models/rocket/simulator.jl")
include("../src/models/rocket/codegen.jl")

