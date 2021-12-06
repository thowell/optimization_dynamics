append!(empty!(LOAD_PATH), Base.DEFAULT_LOAD_PATH)
using Pkg

# Add iLQR
Pkg.add(url="https://github.com/thowell/IterativeLQR.jl") # add this way temporarily 


################################################################################
# Generate notebooks
# ################################################################################
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
include("../models/acrobot/model.jl")
include("../models/acrobot/codegen.jl") 

# cartpole
include("../models/cartpole/model.jl")
include("../models/cartpole/simulator_friction.jl")
include("../models/cartpole/simulator_no_friction.jl")
include("../models/cartpole/codegen.jl")

# hopper from RoboDojo.jl 

# planar push 
include("../models/planar_push/model.jl")
include("../models/planar_push/simulator.jl")
include("../models/planar_push/codegen.jl")

# rocket
include("../models/rocket/model.jl")
include("../models/rocket/simulator.jl")
include("../models/rocket/codegen.jl")

