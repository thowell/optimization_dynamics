append!(empty!(LOAD_PATH), Base.DEFAULT_LOAD_PATH)
using Pkg

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
include(joinpath(pkgdir, "models/acrobot/model.jl"))
include(joinpath(pkgdir, "models/acrobot/codegen.jl"))

# cart-pole 
include(joinpath(pkgdir, "models/cartpole/model.jl"))
include(joinpath(pkgdir, "models/cartpole/simulator_friction.jl"))
include(joinpath(pkgdir, "models/cartpole/simulator_no_friction.jl"))
include(joinpath(pkgdir, "models/cartpole/codegen.jl"))

# hopper from RoboDojo.jl 

# planar push 
include(joinpath(pkgdir, "models/planar_push/model.jl"))
include(joinpath(pkgdir, "models/planar_push/simulator.jl"))
include(joinpath(pkgdir, "models/planar_push/codegen.jl"))

# rocket
include(joinpath(pkgdir, "models/rocket/model.jl"))
include(joinpath(pkgdir, "models/rocket/simulator.jl"))
include(joinpath(pkgdir, "models/rocket/codegen.jl"))


