using Symbolics 
using LinearAlgebra
using IfElse
using Scratch 
using JLD2
using Pkg 
abstract type Model{T} end
include("model.jl")
include("utils.jl")

nq = planarpush.nq
nu = planarpush.nu
nc = planarpush.nc
nz = num_var(planarpush)
nθ = 2 * planarpush.nq + planarpush.nu + planarpush.nw + 1 

# Declare variables
@variables z[1:nz]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r = vec(residual(planarpush, [z...], [θ...], [κ...]))
rz = Symbolics.jacobian(r, [z...])
rθ = Symbolics.jacobian(r, [θ...])

# Build function
r_func = build_function(r, [z...], [θ...], [κ...])[1]
rz_func = build_function(rz, [z...], [θ...])[1]
rθ_func = build_function(rθ, [z...], [θ...])[1];

rz_array = similar(rz, Float64)
rθ_array = similar(rθ, Float64)

# path = @get_scratch!("planarpush")

# @save joinpath(path, "residual.jld2") r_func rz_func rθ_func rz_array rθ_array
# @load joinpath(path, "residual.jld2") r_func rz_func rθ_func rz_array rθ_array

# @save "/home/taylor/Research/optimization_based_dynamics/residual.jld2" r_func rz_func rθ_func rz_array rθ_array
# @load "/home/taylor/Research/optimization_based_dynamics/residual.jld2" r_func rz_func rθ_func rz_array rθ_array

# @save "/home/taylor/Research/optimization_based_dynamics/residual_alt.jld2" r_func rz_func rθ_func rz_array rθ_array
# @load "/home/taylor/Research/optimization_based_dynamics/residual_alt.jld2" r_func rz_func rθ_func rz_array rθ_array

eval(r_func)(rand(nz), rand(nθ), [1.0])

using ForwardDiff
res(a) = eval(r_func)(a, rand(nθ), [1.0])
ForwardDiff.jacobian(res, rand(nz))


eval(rz_func)(rand(nz), rand(nθ))
eval(rθ_func)(rand(nz), rand(nθ))

eval(r_func)(ones(nz), rand(nz), rand(nθ), [1.0])
eval(rz_func)(rz_array, rand(nz), rand(nθ))
eval(rz_func)(zeros(nz, nz), rand(nz), rand(nθ))
eval(rθ_func)(rθ_array, rand(nz), rand(nθ))
eval(rθ_func)(zeros(nz, nθ), rand(nz), rand(nθ))

