# generate residual methods
nq = acrobot.nq
nu = acrobot.nu
nc = acrobot.nc
nz = nq + nc + nc
nz_no_impact = nq
nθ = nq + nq + nu + 1

# Declare variables
@variables z[1:nz]
@variables z_no_impact[1:nz_no_impact]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r = residual(acrobot, z, θ, κ)
r = Symbolics.simplify.(r)
rz = Symbolics.jacobian(r, z, simplify=true)
rθ = Symbolics.jacobian(r, θ, simplify=true)

# Build function
r_func = build_function(r, z, θ, κ)[2]
rz_func = build_function(rz, z, θ)[2]
rθ_func = build_function(rθ, z, θ)[2]

rz_array = similar(rz, Float64)
rθ_array = similar(rθ, Float64)

path = @get_scratch!("acrobot")
@save joinpath(path, "impact.jld2") r_func rz_func rθ_func rz_array rθ_array
@load joinpath(path, "impact.jld2") r_func rz_func rθ_func rz_array rθ_array

# Residual
r_no_impact = residual_no_impact(acrobot, z_no_impact, θ, κ)
r_no_impact = Symbolics.simplify.(r_no_impact)
rz_no_impact = Symbolics.jacobian(r_no_impact, z_no_impact, simplify=true)
rθ_no_impact = Symbolics.jacobian(r_no_impact, θ, simplify=true)

# Build function
r_no_impact_func = build_function(r_no_impact, z_no_impact, θ, κ)[2]
rz_no_impact_func = build_function(rz_no_impact, z_no_impact, θ)[2]
rθ_no_impact_func = build_function(rθ_no_impact, z_no_impact, θ)[2]

rz_no_impact_array = similar(rz_no_impact, Float64)
rθ_no_impact_array = similar(rθ_no_impact, Float64)

path = @get_scratch!("acrobot")
@save joinpath(path, "no_impact.jld2") r_no_impact_func rz_no_impact_func rθ_no_impact_func rz_no_impact_array rθ_no_impact_array
@load joinpath(path, "no_impact.jld2") r_no_impact_func rz_no_impact_func rθ_no_impact_func rz_no_impact_array rθ_no_impact_array

eval(r_no_impact_func)(zeros(nz), rand(nz), rand(nθ), [1.0])
eval(rz_no_impact_func)(zeros(nz, nz), rand(nz), rand(nθ))
eval(rθ_no_impact_func)(zeros(nz, nθ), rand(nz), rand(nθ))