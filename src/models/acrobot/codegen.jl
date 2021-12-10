# generate residual methods
nq = acrobot_impact.nq
nu = acrobot_impact.nu
nc = acrobot_impact.nc
nz = nq + nc + nc
nz_nominal = nq
nθ = nq + nq + nu + 1

# Declare variables
@variables z[1:nz]
@variables z_nominal[1:nz_nominal]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r_impact = residual(acrobot_impact, z, θ, κ)
r_impact = Symbolics.simplify.(r_impact)
rz_impact = Symbolics.jacobian(r_impact, z, simplify=true)
rθ_impact = Symbolics.jacobian(r_impact, θ, simplify=true)

# Build function
r_acrobot_impact_func = build_function(r_impact, z, θ, κ)[2]
rz_acrobot_impact_func = build_function(rz_impact, z, θ)[2]
rθ_acrobot_impact_func = build_function(rθ_impact, z, θ)[2]

rz_acrobot_impact_array = similar(rz_impact, Float64)
rθ_acrobot_impact_array = similar(rθ_impact, Float64)

path = @get_scratch!("acrobot")
@save joinpath(path, "impact.jld2") r_acrobot_impact_func rz_acrobot_impact_func rθ_acrobot_impact_func rz_acrobot_impact_array rθ_acrobot_impact_array
@load joinpath(path, "impact.jld2") r_acrobot_impact_func rz_acrobot_impact_func rθ_acrobot_impact_func rz_acrobot_impact_array rθ_acrobot_impact_array

# Residual
r_nominal = residual(acrobot_nominal, z_nominal, θ, κ)
r_nominal = Symbolics.simplify.(r_nominal)
rz_nominal = Symbolics.jacobian(r_nominal, z_nominal, simplify=true)
rθ_nominal = Symbolics.jacobian(r_nominal, θ, simplify=true)

# Build function
r_acrobot_nominal_func = build_function(r_nominal, z_nominal, θ, κ)[2]
rz_acrobot_nominal_func = build_function(rz_nominal, z_nominal, θ)[2]
rθ_acrobot_nominal_func = build_function(rθ_nominal, z_nominal, θ)[2]

rz_acrobot_nominal_array = similar(rz_nominal, Float64)
rθ_acrobot_nominal_array = similar(rθ_nominal, Float64)

path = @get_scratch!("acrobot")
@save joinpath(path, "nominal.jld2") r_acrobot_nominal_func rz_acrobot_nominal_func rθ_acrobot_nominal_func rz_acrobot_nominal_array rθ_acrobot_nominal_array
@load joinpath(path, "nominal.jld2") r_acrobot_nominal_func rz_acrobot_nominal_func rθ_acrobot_nominal_func rz_acrobot_nominal_array rθ_acrobot_nominal_array
