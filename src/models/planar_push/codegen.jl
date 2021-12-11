path = @get_scratch!("planarpush")

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
r_pp = residual(planarpush, z, θ, κ)
rz_pp = Symbolics.jacobian(r_pp, z)
rθ_pp = Symbolics.jacobian(r_pp, θ)

# Build function
r_pp_func = build_function(r_pp, z, θ, κ)[2]
rz_pp_func = build_function(rz_pp, z, θ)[2]
rθ_pp_func = build_function(rθ_pp, z, θ)[2]
rz_pp_array = similar(rz_pp, Float64)
rθ_pp_array = similar(rθ_pp, Float64)

@save joinpath(path, "residual.jld2") r_pp_func rz_pp_func rθ_pp_func rz_pp_array rθ_pp_array
@load joinpath(path, "residual.jld2") r_pp_func rz_pp_func rθ_pp_func rz_pp_array rθ_pp_array