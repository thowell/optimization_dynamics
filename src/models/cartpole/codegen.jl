path = @get_scratch!("cartpole")

nq = cartpole_friction.nq
nu = cartpole_friction.nu
nc = cartpole_friction.nc
nz = nq + 4 * nc
nθ = nq + nq + nu + 2 + 1
nz_nf = nq
nθ_nf = nq + nq + nu + 1

# Declare variables
@variables z[1:nz]
@variables θ[1:nθ]
@variables κ[1:1]
@variables z_nf[1:nz_nf]
@variables θ_nf[1:nθ_nf]
@variables κ_nf[1:1]

# Residual
r_friction= residual(cartpole_friction, z, θ, κ)
r_friction = Symbolics.simplify.(r_friction)
rz_friction = Symbolics.jacobian(r_friction, z, simplify=true)
rθ_friction = Symbolics.jacobian(r_friction, θ, simplify=true)

# Build function
r_cartpole_friction_func = build_function(r_friction, z, θ, κ)[2]
rz_cartpole_friction_func = build_function(rz_friction, z, θ)[2]
rθ_cartpole_friction_func = build_function(rθ_friction, z, θ)[2]

rz_cartpole_friction_array = similar(rz_friction, Float64)
rθ_cartpole_friction_array = similar(rθ_friction, Float64)

@save joinpath(path, "friction.jld2") r_cartpole_friction_func rz_cartpole_friction_func rθ_cartpole_friction_func rz_cartpole_friction_array rθ_cartpole_friction_array
@load joinpath(path, "friction.jld2") r_cartpole_friction_func rz_cartpole_friction_func rθ_cartpole_friction_func rz_cartpole_friction_array rθ_cartpole_friction_array

# Residual
r_frictionless = residual(cartpole_frictionless, z_nf, θ_nf, κ_nf)
r_frictionless = Symbolics.simplify.(r_frictionless)
rz_frictionless = Symbolics.jacobian(r_frictionless, z_nf, simplify=true)
rθ_frictionless = Symbolics.jacobian(r_frictionless, θ_nf, simplify=true)

# Build function
r_cartpole_frictionless_func = build_function(r_frictionless, z_nf, θ_nf, κ_nf)[2]
rz_cartpole_frictionless_func = build_function(rz_frictionless, z_nf, θ_nf)[2]
rθ_cartpole_frictionless_func = build_function(rθ_frictionless, z_nf, θ_nf)[2]

rz_cartpole_frictionless_array = similar(rz_frictionless, Float64)
rθ_cartpole_frictionless_array = similar(rθ_frictionless, Float64)

@save joinpath(path, "frictionless.jld2") r_cartpole_frictionless_func rz_cartpole_frictionless_func rθ_cartpole_frictionless_func rz_cartpole_frictionless_array rθ_cartpole_frictionless_array
@load joinpath(path, "frictionless.jld2") r_cartpole_frictionless_func rz_cartpole_frictionless_func rθ_cartpole_frictionless_func rz_cartpole_frictionless_array rθ_cartpole_frictionless_array