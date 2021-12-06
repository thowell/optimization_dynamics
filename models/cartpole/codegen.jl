nq = cartpole.nq
nu = cartpole.nu
nc = cartpole.nc
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
r = residual(cartpole, z, θ, κ)
r = Symbolics.simplify.(r)
rz = Symbolics.jacobian(r, z, simplify=true)
rθ = Symbolics.jacobian(r, θ, simplify=true)

# Build function
r_func = build_function(r, z, θ, κ)[2]
rz_func = build_function(rz, z, θ)[2]
rθ_func = build_function(rθ, z, θ)[2]

rz_array = similar(rz, Float64)
rθ_array = similar(rθ, Float64)

path = @get_scratch!("cartpole")

@save joinpath(path, "friction.jld2") r_func rz_func rθ_func rz_array rθ_array
@load joinpath(path, "friction.jld2") r_func rz_func rθ_func rz_array rθ_array

# Residual
r_no_friction = residual_no_friction(cartpole, z_nf, θ_nf, κ_nf)
r_no_friction = Symbolics.simplify.(r_no_friction)
rz_no_friction = Symbolics.jacobian(r_no_friction, z_nf, simplify=true)
rθ_no_friction = Symbolics.jacobian(r_no_friction, θ_nf, simplify=true)

# Build function
r_no_friction_func = build_function(r_no_friction, z_nf, θ_nf, κ_nf)[2]
rz_no_friction_func = build_function(rz_no_friction, z_nf, θ_nf)[2]
rθ_no_friction_func = build_function(rθ_no_friction, z_nf, θ_nf)[2]

rz_no_friction_array = similar(rz_no_friction, Float64)
rθ_no_friction_array = similar(rθ_no_friction, Float64)

@save joinpath(path, "no_friction.jld2") r_no_friction_func rz_no_friction_func rθ_no_friction_func rz_no_friction_array rθ_no_friction_array
@load joinpath(path, "no_friction.jld2") r_no_friction_func rz_no_friction_func rθ_no_friction_func rz_no_friction_array rθ_no_friction_array