
nq = cartpole.nq
nu = cartpole.nu
nc = cartpole.nc
nz = num_var(cartpole)
nz_no_friction = nq
nθ = num_data(cartpole, nf=length(friction_coefficients(cartpole)))

# Declare variables
@variables z[1:nz]
@variables z_no_friction[1:nz_no_friction]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r = vec(residual(cartpole, z, θ, κ))
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
r_no_friction = residual_no_friction(cartpole, z_no_friction, θ, κ)
r_no_friction = Symbolics.simplify.(r_no_friction)
rz_no_friction = Symbolics.jacobian(r_no_friction, z_no_friction, simplify=true)
rθ_no_friction = Symbolics.jacobian(r_no_friction, θ, simplify=true)

# Build function
r_no_friction_func = build_function(r_no_friction, z_no_friction, θ, κ)[2]
rz_no_friction_func = build_function(rz_no_friction, z_no_friction, θ)[2]
rθ_no_friction_func = build_function(rθ_no_friction, z_no_friction, θ)[2]

rz_no_friction_array = similar(rz_no_friction, Float64)
rθ_no_friction_array = similar(rθ_no_friction, Float64)

@save joinpath(path, "no_friction.jld2") r_no_friction_func rz_no_friction_func rθ_no_friction_func rz_no_friction_array rθ_no_friction_array
@load joinpath(path, "no_friction.jld2") r_no_friction_func rz_no_friction_func rθ_no_friction_func rz_no_friction_array rθ_no_friction_array