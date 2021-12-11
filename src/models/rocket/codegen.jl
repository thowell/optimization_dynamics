path = @get_scratch!("rocket")

# rocket
nx = rocket.nq
nu = rocket.nu 
nw = rocket.nw 

nz = nx 
nθ = nx + nu + 1

@variables z[1:nz] θ[1:nθ] κ[1:1] 

# Residual
function residual(rocket, z, θ, κ)
    y = z[1:nx] 
    x = θ[1:nx] 
    u = θ[nx .+ (1:nu)] 
    h = θ[nx + nu .+ (1:1)] 
    w = zeros(rocket.nw) 

    return y - (x + h[1] * f(rocket, 0.5 * (x + y), u, w)) 
end

r_rocket = residual(rocket, z, θ, κ)
r_rocket = Symbolics.simplify.(r_rocket)
rz_rocket = Symbolics.jacobian(r_rocket, z, simplify=true)
rθ_rocket = Symbolics.jacobian(r_rocket, θ, simplify=true)

# Build function
r_rocket_func = build_function(r_rocket, z, θ, κ)[2]
rz_rocket_func = build_function(rz_rocket, z, θ)[2]
rθ_rocket_func = build_function(rθ_rocket, z, θ)[2]

rz_rocket_array = similar(rz_rocket, Float64)
rθ_rocket_array = similar(rθ_rocket, Float64)

@save joinpath(path, "residual.jld2") r_rocket_func rz_rocket_func rθ_rocket_func rz_rocket_array rθ_rocket_array
@load joinpath(path, "residual.jld2") r_rocket_func rz_rocket_func rθ_rocket_func rz_rocket_array rθ_rocket_array


# projection
nz = 3 + 1 + 1 + 1 + 1 + 3
nθ = 3 + 1

function residual_projection(z, θ, κ)
    u = z[1:3]
    p = z[4:4]
    s = z[5:5]
    w = z[6:6] 
    y = z[7:7]
    v = z[8:10]
   
    ū = θ[1:3]
    uu = θ[3 .+ (1:1)]
    idx = [3; 1; 2]
    [
     u - ū - v - [0.0; 0.0; y[1] + p[1]];
     uu[1] - u[3] - s[1];
     -y[1] - w[1];
     w[1] * s[1] - κ[1];
     p[1] * u[3] - κ[1];
     cone_product(u[idx], v[idx]) - [κ[1]; 0.0; 0.0]
    ]
end

@variables z[1:nz], θ[1:nθ], κ[1:1]

r_proj = residual_projection(z, θ, κ)
r_proj .= simplify(r_proj)
r_proj_func = Symbolics.build_function(r_proj, z, θ, κ)[2]

rz_proj = Symbolics.jacobian(r_proj, z)
rz_proj = simplify.(rz_proj)
rz_proj_func = Symbolics.build_function(rz_proj, z, θ)[2]
rθ_proj = Symbolics.jacobian(r_proj, θ)
rθ_proj = simplify.(rθ_proj)
rθ_proj_func = Symbolics.build_function(rθ_proj, z, θ)[2]

rz_proj_array = similar(rz_proj, Float64)
rθ_proj_array = similar(rθ_proj, Float64)

@save joinpath(path, "projection.jld2") r_proj_func rz_proj_func rθ_proj_func rz_proj_array rθ_proj_array
@load joinpath(path, "projection.jld2") r_proj_func rz_proj_func rθ_proj_func rz_proj_array rθ_proj_array