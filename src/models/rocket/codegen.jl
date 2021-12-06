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

r = residual(rocket, z, θ, κ)
r = Symbolics.simplify.(r)
rz = Symbolics.jacobian(r, z, simplify=true)
rθ = Symbolics.jacobian(r, θ, simplify=true)

# Build function
r_func = build_function(r, z, θ, κ)[2]
rz_func = build_function(rz, z, θ)[2]
rθ_func = build_function(rθ, z, θ)[2]

rz_array = similar(rz, Float64)
rθ_array = similar(rθ, Float64)

path = @get_scratch!("rocket")

@save joinpath(path, "residual.jld2") r_func rz_func rθ_func rz_array rθ_array
@load joinpath(path, "residual.jld2") r_func rz_func rθ_func rz_array rθ_array


# projection
nz = 3 + 1 + 1 + 1 + 1 + 3
nθ = 3 + 1

function residual(z, θ, κ)
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

r = residual(z, θ, κ)
r .= simplify(r)
r_func_proj = Symbolics.build_function(r, z, θ, κ)[2]

rz = Symbolics.jacobian(r, z)
rz = simplify.(rz)
rz_func_proj = Symbolics.build_function(rz, z, θ)[2]
rθ = Symbolics.jacobian(r, θ)
rθ = simplify.(rθ)
rθ_func_proj = Symbolics.build_function(rθ, z, θ)[2]

rz_array_proj = similar(rz, Float64)
rθ_array_proj = similar(rθ, Float64)

@save joinpath(path, "projection.jld2") r_func_proj rz_func_proj rθ_func_proj rz_array_proj rθ_array_proj
@load joinpath(path, "projection.jld2") r_func_proj rz_func_proj rθ_func_proj rz_array_proj rθ_array_proj