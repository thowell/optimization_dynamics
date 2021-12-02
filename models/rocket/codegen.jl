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

solver_opts=InteriorPointOptions(
            r_tol=1e-8,
            κ_tol=1.0,  
            max_ls=25,
            ϵ_min=0.25,
            diff_sol=true,
            verbose=false)

z0 = zeros(nx)
θ0 = zeros(nθ)

idx_opt = RoboDojo.indices_optimization(rocket) 

ip = interior_point(
    z0,
    θ0,
    idx=idx_opt,
    r! = eval(r_func),
    rz! = eval(rz_func),
    rθ! = eval(rθ_func),
    rz=zeros(nz, nz),
    rθ=zeros(nz, nθ),
    opts=solver_opts)

idx_z = RoboDojo.indices_z(rocket)
idx_θ = RoboDojo.indices_θ(rocket)

function f_rocket(d, x, u, w)
    # initialize
    ip.z[idx_z.q] .= copy(x)
    ip.θ[idx_θ.q1] .= copy(x)
    ip.θ[idx_θ.u] .= u
    ip.θ[idx_θ.h] .= h
    # solve
    ip.opts.diff_sol = false
    interior_point_solve!(ip)
    # solution 
    d .= copy(ip.z[idx_z.q])
    return ip.z[idx_z.q] 
end

function fx_rocket(dx, x, u, w)
    # initialize
    ip.z[idx_z.q] .= x
    ip.θ[idx_θ.q1] .= x
    ip.θ[idx_θ.u] .= u
    ip.θ[idx_θ.h] .= h
    # solve
    ip.opts.diff_sol = true
    interior_point_solve!(ip)
    # solution 
    dx .= ip.δz[idx_z.q, idx_θ.q1]
    return ip.δz[idx_z.q, idx_θ.q1] 
end

function fu_rocket(du, x, u, w)
    # initialize
    ip.z[idx_z.q] .= x
    ip.θ[idx_θ.q1] .= x
    ip.θ[idx_θ.u] .= u
    ip.θ[idx_θ.h] .= h
    # solve
    ip.opts.diff_sol = true
    interior_point_solve!(ip)
    # solution 
    du .= ip.δz[idx_z.q, idx_θ.u]
    return ip.δz[idx_z.q, idx_θ.u] 
end

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

idx_opt_proj = IndicesOptimization(
                    10, 
                    10, 
                    [[5, 3], [6, 4]],
                    [[5, 3], [6, 4]],
                    [[collect([3, 1, 2]), collect([10, 8, 9])]],
                    [[collect([3, 1, 2]), collect([10, 8, 9])]],
                    collect(1:5),
                    collect(6:7),
                    collect(8:10),
                    [collect(8:10)],
                    collect(6:10))

z0 = zeros(nz) 
θ0 = zeros(nθ)

eval_ip = interior_point(
    z0,
    θ0,
    idx=idx_opt_proj,
    r! = eval(r_func_proj),
    rz! = eval(rz_func_proj),
    rθ! = eval(rθ_func_proj),
    rz=zeros(nz, nz),
    rθ=zeros(nz, nθ),
    opts=InteriorPointOptions(
        r_tol=1e-8,
        κ_tol=1.0e-4,  
        max_ls=25,
        ϵ_min=0.0,
        undercut=Inf,
        γ_reg=0.0, 
        κ_reg=0.0,
        diff_sol=false,
        verbose=false))

grad_ip = interior_point(
    z0,
    θ0,
    idx=idx_opt_proj,
    r! = eval(r_func_proj),
    rz! = eval(rz_func_proj),
    rθ! = eval(rθ_func_proj),
    rz=zeros(nz, nz),
    rθ=zeros(nz, nθ),
    opts=InteriorPointOptions(
        r_tol=1e-8,
        κ_tol=1.0e-3,  
        max_ls=25,
        ϵ_min=0.0,
        undercut=Inf,
        γ_reg=0.0, 
        κ_reg=0.0,
        diff_sol=true,
        verbose=false))

function soc_projection(x, uu)
	eval_ip.z .= 0.1
    eval_ip.z[3] += 1.0
    eval_ip.z[10] += 1.0
    eval_ip.z[7] = 0.0

	eval_ip.θ .= [x; uu]

	status = interior_point_solve!(eval_ip)

    !status && (@warn "projection failure (res norm: $(norm(eval_ip.r, Inf))) \n
		               z = $(eval_ip.z), \n
					   θ = $(eval_ip.θ)")

	return eval_ip.z[1:3]
end

# soc_projection([0.0, 0.0, 10.0], 1.0)

function soc_projection_gradient(x, uu)
	grad_ip.z .= 0.1
    grad_ip.z[3] += 1.0
    grad_ip.z[10] += 1.0
    grad_ip.z[7] = 0.0

	grad_ip.θ .= [x; uu]

	status = interior_point_solve!(grad_ip)

    !status && (@warn "projection failure (res norm: $(norm(grad_ip.r, Inf))) \n
		               z = $(grad_ip.z), \n
					   θ = $(grad_ip.θ)")

	return grad_ip.δz[1:3, 1:3]
end

# soc_projection_gradient([10.0, 0.0, 0.0], 1.0)

function f_rocket_proj(d, x, u, w)
    # thrust projection
    u_proj = soc_projection(u, uu[3])

    # initialize
    ip.z[idx_z.q] .= copy(x)
    ip.θ[idx_θ.q1] .= copy(x)
    ip.θ[idx_θ.u] .= u_proj
    ip.θ[idx_θ.h] .= h
    # solve
    ip.opts.diff_sol = false
    interior_point_solve!(ip)
    # solution 
    d .= ip.z[idx_z.q]
    return d
end

function fx_rocket_proj(dx, x, u, w)
    # thrust projection
    u_proj = soc_projection(u, uu[3])

    # initialize
    ip.z[idx_z.q] .= x
    ip.θ[idx_θ.q1] .= x
    ip.θ[idx_θ.u] .= u_proj
    ip.θ[idx_θ.h] .= h
    # solve
    ip.opts.diff_sol = true
    interior_point_solve!(ip)
    # solution 
    dx .= ip.δz[idx_z.q, idx_θ.q1]
    return dx 
end

function fu_rocket_proj(du, x, u, w)
    # thrust projection
    u_proj = soc_projection(u, uu[3]) 
    δu_proj = soc_projection_gradient(u, uu[3] )
    # initialize
    ip.z[idx_z.q] .= x
    ip.θ[idx_θ.q1] .= x
    ip.θ[idx_θ.u] .= u_proj
    ip.θ[idx_θ.h] .= h
    # solve
    ip.opts.diff_sol = true
    interior_point_solve!(ip)
    # solution 
    du .= ip.δz[idx_z.q, idx_θ.u] * δu_proj
    return du 
end