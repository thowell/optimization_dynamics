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

ip.z[idx_z.q] .= x1 
ip.θ[idx_θ.q1] .= x1 
ip.θ[idx_θ.u] .= 0.0
ip.θ[idx_θ.h] .= h

interior_point_solve!(ip)
ip.z[idx_z.q]

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

d0 = zeros(nz) 
x2 = f(d0, x1, zeros(nu), zeros(nw))
x3 = f(d0, x2, zeros(nu), zeros(nw))
x4 = f(d0, x3, zeros(nu), zeros(nw))

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


##
m = 3
nz = m + 1 + 1 + 1 + 1 + m
nθ = m + 1

function residual(z, θ, κ)
    u = z[1:m]
    s = z[m .+ (1:1)]
    y = z[m + 1 .+ (1:1)]
    w = z[m + 1 + 1 .+ (1:1)]
    p = z[m + 1 + 1 + 1 .+ (1:1)]
    v = z[m + 1 + 1 + 1 + 1 .+ (1:m)]

    ū = θ[1:m]
    T = θ[m .+ (1:1)]

    idx = [3; 1; 2]

    [
     u - ū - v - [0.0; 0.0; y[1] + p[1]];
     -y[1] - w[1];
     T[1] - u[3] - s[1];
     w .* s .- κ
     p .* u[3] .- κ
     cone_product(v[idx], u[idx]) - [κ[1]; 0.0; 0.0]
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

idx_ineq = collect([3, 4, 6, 7])
idx_soc = [collect([3, 1, 2]), collect([10, 8, 9])]

rz_array_proj = similar(rz, Float64)
rθ_array_proj = similar(rθ, Float64)

@save joinpath(path, "projection.jld2") r_func_proj rz_func_proj rθ_func_proj rz_array_proj rθ_array_proj
@load joinpath(path, "projection.jld2") r_func_proj rz_func_proj rθ_func_proj rz_array_proj rθ_array_proj

idx_opt_proj = IndicesOptimization(
                    nz, 
                    nz, 
                    [collect([3, 4]), collect([7, 6])],
                    [collect([3, 4]), collect([7, 6])],
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
        ϵ_min=0.25,
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
        ϵ_min=0.25,
        diff_sol=true,
        verbose=false))

function cone_projection(z)
    n = length(z)

    z0 = z[1]
    z1 = view(z, 2:n)

    if norm(z1) <= z0
        return z, true
    elseif norm(z1) <= -z0
        return zero(z), false
    else
        a = 0.5 * (1.0 + z0 / norm(z1))
        z_proj = zero(z)
        z_proj[1] = a * norm(z1)
        z_proj[2:end] = a * z1
        return z_proj, false
    end
end

function soc_projection(x)
    proj = cone_projection(x[[3;1;2]])[1]
	eval_ip.z .= [proj[2:3]; proj[1]; 0.1 * ones(7)]
    eval_ip.z[3] += max(1.0, norm(proj[2:3])) * 2.0
    eval_ip.z[10] += 1.0
	eval_ip.θ .= [x; uu[3]]

	status = interior_point_solve!(eval_ip)

    !status && (@warn "projection failure (res norm: $(norm(eval_ip.r, Inf))) \n
		               z = $(eval_ip.z), \n
					   θ = $(eval_ip.θ)")

	return eval_ip.z[1:3]
end

soc_projection([100.0, 0.0, 1.0])

function soc_projection_jacobian(x)
    proj = cone_projection(x[[3;1;2]])[1]
	grad_ip.z .= [proj[2:3]; proj[1]; 0.1 * ones(7)]
    grad_ip.z[3] += max(1.0, norm(proj[2:3])) * 2.0
    grad_ip.z[10] += 1.0
	grad_ip.θ .= [x; uu[3]]

	status = interior_point_solve!(grad_ip)

    !status && (@warn "projection failure (res norm: $(norm(grad_ip.r, Inf))) \n
		               z = $(grad_ip.z), \n
					   θ = $(grad_ip.θ)")

	return grad_ip.δz[1:3, 1:3]
end

soc_projection_jacobian([10.0, 0.0, 1.0])
