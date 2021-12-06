struct RocketInfo1{T,R,RZ,Rθ}
	ip_dyn::InteriorPoint{T,R,RZ,Rθ}
	ip_proj::InteriorPoint{T,R,RZ,Rθ}
	idx_z::IndicesZ
	idx_θ::Indicesθ
	idx3::Vector{Int}
	h::T 
	u_max::T 
	du_dyn_cache::Matrix{T} 
	du_proj_cache::Matrix{T}
end

function RocketInfo(rocket, u_max, h, r_func, rz_func, rθ_func, r_func_proj, rz_func_proj, rθ_func_proj) 
	nx = rocket.nq
	nu = rocket.nu 
	nw = rocket.nw 
	
	nz = nx 
	nθ = nx + nu + 1
	
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
	
	ip_dyn = interior_point(
		z0,
		θ0,
		idx=idx_opt,
		r! = r_func,
		rz! = rz_func,
		rθ! = rθ_func,
		rz=zeros(nz, nz),
		rθ=zeros(nz, nθ),
		opts=solver_opts)
	
	idx_z = RoboDojo.indices_z(rocket)
	idx_θ = RoboDojo.indices_θ(rocket)

	# projection
	nz_proj = 10
	nθ_proj = 4

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

		z0_proj = zeros(nz_proj) 
		θ0_proj = zeros(nθ_proj)

	ip_proj = interior_point(
		z0_proj,
		θ0_proj,
		idx=idx_opt_proj,
		r! = r_func_proj,
		rz! = rz_func_proj,
		rθ! = rθ_func_proj,
		rz=zeros(nz_proj, nz_proj),
		rθ=zeros(nz_proj, nθ_proj),
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

	RocketInfo1(
		ip_dyn, 
		ip_proj, 
		idx_z, 
		idx_θ, 
		collect(1:3),
		h, 
		u_max,
		zeros(nx, nu), 
		zeros(nu, nu)
	)
end

function f_rocket(d, info::RocketInfo1, x, u, w)
    # initialize
    info.ip_dyn.z[info.idx_z.q] .= x
    info.ip_dyn.θ[info.idx_θ.q1] .= x
    info.ip_dyn.θ[info.idx_θ.u] .= u
    info.ip_dyn.θ[info.idx_θ.h] .= info.h
    # solve
    info.ip_dyn.opts.diff_sol = false
    interior_point_solve!(info.ip_dyn)
    # solution 
	sol = @views info.ip_dyn.z[info.idx_z.q]
    d .= sol
    return d
end

# d = zeros(nx) 
# dx = zeros(nx, nx)
# du = zeros(nx, nu)

# x = rand(nx) 
# u = rand(nu) 
# w = rand(nw)

# using BenchmarkTools
# f_rocket(d, info, x, u, w)
# @benchmark f_rocket($d, $info, $x, $u, $w)

# fx_rocket(dx, info, x, u, w)
# @benchmark fx_rocket($dx, $info, $x, $u, $w)

# fu_rocket(du, info, x, u, w)
# @benchmark fu_rocket($du, $info, $x, $u, $w)

function fx_rocket(dx, info::RocketInfo1, x, u, w)
    # initialize
    info.ip_dyn.z[info.idx_z.q] .= x
    info.ip_dyn.θ[info.idx_θ.q1] .= x
    info.ip_dyn.θ[info.idx_θ.u] .= u
    info.ip_dyn.θ[info.idx_θ.h] .= info.h
    # solve
    info.ip_dyn.opts.diff_sol = true
    interior_point_solve!(info.ip_dyn)
    # solution 
	sol = @views info.ip_dyn.δz[info.idx_z.q, info.idx_θ.q1]
    dx .= sol
    return dx
end

function fu_rocket(du, info::RocketInfo1, x, u, w)
    # initialize
    info.ip_dyn.z[info.idx_z.q] .= x
    info.ip_dyn.θ[info.idx_θ.q1] .= x
    info.ip_dyn.θ[info.idx_θ.u] .= u
    info.ip_dyn.θ[info.idx_θ.h] .= info.h
    # solve
    info.ip_dyn.opts.diff_sol = true
    interior_point_solve!(info.ip_dyn)

    # solution 
	sol = @views info.ip_dyn.δz[info.idx_z.q, info.idx_θ.u]
    du .= sol
    return du
end

# info = RocketInfo(rocket, eval(r_func), eval(rz_func), eval(rθ_func), eval(r_func_proj), eval(rz_func_proj), eval(rθ_func_proj))
	

function soc_projection(x, info::RocketInfo1)
	info.ip_proj.z .= 0.1
    info.ip_proj.z[3] += 1.0
    info.ip_proj.z[10] += 1.0
    info.ip_proj.z[7] = 0.0

	info.ip_proj.θ[info.idx3] .= x 
	info.ip_proj.θ[4] = info.u_max

	info.ip_proj.opts.diff_sol = false
	status = interior_point_solve!(info.ip_proj)

    # !status && (@warn "projection failure (res norm: $(norm(info.ip_proj.r, Inf))) \n
	# 	               z = $(info.ip_proj.z), \n
	# 				   θ = $(info.ip_proj.θ)")

	sol = @views info.ip_proj.z[info.idx3]
	return sol
end
# soc_projection(u, info)
# @benchmark soc_projection($u, $info)

function soc_projection_gradient(x, info::RocketInfo1)
	info.ip_proj.z .= 0.1
    info.ip_proj.z[3] += 1.0
    info.ip_proj.z[10] += 1.0
    info.ip_proj.z[7] = 0.0

	# info.ip_proj.θ .= [x; uu]
	info.ip_proj.θ[info.idx3] .= x 
	info.ip_proj.θ[4] = info.u_max

	info.ip_proj.opts.diff_sol = true
	status = interior_point_solve!(info.ip_proj)

    # !status && (@warn "projection failure (res norm: $(norm(info.ip_proj.r, Inf))) \n
	# 	               z = $(info.ip_proj.z), \n
	# 				   θ = $(info.ip_proj.θ)")

	sol = @views info.ip_proj.δz[info.idx3, info.idx3]

	return sol
end

# soc_projection_gradient(u, info)
# @benchmark soc_projection_gradient($u, $info)

function f_rocket_proj(d, info::RocketInfo1, x, u, w)
    # initialize
    info.ip_dyn.z[info.idx_z.q] .= x
    info.ip_dyn.θ[info.idx_θ.q1] .= x
    info.ip_dyn.θ[info.idx_θ.u] .= soc_projection(u, info)
    info.ip_dyn.θ[info.idx_θ.h] .= info.h
    # solve
    info.ip_dyn.opts.diff_sol = false
    interior_point_solve!(info.ip_dyn)
    # solution 
    sol = @views info.ip_dyn.z[info.idx_z.q]
	d .= sol
    return d
end

# f_rocket_proj(d, info, x, u, w)
# @benchmark f_rocket_proj($d, $info, $x, $u, $w)

# fx_rocket_proj(dx, info, x, u, w)
# @benchmark fx_rocket_proj($dx, $info, $x, $u, $w)

# fu_rocket_proj(du, info, x, u, w)
# @benchmark fu_rocket_proj($du, $info, $x, $u, $w)

function fx_rocket_proj(dx, info::RocketInfo1, x, u, w)
    # initialize
    info.ip_dyn.z[info.idx_z.q] .= x
    info.ip_dyn.θ[info.idx_θ.q1] .= x
    info.ip_dyn.θ[info.idx_θ.u] .= soc_projection(u, info)
    info.ip_dyn.θ[info.idx_θ.h] .= info.h
    # solve
    info.ip_dyn.opts.diff_sol = true
    interior_point_solve!(info.ip_dyn)
    # solution 
    sol = @views info.ip_dyn.δz[info.idx_z.q, info.idx_θ.q1]
    dx .= sol
    return dx 
end

function fu_rocket_proj(du, info::RocketInfo1, x, u, w)
    # initialize
    info.ip_dyn.z[info.idx_z.q] .= x
    info.ip_dyn.θ[info.idx_θ.q1] .= x
    info.ip_dyn.θ[info.idx_θ.u] .= soc_projection(u, info)
    info.ip_dyn.θ[info.idx_θ.h] .= info.h
    # solve
    info.ip_dyn.opts.diff_sol = true
    interior_point_solve!(info.ip_dyn)
    # solution 
    info.du_dyn_cache .= @views info.ip_dyn.δz[info.idx_z.q, info.idx_θ.u]
	info.du_proj_cache .= soc_projection_gradient(u, info)
    # du .= sol_dyn * sol_proj
	mul!(du, info.du_dyn_cache, info.du_proj_cache)
    return du 
end