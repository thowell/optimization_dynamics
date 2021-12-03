struct ImplicitDynamics{T,R,RZ,Rθ,M<:RoboDojo.Model{T},P<:RoboDojo.Policy{T},D<:RoboDojo.Disturbances{T},I} <: Model{T}
    n::Int
    m::Int
    d::Int
	eval_sim::Simulator{T,R,RZ,Rθ,M,P,D}
	grad_sim::Simulator{T,R,RZ,Rθ,M,P,D}
	q1::Vector{T} 
	q2::Vector{T} 
	v1::Vector{T}
	idx_q1::Vector{Int} 
	idx_q2::Vector{Int}
	idx_u1::Vector{Int}
	info::I
end

function get_simulator(model, h, r_func, rz_func, rθ_func; 
	T=1, r_tol=1.0e-6, κ_eval_tol=1.0e-4, nc=model.nc, nb=model.nc, diff_sol=true)

	sim = Simulator(model, T; 
        h=h, 
        residual=r_func, 
        jacobian_z=rz_func, 
        jacobian_θ=rθ_func,
        diff_sol=diff_sol,
        solver_opts=InteriorPointOptions(
            undercut=Inf,
            γ_reg=0.1,
            r_tol=r_tol,
            κ_tol=κ_eval_tol,  
            max_ls=25,
            ϵ_min=0.25,
            diff_sol=diff_sol,
            verbose=false))  

    # set trajectory sizes
	sim.traj.γ .= [zeros(nc) for t = 1:T] 
	sim.traj.b .= [zeros(nb) for t = 1:T] 

    sim.grad.∂γ1∂q1 .= [zeros(nc, model.nq) for t = 1:T] 
	sim.grad.∂γ1∂q2 .= [zeros(nc, model.nq) for t = 1:T]
	sim.grad.∂γ1∂u1 .= [zeros(nc, model.nu) for t = 1:T]
	sim.grad.∂b1∂q1 .= [zeros(nb, model.nq) for t = 1:T] 
	sim.grad.∂b1∂q2 .= [zeros(nb, model.nq) for t = 1:T]
	sim.grad.∂b1∂u1 .= [zeros(nb, model.nu) for t = 1:T]
	
    return sim
end

function ImplicitDynamics(model, h, r_func, rz_func, rθ_func; 
	T=1, r_tol=1.0e-6, κ_eval_tol=1.0e-6, κ_grad_tol=1.0e-6, 
	no_impact=false, no_friction=false, 
	n=(2 * model.nq), m=model.nu, d=model.nw, nc=model.nc, nb=model.nc,
	info=nothing) 

	# set trajectory sizes
	no_impact && (nc = 0) 
	no_friction && (nb = 0) 

	eval_sim = get_simulator(model, h, r_func, rz_func, rθ_func; 
			T=T, r_tol=r_tol, κ_eval_tol=κ_eval_tol, nc=nc, nb=nb, diff_sol=false)

	grad_sim = get_simulator(model, h, r_func, rz_func, rθ_func; 
			T=T, r_tol=r_tol, κ_eval_tol=κ_grad_tol, nc=nc, nb=nb, diff_sol=true)

	q1 = zeros(model.nq) 
	q2 = zeros(model.nq) 
	v1 = zeros(model.nq) 

	idx_q1 = collect(1:model.nq) 
	idx_q2 = collect(model.nq .+ (1:model.nq)) 
	idx_u1 = collect(1:model.nu)
	
	ImplicitDynamics(n, m, d, 
		eval_sim, grad_sim, 
		q1, q2, v1,
		idx_q1, idx_q2, idx_u1, info)
end

function f(d, model::ImplicitDynamics, x, u, w)
	q1 = @views x[model.idx_q1]
	q2 = @views x[model.idx_q2]
	model.v1 .= q2 
	model.v1 .-= q1 
	model.v1 ./= model.eval_sim.h

	q3 = RoboDojo.step!(model.eval_sim, q2, model.v1, u, 1)

	d[model.idx_q1] .= q2 
	d[model.idx_q2] .= q3

	return d
end

# using BenchmarkTools
# using InteractiveUtils
# x = x̄[1]
# u = ū[1]
# w_ = w[1]
# d = zeros(nx)
# dx = zeros(nx, nx)
# du = zeros(nx, nu)

# f(d, im_dyn, x, u, w_)
# @benchmark f($d, $im_dyn, $x, $u, $w_)
# @code_warntype f(d, im_dyn, x, u, w_)

function fx(dx, model::ImplicitDynamics, x, u, w)
	q1 = @views x[model.idx_q1]
	q2 = @views x[model.idx_q2]
	model.v1 .= q2 
	model.v1 .-= q1 
	model.v1 ./= model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, model.v1, u, 1)

	nq = model.grad_sim.model.nq
	for i = 1:nq
		dx[model.idx_q1[i], model.idx_q2[i]] = 1.0
	end

	dx[model.idx_q2, model.idx_q1] .= model.grad_sim.grad.∂q3∂q1[1]
	dx[model.idx_q2, model.idx_q2] .= model.grad_sim.grad.∂q3∂q2[1]

	return dx
end

# fx(dx, im_dyn, x, u, w_)
# @benchmark fx($dx, $im_dyn, $x, $u, $w_)
# @code_warntype fx(dx, im_dyn, x, u, w_)


function fu(du, model::ImplicitDynamics, x, u, w)
	q1 = @views x[model.idx_q1]
	q2 = @views x[model.idx_q2]
	model.v1 .= q2 
	model.v1 .-= q1 
	model.v1 ./= model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, model.v1, u, 1)

	du[model.idx_q2, :] = model.grad_sim.grad.∂q3∂u1[1]

	return du
end

# fu(du, im_dyn, x, u, w_)
# @benchmark fu($du, $im_dyn, $x, $u, $w_)
# @code_warntype fu(du, im_dyn, x, u, w_)


function state_to_configuration(x::Vector{Vector{T}}) where T 
	H = length(x) 
	n = length(x[1]) 
	nq = convert(Int, floor(length(x[1]) / 2))
	q = Vector{T}[] 

	for t = 1:H 
		if t == 1 
			push!(q, x[t][1:nq]) 
		end
		push!(q, x[t][nq .+ (1:nq)])
	end
	
	return q 
end