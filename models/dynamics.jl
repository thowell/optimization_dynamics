struct ImplicitDynamics{T} <: Model{T}
    n::Int
    m::Int
    d::Int
	eval_sim::Simulator{T} 
	grad_sim::Simulator{T}
	f_tmp::Vector{T} 
	fx_tmp::Matrix{T} 
	fu_tmp::Matrix{T}
	q1_idx::Vector{Int} 
	q2_idx::Vector{Int}
	u1_idx::Vector{Int}
end

function ImplicitDynamics(model, h, r_func, rz_func, rθ_func; 
	T=1, r_tol=1.0e-6, κ_eval_tol=1.0e-6, κ_grad_tol=1.0e-6, 
	no_impact=false, no_friction=false, 
	nn=2 * model.nq, n=2 * model.nq, m=model.nu, d=model.nw, nc=model.nc, nb=model.nc) 

	eval_sim = Simulator(model, T; 
        h=h, 
        residual=r_func, 
        jacobian_z=rz_func, 
        jacobian_θ=rθ_func,
        diff_sol=false,
        solver_opts=InteriorPointOptions(
            undercut=Inf,
            γ_reg=0.1,
            r_tol=r_tol,
            κ_tol=κ_eval_tol,  
            max_ls=25,
            ϵ_min=0.25,
            diff_sol=false,
            verbose=false))  

	grad_sim = Simulator(model, T; 
		h=h, 
		residual=r_func, 
		jacobian_z=rz_func, 
		jacobian_θ=rθ_func,
		diff_sol=true,
		solver_opts=InteriorPointOptions(
			undercut=Inf,
			γ_reg=0.1,
			r_tol=r_tol,
			κ_tol=κ_grad_tol,  
			max_ls=25,
			ϵ_min=0.25,
			diff_sol=true,
			verbose=false))  

	# set trajectory sizes
	eval_sim.traj.γ .= [zeros(nc) for t = 1:T] 
	grad_sim.traj.γ .= [zeros(nc) for t = 1:T] 

	eval_sim.grad.∂γ1∂q1 .= [zeros(nc, model.nq) for t = 1:T] 
	eval_sim.grad.∂γ1∂q2 .= [zeros(nc, model.nq) for t = 1:T]
	eval_sim.grad.∂γ1∂u1 .= [zeros(nc, model.nu) for t = 1:T]
	grad_sim.grad.∂γ1∂q1 .= [zeros(nc, model.nq) for t = 1:T] 
	grad_sim.grad.∂γ1∂q2 .= [zeros(nc, model.nq) for t = 1:T]
	grad_sim.grad.∂γ1∂u1 .= [zeros(nc, model.nu) for t = 1:T]

	eval_sim.traj.b .= [zeros(nb) for t = 1:T] 
	grad_sim.traj.b .= [zeros(nb) for t = 1:T]

	eval_sim.grad.∂b1∂q1 .= [zeros(nb, model.nq) for t = 1:T] 
	eval_sim.grad.∂b1∂q2 .= [zeros(nb, model.nq) for t = 1:T]
	eval_sim.grad.∂b1∂u1 .= [zeros(nb, model.nu) for t = 1:T]
	grad_sim.grad.∂b1∂q1 .= [zeros(nb, model.nq) for t = 1:T] 
	grad_sim.grad.∂b1∂q2 .= [zeros(nb, model.nq) for t = 1:T]
	grad_sim.grad.∂b1∂u1 .= [zeros(nb, model.nu) for t = 1:T]

	if no_impact 
		eval_sim.traj.γ .= [zeros(0) for t = 1:T]
		grad_sim.traj.γ .= [zeros(0) for t = 1:T]

		eval_sim.grad.∂γ1∂q1 .= [zeros(0, model.nq) for t = 1:T] 
    	eval_sim.grad.∂γ1∂q2 .= [zeros(0, model.nq) for t = 1:T]
    	eval_sim.grad.∂γ1∂u1 .= [zeros(0, model.nu) for t = 1:T]
		grad_sim.grad.∂γ1∂q1 .= [zeros(0, model.nq) for t = 1:T] 
    	grad_sim.grad.∂γ1∂q2 .= [zeros(0, model.nq) for t = 1:T]
    	grad_sim.grad.∂γ1∂u1 .= [zeros(0, model.nu) for t = 1:T]
	end

	if no_friction 
		eval_sim.traj.b .= [zeros(0)]
		grad_sim.traj.b .= [zeros(0)]

		eval_sim.grad.∂b1∂q1 .= [zeros(0, model.nq)] 
    	eval_sim.grad.∂b1∂q2 .= [zeros(0, model.nq)]
    	eval_sim.grad.∂b1∂u1 .= [zeros(0, model.nu)]
		grad_sim.grad.∂b1∂q1 .= [zeros(0, model.nq)] 
    	grad_sim.grad.∂b1∂q2 .= [zeros(0, model.nq)]
    	grad_sim.grad.∂b1∂u1 .= [zeros(0, model.nu)]
	end

	f_tmp = zeros(nn) 
	fx_tmp = zeros(nn, n) 
	fu_tmp = zeros(nn, m) 

	idx_q1 = collect(1:model.nq) 
	idx_q2 = collect(model.nq .+ (1:model.nq)) 
	idx_u1 = collect(1:model.nu)
	
	ImplicitDynamics(n, m, d, 
		eval_sim, grad_sim, 
		f_tmp, fx_tmp, fu_tmp,
		idx_q1, idx_q2, idx_u1)
end

function f(d, model::ImplicitDynamics, x, u, w)
	nq = model.eval_sim.model.nq

	# q1 = view(x, 1:nq)
	# q2 = view(x, nq .+ (1:nq))
	q1 = x[1:nq] 
	q2 = x[nq .+ (1:nq)]
	v1 = (q2 - q1) ./ model.eval_sim.h

	q3 = RoboDojo.step!(model.eval_sim, q2, v1, u, 1)

	d[1:nq] .= q2 
	d[nq .+ (1:nq)] .= q3

	return d
end

function fx(dx, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq

	# q1 = view(x, 1:nq)
	# q2 = view(x, nq .+ (1:nq))
	q1 = x[1:nq] 
	q2 = x[nq .+ (1:nq)]
	v1 = (q2 - q1) ./ model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, v1, u, 1)

	∂q3∂q1 = model.grad_sim.grad.∂q3∂q1[1]
	∂q3∂q2 = model.grad_sim.grad.∂q3∂q2[1]

	dx .= [zeros(nq, nq) I; ∂q3∂q1 ∂q3∂q2]

	return dx
end

function fu(du, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq

	# q1 = view(x, 1:nq)
	# q2 = view(x, nq .+ (1:nq))
	q1 = x[1:nq] 
	q2 = x[nq .+ (1:nq)]
	v1 = (q2 - q1) ./ model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, v1, u, 1)

	∂q3∂u1 = model.grad_sim.grad.∂q3∂u1[1]
	
	du .= [zeros(nq, model.m); ∂q3∂u1]

	return du
end

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