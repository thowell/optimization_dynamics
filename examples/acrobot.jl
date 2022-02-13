using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR
using LinearAlgebra
using Random

# ## visualization 
vis = Visualizer() 
render(vis)

# ## mode
MODE = :impact 
MODE = :nominal

# ## state-space model
h = 0.05
T = 101
κ_grad = 1.0e-3 # gradient smoothness 

if MODE == :impact 
	im_dyn = ImplicitDynamics(acrobot_impact, h, eval(r_acrobot_impact_func), eval(rz_acrobot_impact_func), eval(rθ_acrobot_impact_func); 
		r_tol=1.0e-8, κ_eval_tol=1.0e-4, 
		κ_grad_tol=κ_grad, 
		no_friction=true) 
else
	im_dyn = ImplicitDynamics(acrobot_nominal, h, eval(r_acrobot_nominal_func), eval(rz_acrobot_nominal_func), eval(rθ_acrobot_nominal_func); 
	    r_tol=1.0e-8, κ_eval_tol=1.0, κ_grad_tol=1.0, no_friction=true) 
end

nx = 2 * acrobot_impact.nq
nu = acrobot_impact.nu 

# ## iLQR model
ilqr_dyn = iLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
					(dx, x, u, w) -> fx(dx, im_dyn, x, u, w), 
					(du, x, u, w) -> fu(du, im_dyn, x, u, w), 
					nx, nx, nu)  
model = [ilqr_dyn for t = 1:T-1]

# ## initial and goal states
q1 = [0.0; 0.0]
q2 = [0.0; 0.0]
qT = [π; 0.0]
q_ref = qT

x1 = [q1; q2]
xT = [qT; qT]

# ## objective
function objt(x, u, w)
	J = 0.0 

	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * 0.1 * transpose(v1) * v1 
	J += 0.5 * transpose(u) * u

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * 0.1 * transpose(v1) * v1

	return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
function terminal_con(x, u, w) 
    [
    x - xT; # goal 
    ]
end

cont = iLQR.Constraint()
conT = iLQR.Constraint(terminal_con, nx, 0)
cons = [[cont for t = 1:T-1]..., conT]

# ## rollout
Random.seed!(1)
ū = [1.0e-3 * randn(nu) for t = 1:T-1]
x̄ = iLQR.rollout(model, x1, ū)
q̄ = state_to_configuration(x̄)
visualize!(vis, acrobot_impact, q̄, Δt=h)

# ## solver
solver = iLQR.solver(model, obj, cons, 
	opts=iLQR.Options(linesearch = :armijo,
		α_min=1.0e-5,
		obj_tol=1.0e-5,
		grad_tol=1.0e-5,
		max_iter=50,
		max_al_iter=20,
		con_tol=0.001,
		ρ_init=1.0, 
		ρ_scale=10.0,
		verbose=true))
iLQR.initialize_controls!(solver, ū)
iLQR.initialize_states!(solver, x̄)

# ## solve
iLQR.reset!(solver.s_data)
iLQR.solve!(solver)

@show solver.s_data.iter[1]
@show iLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
@show norm(terminal_con(solver.m_data.x[T], zeros(0), zeros(0)), Inf)
@show solver.s_data.obj[1] # augmented Lagrangian cost
	
# ## solution
x_sol, u_sol = iLQR.get_trajectory(solver)
q_sol = state_to_configuration(x_sol)
visualize!(vis, acrobot_impact, q_sol, Δt=h)

# ## benchmark
using BenchmarkTools
solver.options.verbose = false
@benchmark iLQR.solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))

