using OptimizationDynamics
using IterativeLQR
using Random

# ## visualization 
vis = Visualizer() 
render(vis)

# ## mode
MODE = :impact 
MODE = :no_impact

# ## state-space model
h = 0.05
T = 101
κ_grad = 1.0e-3 # gradient smoothness 
include("../src/models/acrobot/simulator_impact.jl")

if MODE == :impact 
	include("../src/models/acrobot/simulator_impact.jl")
	@load joinpath(@get_scratch!("acrobot"), "impact.jld2") r_func rz_func rθ_func rz_array rθ_array
	im_dyn = ImplicitDynamics(acrobot, h, eval(r_func), eval(rz_func), eval(rθ_func); 
		r_tol=1.0e-8, κ_eval_tol=1.0e-4, 
		κ_grad_tol=κ_grad, 
		no_friction=true) 
else
	include("../src/models/acrobot/simulator_no_impact.jl")
	@load joinpath(@get_scratch!("acrobot"), "no_impact.jld2") r_no_impact_func rz_no_impact_func rθ_no_impact_func rz_no_impact_array rθ_no_impact_array
	im_dyn = ImplicitDynamics(acrobot_no_impact, h, eval(r_no_impact_func), eval(rz_no_impact_func), eval(rθ_no_impact_func); 
	    r_tol=1.0e-8, κ_eval_tol=1.0, κ_grad_tol=1.0, no_impact=true, no_friction=true) 
end

nx = 2 * acrobot.nq
nu = acrobot.nu 
nw = acrobot.nw

# ## iLQR model
ilqr_dyn = IterativeLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
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

	q1 = x[1:acrobot.nq] 
	q2 = x[acrobot.nq .+ (1:acrobot.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * 0.1 * transpose(v1) * v1 
	J += 0.5 * transpose(u) * u

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	q1 = x[1:acrobot.nq] 
	q2 = x[acrobot.nq .+ (1:acrobot.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * 0.1 * transpose(v1) * v1

	return J
end

ct = IterativeLQR.Cost(objt, nx, nu, nw)
cT = IterativeLQR.Cost(objT, nx, 0, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
function terminal_con(x, u, w) 
    [
    x - xT; # goal 
    ]
end

cont = Constraint()
conT = Constraint(terminal_con, nx, 0)
cons = [[cont for t = 1:T-1]..., conT]

# ## rollout
Random.seed!(1)
ū = [1.0e-3 * randn(nu) for t = 1:T-1]
w = [zeros(nw) for t = 1:T-1]
x̄ = rollout(model, x1, ū)
q̄ = state_to_configuration(x̄)
visualize!(vis, acrobot, q̄, Δt=h)

# ## problem 
prob = problem_data(model, obj, cons)
initialize_controls!(prob, ū)
initialize_states!(prob, x̄)

# ## solve
IterativeLQR.reset!(prob.s_data)
IterativeLQR.solve!(prob, 
	linesearch = :armijo,
    α_min=1.0e-5,
    obj_tol=1.0e-5,
    grad_tol=1.0e-5,
    max_iter=50,
    max_al_iter=20,
    con_tol=0.001,
    ρ_init=1.0, 
    ρ_scale=10.0,
	verbose=true)

@show prob.s_data.iter[1]
@show IterativeLQR.eval_obj(prob.m_data.obj.costs, prob.m_data.x, prob.m_data.u, prob.m_data.w)
@show norm(terminal_con(prob.m_data.x[T], zeros(0), zeros(0)), Inf)
@show prob.s_data.obj[1] # augmented Lagrangian cost
	
# ## solution
x_sol, u_sol = get_trajectory(prob)
q_sol = state_to_configuration(x_sol)
visualize!(vis, acrobot, q_sol, Δt=h)

# ## benchmark
@benchmark IterativeLQR.solve!($prob, x̄, ū,
	linesearch = :armijo,
    α_min=1.0e-5,
    obj_tol=1.0e-5,
    grad_tol=1.0e-5,
    max_iter=50,
    max_al_iter=20,
    con_tol=0.001,
    ρ_init=1.0, 
    ρ_scale=10.0,
	verbose=false) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))

