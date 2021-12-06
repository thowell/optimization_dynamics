using OptimizationDynamics
using IterativeLQR
using Random
Random.seed!(1)

# ## visualization 
include("../models/planar_push/visuals.jl")
include("../models/visualize.jl")
vis = Visualizer() 
render(vis)

# ## planar push model 
include("../models/planar_push/model.jl")
include("../models/planar_push/simulator.jl")

# ## mode
MODE = :translate
MODE = :rotate 

# ## gradient bundle
GB = true 
include("../src/gradient_bundle.jl")

# ## state-space model
h = 0.1
T = 26
path = @get_scratch!("planarpush")
@load joinpath(path, "residual.jld2") r_func rz_func rθ_func rz_array rθ_array

im_dyn = ImplicitDynamics(planarpush, h, eval(r_func), eval(rz_func), eval(rθ_func); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-2, nc=1, nb=9, info=(GB ? GradientBundle(planarpush, N=50, ϵ=1.0e-4) : nothing)) 

nx = 2 * planarpush.nq
nu = planarpush.nu 
nw = planarpush.nw

# ## iLQR model
ilqr_dyn = IterativeLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
	(dx, x, u, w) -> GB ? fx_gb(dx, im_dyn, x, u, w) : fx(dx, im_dyn, x, u, w), 
	(du, x, u, w) -> GB ? fu_gb(du, im_dyn, x, u, w) : fu(du, im_dyn, x, u, w), 
	nx, nx, nu) 

	model = [ilqr_dyn for t = 1:T-1]

# ## initial conditions and goal
if MODE == :translate 
	q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
	q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
	x_goal = 1.0
	y_goal = 0.0
	θ_goal = 0.0 * π
	qT = [x_goal, y_goal, θ_goal, x_goal - r_dim, y_goal - r_dim]
	xT = [qT; qT]
elseif MODE == :rotate 
	q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
	q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
	x1 = [q1; q1]
	x_goal = 0.5
	y_goal = 0.5
	θ_goal = 0.5 * π
	qT = [x_goal, y_goal, θ_goal, x_goal-r_dim, y_goal-r_dim]
	xT = [qT; qT]
end

# ## objective
function objt(x, u, w)
	J = 0.0 

	q1 = x[1:nq] 
	q2 = x[nq .+ (1:nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) * v1 
	J += 0.5 * transpose(x - xT) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1]) * (x - xT) 
	J += 0.5 * (MODE == :translate ? 1.0e-1 : 1.0e-2) * transpose(u) * u

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	q1 = x[1:nq] 
	q2 = x[nq .+ (1:nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) * v1 
	J += 0.5 * transpose(x - xT) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1]) * (x - xT) 

	return J
end

ct = IterativeLQR.Cost(objt, nx, nu, nw)
cT = IterativeLQR.Cost(objT, nx, 0, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
ul = [-5.0; -5.0]
uu = [5.0; 5.0]

function stage_con(x, u, w) 
    [
     ul - u; # control limit (lower)
     u - uu; # control limit (upper)
    ]
end 

function terminal_con(x, u, w) 
    [
     (x - xT)[collect([(1:3)..., (6:8)...])]; # goal 
    ]
end

cont = Constraint(stage_con, nx, nu, idx_ineq=collect(1:(2 * nu)))
conT = Constraint(terminal_con, nx, 0)
cons = [[cont for t = 1:T-1]..., conT]

# ## rollout
x1 = [q0; q1]
ū = MODE == :translate ? [t < 5 ? [1.0; 0.0] : [0.0; 0.0] for t = 1:T-1] : [t < 5 ? [1.0; 0.0] : t < 10 ? [0.5; 0.0] : [0.0; 0.0] for t = 1:T-1]
w = [zeros(nw) for t = 1:T-1]
x̄ = rollout(model, x1, ū)
q̄ = state_to_configuration(x̄)
visualize!(vis, planarpush, q̄, Δt=h)

# ## problem 
prob = problem_data(model, obj, cons)
initialize_controls!(prob, ū)
initialize_states!(prob, x̄)

# ## solve
IterativeLQR.reset!(prob.s_data)
IterativeLQR.solve!(prob, 
	linesearch = :armijo,
	α_min=1.0e-5,
	obj_tol=1.0e-3,
	grad_tol=1.0e-3,
	max_iter=10,
	max_al_iter=10,
	con_tol=0.005,
	ρ_init=1.0, 
	ρ_scale=10.0, 
	verbose=true)

@show prob.s_data.iter[1]

# ## solution
x_sol, u_sol = get_trajectory(prob)
q_sol = state_to_configuration(x_sol)
visualize!(vis, planarpush, q_sol, Δt=h)

# ## benchmark 
@benchmark IterativeLQR.solve!($prob, x̄, ū,
	linesearch = :armijo,
	α_min=1.0e-5,
	obj_tol=1.0e-3,
	grad_tol=1.0e-3,
	max_iter=10,
	max_al_iter=10,
	con_tol=0.005,
	ρ_init=1.0, 
	ρ_scale=10.0, 
	verbose=false) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))
