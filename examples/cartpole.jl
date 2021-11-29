using Plots
using Random
Random.seed!(1)

# ## cart-pole model 
include("../models/cartpole/model.jl")
# include("../models/cartpole/simulator_friction.jl")
include("../models/cartpole/simulator_no_friction.jl")

path = @get_scratch!("cartpole")
# @load joinpath(path, "friction.jld2") r_func rz_func rθ_func rz_array rθ_array
@load joinpath(path, "no_friction.jld2") r_no_friction_func rz_no_friction_func rθ_no_friction_func rz_no_friction_array rθ_no_friction_array

# ## visualization 
include("../models/cartpole/visuals.jl")
include("../models/visualize.jl")
vis = Visualizer() 
render(vis)

# ## build implicit dynamics
h = 0.05
T = 51

# ## friction coefficients
# cartpole.friction .= 0.35
# cartpole.friction .= 0.25 
cartpole.friction .= 0.1 
# cartpole.friction .= 0.01 
cartpole.friction .= 0.0

# ## discrete-time state-space model
# im_dyn = ImplicitDynamics(cartpole, h, eval(r_func), eval(rz_func), eval(rθ_func); 
#     r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3, no_impact=true) 

im_dyn = ImplicitDynamics(cartpole_no_friction, h, eval(r_no_friction_func), eval(rz_no_friction_func), eval(rθ_no_friction_func); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3, no_impact=true, no_friction=true) 

nx = 2 * cartpole.nq
nu = cartpole.nu 
nw = cartpole.nw

# ## dynamics for iLQR
ilqr_dyn = IterativeLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
					(dx, x, u, w) -> fx(dx, im_dyn, x, u, w), 
					(du, x, u, w) -> fu(du, im_dyn, x, u, w), 
					nx, nx, nu)  

# ## model for iLQR
model = [ilqr_dyn for t = 1:T-1]

# ## initial conditions
q0 = [0.0; 0.0]
q1 = [0.0; 0.0]
qT = [0.0; π]
q_ref = [0.0; π]

x1 = [q1; q1]
xT = [qT; qT]

# ## objective
function objt(x, u, w)
	J = 0.0 

	q1 = x[1:cartpole.nq] 
	q2 = x[cartpole.nq .+ (1:cartpole.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * v1 
	J += 0.5 * transpose(x - xT) * (x - xT) 
	J += 0.5 * transpose(u) * u

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	q1 = x[1:cartpole.nq] 
	q2 = x[cartpole.nq .+ (1:cartpole.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * v1
	J += 0.5 * transpose(x - xT) * (x - xT) 

	return J
end

ct = IterativeLQR.Cost(objt, nx, nu, nw)
cT = IterativeLQR.Cost(objT, nx, 0, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
ul = [-10.0]
uu = [10.0]

function stage_con(x, u, w) 
    [
    ul - u; # control limit (lower)
    u - uu; # control limit (upper)
    ]
end 

function terminal_con(x, u, w) 
    [
    x - xT; # goal 
    ]
end

cont = Constraint(stage_con, nx, nu, idx_ineq=collect(1:2))
conT = Constraint(terminal_con, nx, 0)
cons = [[cont for t = 1:T-1]..., conT]

# ## rollout
# ū = [(t == 1 ? -1.0 : 0.0) * ones(nu) for t = 1:T-1]
ū = [(t == 1 ? 1.0 : 0.0) * ones(nu) for t = 1:T-1] # initialize no friction model this direction (not sure why it goes the oppostive direction...)
w = [zeros(nw) for t = 1:T-1]
x̄ = rollout(model, x1, ū)
q̄ = state_to_configuration(x̄)
visualize!(vis, cartpole, q̄, Δt=h)

prob = problem_data(model, obj, cons)
initialize_controls!(prob, ū)
initialize_states!(prob, x̄)

# ## solve
IterativeLQR.solve!(prob, verbose=true)

# ## solution
x_sol, u_sol = get_trajectory(prob)
q_sol = state_to_configuration(x_sol)
visualize!(vis, cartpole, q_sol, Δt=h)
