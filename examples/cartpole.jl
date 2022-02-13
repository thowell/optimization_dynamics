using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR
using LinearAlgebra
using Random

# ## visualization 
vis = Visualizer() 
render(vis)

# ## mode
MODE = :friction 
MODE = :frictionless

# ## state-space model
h = 0.05
T = 51

if MODE == :friction 
	im_dyn = ImplicitDynamics(cartpole_friction, h, eval(r_cartpole_friction_func), eval(rz_cartpole_friction_func), eval(rθ_cartpole_friction_func); 
		r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3, no_impact=true) 
        cartpole_friction.friction .= [0.35; 0.35]
        ## cartpole_friction.friction .= [0.25; 0.25]
        ## cartpole_friction.friction .= [0.1; 0.1]
        ## cartpole_friction.friction .= [0.01; 0.01]
else
	im_dyn = ImplicitDynamics(cartpole_frictionless, h, eval(r_cartpole_frictionless_func), eval(rz_cartpole_frictionless_func), eval(rθ_cartpole_frictionless_func); 
    	        r_tol=1.0e-8, κ_eval_tol=1.0, κ_grad_tol=1.0, no_impact=true, no_friction=true) 
end

nx = 2 * cartpole_friction.nq
nu = cartpole_friction.nu 

# ## iLQR model
ilqr_dyn = iLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
					(dx, x, u, w) -> fx(dx, im_dyn, x, u, w), 
					(du, x, u, w) -> fu(du, im_dyn, x, u, w), 
					nx, nx, nu)  
model = [ilqr_dyn for t = 1:T-1];

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
	J += transpose(u) * u
	return J
end

function objT(x, u, w)
        J = 0.0 
        J += transpose(x - xT) * (x - xT) 
        return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for t = 1:T-1]..., cT];

# ## constraints
function terminal_con(x, u, w) 
    [
    x - xT; # goal 
    ]
end

cont = iLQR.Constraint()
conT = iLQR.Constraint(terminal_con, nx, 0)
cons = [[cont for t = 1:T-1]..., conT];

# ## rollout
ū = [(t == 1 ? -1.5 : 0.0) * ones(nu) for t = 1:T-1] # set value to -1.0 when friction coefficient = 0.25
x̄ = iLQR.rollout(model, x1, ū)
q̄ = state_to_configuration(x̄)
visualize!(vis, cartpole_friction, q̄, Δt=h);

# ## solver 
solver = iLQR.solver(model, obj, cons, 
    opts=iLQR.Options(linesearch = :armijo,
    α_min=1.0e-5,
    obj_tol=1.0e-5,
    grad_tol=1.0e-3,
    max_iter=100,
    max_al_iter=20,
    con_tol=0.005,
    ρ_init=1.0, 
    ρ_scale=10.0, 
    verbose=true))
iLQR.initialize_controls!(solver, ū)
iLQR.initialize_states!(solver, x̄);

# ## solve
iLQR.reset!(solver.s_data)
iLQR.solve!(solver);

@show iLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
@show solver.s_data.iter[1]
@show norm(terminal_con(solver.m_data.x[T], zeros(0), zeros(0)), Inf)
@show solver.s_data.obj[1] # augmented Lagrangian cost
        
# ## solution
x_sol, u_sol = iLQR.get_trajectory(solver)
q_sol = state_to_configuration(x_sol)
visualize!(vis, cartpole_friction, q_sol, Δt=h);

# ## benchmark 
solver.options.verbose = false
@benchmark iLQR.solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū));