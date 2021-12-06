using OptimizationDynamics
using IterativeLQR
using Random
Random.seed!(1)

# ## visualization 
vis = Visualizer() 
render(vis)

# ## mode
MODE = :projection 
MODE = :nominal

# ## thrust max
u_max = 12.5 

# ## rocket model 
@load joinpath(path_rocket, "residual.jld2") r_func rz_func rθ_func rz_array rθ_array
@load joinpath(path_rocket, "projection.jld2") r_func_proj rz_func_proj rθ_func_proj rz_array_proj rθ_array_proj

# ## state-space model
h = 0.05
T = 61
info = RocketInfo(rocket, u_max, h, eval(r_func), eval(rz_func), eval(rθ_func), eval(r_func_proj), eval(rz_func_proj), eval(rθ_func_proj))

nx = rocket.nq
nu = rocket.nu 
nw = rocket.nw

# ## iLQR model
if MODE == :projection
    ilqr_dyn = IterativeLQR.Dynamics((d, x, u, w) -> f_rocket_proj(d, info, x, u, w), 
                        (dx, x, u, w) -> fx_rocket_proj(dx, info, x, u, w), 
                        (du, x, u, w) -> fu_rocket_proj(du, info, x, u, w), 
                        nx, nx, nu)  
else
    ilqr_dyn = IterativeLQR.Dynamics((d, x, u, w) -> f_rocket(d, info, x, u, w), 
                    (dx, x, u, w) -> fx_rocket(dx, info, x, u, w), 
                    (du, x, u, w) -> fu_rocket(du, info, x, u, w), 
                    nx, nx, nu)  
end

model = [ilqr_dyn for t = 1:T-1]

# ## initial conditions
x1 = zeros(rocket.nq)
x1[1] = 2.5
x1[2] = 2.5
x1[3] = 10.0
mrp = MRP(RotZ(0.25 * π) * RotY(-0.5 * π))
x1[4:6] = [mrp.x; mrp.y; mrp.z]
x1[9] = -1.0

xT = zeros(rocket.nq)
xT[3] = rocket.length
mrpT = MRP(RotZ(0.25 * π) * RotY(0.0))
xT[4:6] = [mrpT.x; mrpT.y; mrpT.z]

# ## objective
function objt(x, u, w)
	J = 0.0 

	J += 0.5 * transpose(x - xT) * Diagonal(h * [1.0e-1 * ones(3); 1.0e-5 * ones(3); 1.0e-1 * ones(3); 1000.0 * ones(3)]) * (x - xT) 
	J += 0.5 * transpose(u) * Diagonal(h * [10000.0; 10000.0; 100.0]) * u

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	J += 0.5 * transpose(x - xT) * Diagonal(h * 1000.0 * ones(rocket.nq)) * (x - xT) 

	return J
end

ct = IterativeLQR.Cost(objt, nx, nu, nw)
cT = IterativeLQR.Cost(objT, nx, 0, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
x_con = [-0.5; 0.5]
y_con = [-0.75; 0.75]

function stage_con(x, u, w) 
    if MODE == :projection 
        [
         rocket.length - x[3]; 
        ]
    else 
        [
         0.0 - u[3]; # control limit (lower)
         u[3] - u_max; # control limit (upper)
         rocket.length - x[3];
        ]
    end
end 

function terminal_con(x, u, w) 
    [
     x_con[1] - x[1];
     x[1] - x_con[2];
     y_con[1] - x[2];
     x[2] - y_con[2];
     (x - xT)[collect([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]
    ]
end

cont = Constraint(stage_con, nx, nu, idx_ineq=collect(1:(1 + (MODE == :projection ? 0 : 2))))
conT = Constraint(terminal_con, nx, 0, idx_ineq=collect(1:4))
cons = [[cont for t = 1:T-1]..., conT]

# ## rollout
ū = [1.0e-3 * randn(nu) for t = 1:T-1]
w = [zeros(nw) for t = 1:T-1]
x̄ = rollout(model, x1, ū)
visualize!(vis, rocket, x̄, Δt=h)

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
    max_iter=100,
    max_al_iter=10,
    con_tol=0.001,
    ρ_init=1.0, 
    ρ_scale=10.0,
    verbose=true)
@show prob.s_data.iter[1]

# ## solution
x_sol, u_sol = get_trajectory(prob)
visualize!(vis, rocket, x_sol, Δt=h)

# ## test thrust cone constraint
all([norm(u[1:2]) <= u[3] for u in u_sol])

# ## benchmark 
@benchmark IterativeLQR.solve!($prob, x̄, ū,
    linesearch = :armijo,
    α_min=1.0e-5,
    obj_tol=1.0e-5,
    grad_tol=1.0e-5,
    max_iter=100,
    max_al_iter=10,
    con_tol=0.001,
    ρ_init=1.0, 
    ρ_scale=10.0,
    verbose=false) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))

