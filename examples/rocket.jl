using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR
const Rotations = OptimizationDynamics.Rotations
using LinearAlgebra
using Random

# ## visualization 
vis = Visualizer() 
render(vis)

# ## mode
MODE = :projection 
MODE = :nominal

# ## thrust max
u_max = 12.5 

# ## state-space model
h = 0.05
T = 61
info = RocketInfo(rocket, u_max, h, 
    eval(r_rocket_func), eval(rz_rocket_func), eval(rθ_rocket_func), 
    eval(r_proj_func), eval(rz_proj_func), eval(rθ_proj_func))

nx = rocket.nq
nu = rocket.nu 

# ## iLQR model
if MODE == :projection
    ilqr_dyn = iLQR.Dynamics((d, x, u, w) -> f_rocket_proj(d, info, x, u, w), 
                        (dx, x, u, w) -> fx_rocket_proj(dx, info, x, u, w), 
                        (du, x, u, w) -> fu_rocket_proj(du, info, x, u, w), 
                        nx, nx, nu)  
else
    ilqr_dyn = iLQR.Dynamics((d, x, u, w) -> f_rocket(d, info, x, u, w), 
                    (dx, x, u, w) -> fx_rocket(dx, info, x, u, w), 
                    (du, x, u, w) -> fu_rocket(du, info, x, u, w), 
                    nx, nx, nu)  
end

model = [ilqr_dyn for t = 1:T-1];

# ## initial conditions
x1 = zeros(rocket.nq)
x1[1] = 2.5
x1[2] = 2.5
x1[3] = 10.0
mrp = Rotations.MRP(Rotations.RotZ(0.25 * π) * Rotations.RotY(-0.5 * π))
x1[4:6] = [mrp.x; mrp.y; mrp.z]
x1[9] = -1.0

xT = zeros(rocket.nq)
xT[3] = rocket.length
mrpT = Rotations.MRP(Rotations.RotZ(0.25 * π) * Rotations.RotY(0.0))
xT[4:6] = [mrpT.x; mrpT.y; mrpT.z]

# ## objective
function objt(x, u, w)
	J = 0.0 

	J += 0.5 * transpose(x - xT) * Diagonal(h * [1.0e-1 * ones(3); 1.0e-5 * ones(3); 1.0e-1 * ones(3); 1000.0 * ones(3)]) * (x - xT) 
	J += 0.5 * transpose(u) * Diagonal(h * [1000.0; 1000.0; 100.0]) * u

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	J += 0.5 * transpose(x - xT) * Diagonal(h * 1000.0 * ones(rocket.nq)) * (x - xT) 

	return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for t = 1:T-1]..., cT];

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
         -1.0 - u[1]; 
         u[1] - 1.0; 
         -1.0 - u[2]; 
         u[2] - 1.0;
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

cont = iLQR.Constraint(stage_con, nx, nu, idx_ineq=collect(1:(1 + (MODE == :projection ? 0 : 6))))
conT = iLQR.Constraint(terminal_con, nx, 0, idx_ineq=collect(1:4))
cons = [[cont for t = 1:T-1]..., conT];

# ## rollout
Random.seed!(1)
ū = [1.0e-3 * randn(nu) for t = 1:T-1]
x̄ = iLQR.rollout(model, x1, ū)
visualize!(vis, rocket, x̄, Δt=h)

# ## solver 
solver = iLQR.solver(model, obj, cons, 
    opts=iLQR.Options(
        linesearch = :armijo,
        α_min=1.0e-5,
        obj_tol=1.0e-3,
        grad_tol=1.0e-3,
        max_iter=100,
        max_al_iter=15,
        con_tol=0.005,
        ρ_init=1.0, 
        ρ_scale=10.0,
        verbose=false))
iLQR.initialize_controls!(solver, ū)
iLQR.initialize_states!(solver, x̄);

# ## solve
iLQR.reset!(solver.s_data)
@time iLQR.solve!(solver);

@show iLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
@show solver.s_data.iter[1]
@show norm(terminal_con(solver.m_data.x[T], zeros(0), zeros(0))[4 .+ (1:10)], Inf)
@show solver.s_data.obj[1] # augmented Lagrangian cost

# ## solution
x_sol, u_sol = iLQR.get_trajectory(solver)
visualize!(vis, rocket, x_sol, Δt=h)
open(vis)
# ## test thrust cone constraint
all([norm(u[1:2]) <= u[3] for u in u_sol])

# ## benchmark 
solver.options.verbose = false
@benchmark iLQR.solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū));

