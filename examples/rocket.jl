using Plots
using Random
Random.seed!(1)

# ## rocket model 
include("../models/rocket/model.jl")
include("../models/rocket/simulator.jl")

# path = @get_scratch!("rocket")
# @load joinpath(path, "residual.jld2") r_func rz_func rθ_func rz_array rθ_array

# ## visualization 
include("../models/rocket/visuals.jl")
include("../models/visualize.jl")
vis = Visualizer() 
render(vis)

# ## build implicit dynamics
h = 0.05
T = 61

# ## discrete-time state-space model
# im_dyn = ImplicitDynamics(rocket, h, eval(r_func), eval(rz_func), eval(rθ_func); 
#     r_tol=1.0e-8, κ_eval_tol=1.0, κ_grad_tol=1.0, no_impact=true, no_friction=true,
#     nn=rocket.nq, n=rocket.nq, m=rocket.nq, nc=0, nb=0) 

nx = rocket.nq
nu = rocket.nu 
nw = rocket.nw

# ## dynamics for iLQR
ilqr_dyn = IterativeLQR.Dynamics((d, x, u, w) -> f_rocket_proj(d, x, u, w), 
					(dx, x, u, w) -> fx_rocket_proj(dx, x, u, w), 
					(du, x, u, w) -> fu_rocket_proj(du, x, u, w), 
					nx, nx, nu)  

ilqr_dyn = IterativeLQR.Dynamics((d, x, u, w) -> f_rocket(d, x, u, w), 
		        (dx, x, u, w) -> fx_rocket(dx, x, u, w), 
                (du, x, u, w) -> fu_rocket(du, x, u, w), 
                nx, nx, nu)  

# ## model for iLQR
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
ul = [-5.0; -5.0; 0.0]
uu = [5.0; 5.0; 12.5]
x_con = [-0.5; 0.5]
y_con = [-0.75; 0.75]

function stage_con(x, u, w) 
    [
    (ul - u)[3]; # control limit (lower)
    (u - uu)[3]; # control limit (upper)
    rocket.length - x[3] 
    ]
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

cont = Constraint(stage_con, nx, nu, idx_ineq=collect(1:3))#collect(1:1))#
conT = Constraint(terminal_con, nx, 0, idx_ineq=collect(1:4))
cons = [[cont for t = 1:T-1]..., conT]

# ## rollout
ū = [0.0 * [1.0e-2; 1.0e-2; 1.0e-2] .* randn(nu) for t = 1:T-1]
w = [zeros(nw) for t = 1:T-1]
x̄ = rollout(model, x1, ū)
visualize!(vis, rocket, x̄, Δt=h)

prob = problem_data(model, obj, cons)
initialize_controls!(prob, ū)
initialize_states!(prob, x̄)

# ## solve
IterativeLQR.solve!(prob, 
    linesearch = :armijo,
    α_min = 1.0e-8,
    obj_tol = 1.0e-3,
    grad_tol = 1.0e-3,
    max_iter = 100,
    max_al_iter = 10,
    con_tol = 0.001,
    ρ_init = 1.0, 
    ρ_scale = 10.0,
    verbose=true)

# ## solution
x_sol, u_sol = get_trajectory(prob)
visualize!(vis, rocket, x_sol, Δt=h)

# ## test thrust cone constraint
all([norm(u[1:2]) <= u[3] for u in u_sol])