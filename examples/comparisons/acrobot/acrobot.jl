using Pkg 
Pkg.activate(@__DIR__)

using MuJoCo
mj_activate("/home/taylor/.mujoco/bin/mjkey.txt") # set location to MuJoCo key path

using LyceumMuJoCo, LyceumMuJoCoViz 
using FiniteDiff

using IterativeLQR
using LinearAlgebra

# ## load MuJoCo model
path = joinpath(@__DIR__, "acrobot.xml")
path = joinpath(@__DIR__, "acrobot_limits.xml")

acrobot = MuJoCoModel(path)
sim = MJSim(acrobot.m, acrobot.d)

# ## horizon 
T = 101

# ## acrobot 
nx = acrobot.nx
nu = acrobot.nu 

# ## model
dyn = Dynamics(
    (y, x, u, w) -> f!(y, acrobot, x, u), 
    (dx, x, u, w) -> fx!(dx, acrobot, x, u), 
    (du, x, u, w) -> fu!(du, acrobot, x, u), 
    nx, nx, nu) 

model = [dyn for t = 1:T-1] 

# ## initial conditions
x1 = [π; 0.0; 0.0; 0.0]
xT = [0.0; 0.0; 0.0; 0.0]

# ## objective
function objt(x, u, w)
	J = 0.0 
    v = x[3:4]
	J += 0.5 * 0.1 * transpose(v) * v
	J += 0.5 * transpose(u) * u
	return J
end

function objT(x, u, w)
	J = 0.0 
    v = x[3:4]
	J += 0.5 * 0.1 * transpose(v) * v
	return J
end

ct = IterativeLQR.Cost(objt, acrobot.nx, acrobot.nu, 0)
cT = IterativeLQR.Cost(objT, acrobot.nx, 0, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
goal(x, u, w) = x - xT

cont = Constraint()
conT = Constraint(goal, nx, 0)
cons = [[cont for t = 1:T-1]..., conT] 

# # rollout
ū = [1.0e-3 * randn(acrobot.nu) for t = 1:T-1]
w = [zeros(0) for t = 1:T-1]
x̄ = rollout(model, x1, ū)

# ## problem
prob = problem_data(model, obj, cons)
initialize_controls!(prob, ū)
initialize_states!(prob, x̄)

# ## solve
solve!(prob, 
    linesearch = :armijo,
    α_min=1.0e-5,
    obj_tol=1.0e-3,
    grad_tol=1.0e-3,
    max_iter=50,
    max_al_iter=10,
    con_tol=0.001,
    ρ_init=1.0, 
    ρ_scale=10.0,
    verbose=true)

# ## solution
x_sol, u_sol = get_trajectory(prob)

# ## MuJoCo visualizer
states = Array(undef, statespace(sim), T-1)
for t = 1:T-1
    sim.d.qpos .= x_sol[t][acrobot.idx_pos]
    sim.d.qvel .= x_sol[t][acrobot.idx_vel]
    sim.d.ctrl .= u_sol[t]
    states[:, t] .= getstate(sim)
end

visualize(sim, trajectories=[states]) # ctrl + LEFT (access trajectory mode)

