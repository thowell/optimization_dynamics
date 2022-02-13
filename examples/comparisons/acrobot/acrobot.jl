using Pkg 
Pkg.activate(@__DIR__)

using MuJoCo
mj_activate("/home/taylor/.mujoco/bin/mjkey.txt") # set location to MuJoCo key path

using LyceumMuJoCo, LyceumMuJoCoViz 
using FiniteDiff

using IterativeLQR
using LinearAlgebra
using Random

# ## load MuJoCo model
path = joinpath(@__DIR__, "acrobot.xml")
path = joinpath(@__DIR__, "acrobot_limits.xml")

include("mujoco_model.jl")
acrobot_mujoco = MuJoCoModel(path)
sim = MJSim(acrobot_mujoco.m, acrobot_mujoco.d)

# ## horizon 
T = 101

# ## acrobot_mujoco 
nx = acrobot_mujoco.nx
nu = acrobot_mujoco.nu 

# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> f_mujoco!(y, acrobot_mujoco, x, u), 
    (dx, x, u, w) -> fx_mujoco!(dx, acrobot_mujoco, x, u), 
    (du, x, u, w) -> fu_mujoco!(du, acrobot_mujoco, x, u), 
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

ct = IterativeLQR.Cost(objt, acrobot_mujoco.nx, acrobot_mujoco.nu)
cT = IterativeLQR.Cost(objT, acrobot_mujoco.nx, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
goal(x, u, w) = x - xT

cont = IterativeLQR.Constraint()
conT = IterativeLQR.Constraint(goal, nx, 0)
cons = [[cont for t = 1:T-1]..., conT] 

# # rollout
Random.seed!(1)
ū = [1.0e-3 * randn(acrobot_mujoco.nu) for t = 1:T-1]
w = [zeros(0) for t = 1:T-1]
x̄ = IterativeLQR.rollout(model, x1, ū)

# ## solver
solver = IterativeLQR.solver(model, obj, cons,
    opts=iLQR.Options(
        linesearch = :armijo,
        α_min=1.0e-5,
        obj_tol=1.0e-5,
        grad_tol=1.0e-5,
        max_iter=50,
        max_al_iter=10,
        con_tol=0.001,
        ρ_init=1.0, 
        ρ_scale=10.0,
        verbose=true))
IterativeLQR.initialize_controls!(solver, ū)
IterativeLQR.initialize_states!(solver, x̄)

# ## solve
IterativeLQR.reset!(solver.s_data)
IterativeLQR.solve!(solver)

@show solver.s_data.iter[1]
@show IterativeLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
@show norm(goal(solver.m_data.x[T], zeros(0), zeros(0)), Inf)
@show solver.s_data.obj[1] # augmented Lagrangian cost

# ## benchmark
solver.options.verbose = true
@benchmark IterativeLQR.solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))

# ## solution
x_sol, u_sol = IterativeLQR.get_trajectory(solver)

# ## MuJoCo visualizer
states = Array(undef, statespace(sim), T-1)
for t = 1:T-1
    sim.d.qpos .= x_sol[t][acrobot_mujoco.idx_pos]
    sim.d.qvel .= x_sol[t][acrobot_mujoco.idx_vel]
    sim.d.ctrl .= u_sol[t]
    states[:, t] .= getstate(sim)
end

visualize(sim, trajectories=[states]) # ctrl + LEFT (access trajectory mode)

