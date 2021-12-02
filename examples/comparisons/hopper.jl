# Model
include_model("hopper")

# modify parameters to match implicit dynamics example
gravity = 9.81 # gravity
μ_world = 0.8 # coeffusing Plots
using Random

# MODE = :translate
MODE = :rotate

# Model
include_model("planar_push_block_v3")

# Horizon
T = 26

# Time step
h = 0.1

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[model.idx_u[1:2]] .= 5.0
_uu[model.idx_λ[1:4]] .= μ_surface * gravity * h * 0.25
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[model.idx_u[1:2]] .= -5.0
_ul[model.idx_λ[1:4]] .= μ_surface * gravity * h * 0.25

ul, uu = control_bounds(model, T, _ul, _uu)

if MODE == :rotate
    # Initial and final states (translate + rotate)
    q1 = [0.0, 0.0, 0.0, -r - 1.0e-8, -0.01]
    x1 = [q1; q1]
    x_goal = 0.5
    y_goal = 0.5
    θ_goal = 0.5 * π
elseif MODE == :translate
    # Initial and final states (translate)
    q1 = [0.0, 0.0, 0.0, -r - 1.0e-8, 0.0]
    x1 = [q1; q1]
    x_goal = 1.0
    y_goal = 0.0
    θ_goal = 0.0 * π
end

qT = [x_goal, y_goal, θ_goal, x_goal-r, y_goal-r]

xT = [qT; qT]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)
xl[T][4:5] .= -Inf
xl[T][9:10] .= -Inf
xu[T][4:5] .= Inf
xu[T][9:10] .= Inf

# Objective
include_objective("velocity")
obj_velocity = velocity_objective(
    [Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3]))

_Q_track = [1.0, 1.0, 1.0, 0.1, 0.1]
Q_track = 1.0 * Diagonal([_Q_track; _Q_track])
obj_tracking = quadratic_tracking_objective(
    [Q_track for t = 1:T],
    # [Diagonal(0.1 * ones(model.m)) for t = 1:T-1],
	[Diagonal([1.0e-1 * ones(model.nu);
		zeros(model.nc);
		zeros(model.nb);
		zeros(model.m - model.nu - model.nc - model.nb)]) for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_penalty = PenaltyObjective(1.0e4, model.m)
obj = MultiObjective([obj_tracking, obj_penalty, obj_velocity])

# Constraints
include_constraints(["contact", "stage"])
t_idx = vcat([t for t = 1:T-1])
con_contact = contact_constraints(model, T)
con = multiple_constraints([con_contact])#, con_ctrl_comp, con_ctrl_lim])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation on state

# Random.seed!(1)
# for i = 1:1
# u0 = [0.1 * randn(model.m) for t = 1:T-1] # random controls
u0 = [0.1 * ones(model.m) for t = 1:T-1] # random controls
if MODE == :rotate
    # u0 = [0.1 * randn(model.m) for t = 1:T-1] # random controls
    for t = 1:T-1
        if t < 5
            u0[t][1:2] = [1.0; 0.0]
        elseif t < 10
            u0[t][1:2] = [0.5; 0.0]
        else
            u0[t][1:2] .= 0.0
        end
    end
elseif MODE == :translate
    # translation
    # u0 = [0.1 * randn(model.m) for t = 1:T-1] # random controls
    for t = 1:T-1
        if t < 5
            u0[t][1:2] = [1.0; 0.0]
        else
            u0[t][1:2] .= 0.0
        end
    end
end

# Pack trajectoriesff into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0),
	tol = 1.0e-3, c_tol = 1.0e-3, max_iter = 2500)

@show check_slack(z̄, prob)
# end

# # translate iters:
# iters = [91, 101, 122, 77, 106]
# # mean(iters)
# # std(iters)
#
# # translate + rotation iters:
# iters = [122, 85, 70, 121, 168]
# mean(iters)
# std(iters)

x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)
q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
h̄ = h

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/planar_push_translate_direct.jld2") x̄ ū
# @load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/planar_push_translate_direct.jld2") x̄ ū

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/planar_push_rotate_direct.jld2") x̄ ū
@load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/planar_push_rotate_direct.jld2") x̄ ū

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
#open(vis)
visualize!(vis, model,
    q, u,
    Δt = h,
	r = model.block_dim[1] + model.block_rnd)

plot(hcat(q̄...)', color = :red, width = 1.0, labels = "")
plot(hcat([ū..., ū[end]]...)[model.idx_u[1:2], :]', linetype = :steppost, color = :black, width = 1.0, labels = "")

# q_array = hcat(q...)
# u_array = hcat(u...)
#
# traj = Dict("q" => q_array, "u" => u_array)

# using NPZ
# i = 2
# file_path = joinpath(pwd(), "examples/contact_implicit/manipulation/", "traj$i.npz")
# npzwrite(file_path, traj)
icient of friction
μ_joint = 0.0

mb = 3.0 # body mass
ml = 0.3  # leg mass
Jb = 0.75 # body inertia
Jl = 0.075 # leg inertia

model = Hopper{Discrete, FixedTime}(n, m, d,
			   mb, ml, Jb, Jl,
			   μ_world, gravity,
			   qL, qU,
			   uL, uU,
			   nq,
		       nu,
		       nc,
		       nf,
		       nb,
		   	   ns,
		       idx_u,
		       idx_λ,
		       idx_b,
		       idx_ψ,
		       idx_η,
		       idx_s)

# Horizon
T = 21
# Time step
h = 0.05
tf = h * (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 10.0 #model_ft.uU
_ul = zeros(model.m)
_ul[model.idx_u] .= -10.0 #model_ft.uL
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0; 0.5; 0.0; 0.5]
# qM = [0.5; 0.5; 0.0; 0.5]
qT = [1.0; 0.5; 0.0; 0.5]
q_ref = [0.5; 0.75; 0.0; 0.25]
x_ref = [q_ref; q_ref]
ql = [-Inf; 0; -Inf; 0.0]
qu = [Inf; Inf; Inf; 1.0]
xl, xu = state_bounds(model, T, [ql; ql], [qu; qu])
xl[1][nq .+ (1:nq)] = q1
xu[1][nq .+ (1:nq)] = q1

# Objective

# # gait 1
Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0])
	: 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])) for t = 1:T]
q = [2.0 * Q[t] * x_ref for t = 1:T]
R = [0.1 * Diagonal(ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

# gait 2
Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0])
	: 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])) for t = 1:T]
q = [2.0 * Q[t] * x_ref for t = 1:T]
R = [1.0 * Diagonal(ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

# gait 3
Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0])
	: 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])) for t = 1:T]
q = [2.0 * Q[t] * x_ref for t = 1:T]
R = [1.0e-3 * Diagonal(ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

obj_tracking = quadratic_tracking_objective(Q, R, q, r)

obj_contact_penalty = PenaltyObjective(1.0e4, model.m)

obj = MultiObjective([obj_tracking, obj_contact_penalty])

# Constraints
include_constraints(["contact", "stage", "loop"])

function pinned!(c, x, u, t)
    _q1 = view(x, 1:4)
    _q2 = view(x, 4 .+ (1:4))
    c[1:2] = kinematics(model, _q1) - kinematics(model, q1)
    c[3:4] = kinematics(model, _q2) - kinematics(model, q1)
    nothing
end
n_pinned = 4
con_pinned = stage_constraints(pinned!, n_pinned, (1:0), collect(1:1))

function distance_traveled!(c, x, u, t)
    x_travel = 0.5
    _q1 = view(x, 1:4)
    _q2 = view(x, 4 .+ (1:4))
	c[1] = x[1] - x_travel #TODO: add methods for trajectory-wise constraints in order to change to exactly match implicit dynamics example
    c[2] = x[5] - x_travel
    nothing
end

n_distance = 2
con_distance = stage_constraints(distance_traveled!, n_distance, (1:2), collect(T:T))

con_contact = contact_constraints(model, T)

con_loop = loop_constraints(model, collect([(2:model.nq)...,
	(nq .+ (2:model.nq))...]), 1, T)

con = multiple_constraints([con_contact, con_pinned, con_distance, con_loop])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
x0 = [[q1; q1] for t = 1:T]
@load joinpath(pwd(), "examples/implicit_dynamics/examples/comparisons/hopper_stand.jld2") u_stand
u0 = [u_stand[1] for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
# include_snopt()

@time z̄, info = solve(prob, copy(z0),
	nlp = :ipopt,
	tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

x̄, ū = unpack(z̄, prob)
@show check_slack(z̄, prob)

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_1_direct.jld2") x̄ ū
# @load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_1_direct.jld2") x̄ ū

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_2_direct.jld2") x̄ ū
# @load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_2_direct.jld2") x̄ ū

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_3_direct.jld2") x̄ ū
@load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_3_direct.jld2") x̄ ū

# compute comparable objective
J = 0.0
x = x̄
u = ū
for t = 1:T
    if t < T
        J += x[t]' * Q[t] * x[t] + q[t]' * x[t] + u[t][1:2]' * R[t][1:2, 1:2] * u[t][1:2] + r[t][1:2]' * u[t][1:2]
    elseif t == T
        J += x[t]' * Q[t] * x[t] + q[t]' * x[t]
    else
        J += 0.0
    end
end
@show J

plot(hcat(ū...)[1:2, :]', linetype = :steppost)


using Plots
t = range(0, stop = h * (T - 1), length = T)
plot(t, hcat(ū..., ū[end])[1:2,:]', linetype=:steppost,
	xlabel="time (s)", ylabel = "control",
	label = ["angle" "length"],
	width = 2.0, legend = :top)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model,
    state_to_configuration(x̄),
	# state_to_configuration([[x̄[1] for i = 1:50]...,x̄..., [x̄[end] for i = 1:50]...]),
	Δt = h,
	scenario = :flip)
