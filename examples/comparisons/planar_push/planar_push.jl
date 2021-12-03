using Plots
using Random
Random.seed!(1)

# ## planar push model 
include("../../../models/planar_push/model.jl")
include("../../../models/planar_push/simulator.jl")

MODE = :translate
MODE = :rotate 

# ## visualization 
include("../../../models/planar_push/visuals.jl")
include("../../../models/visualize.jl")
vis = Visualizer() 
render(vis)

# ## build implicit dynamics
h = 0.1
T = 26

path = @get_scratch!("planarpush")
@load joinpath(path, "residual.jld2") r_func rz_func rθ_func rz_array rθ_array

function get_simulator(model, h, r_func, rz_func, rθ_func; 
	T=1, r_tol=1.0e-6, κ_eval_tol=1.0e-4, nc=model.nc, nb=model.nc, diff_sol=true)

	sim = Simulator(model, T; 
        h=h, 
        residual=r_func, 
        jacobian_z=rz_func, 
        jacobian_θ=rθ_func,
        diff_sol=diff_sol,
        solver_opts=InteriorPointOptions(
            undercut=Inf,
            γ_reg=0.1,
            r_tol=r_tol,
            κ_tol=κ_eval_tol,  
            max_ls=25,
            ϵ_min=0.25,
            diff_sol=diff_sol,
            verbose=false))  

    # set trajectory sizes
	sim.traj.γ .= [zeros(nc) for t = 1:T] 
	sim.traj.b .= [zeros(nb) for t = 1:T] 

    sim.grad.∂γ1∂q1 .= [zeros(nc, model.nq) for t = 1:T] 
	sim.grad.∂γ1∂q2 .= [zeros(nc, model.nq) for t = 1:T]
	sim.grad.∂γ1∂u1 .= [zeros(nc, model.nu) for t = 1:T]
	sim.grad.∂b1∂q1 .= [zeros(nb, model.nq) for t = 1:T] 
	sim.grad.∂b1∂q2 .= [zeros(nb, model.nq) for t = 1:T]
	sim.grad.∂b1∂u1 .= [zeros(nb, model.nu) for t = 1:T]
	
    return sim
end

sim = get_simulator(planarpush, h, eval(r_func), eval(rz_func), eval(rθ_func); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, nc=1, nb=9)

q1 = nominal_configuration(planarpush)
q2 = nominal_configuration(planarpush)
v1 = (q2 - q1) ./ h
u1 = zeros(planarpush.nu)
q3 = RoboDojo.step!(sim, q2, v1, u1, 1)
sim.grad.∂q3∂q1[1] 
sim.grad.∂q3∂q2[1] 
sim.grad.∂q3∂u1[1] 

nx = 2 * planarpush.nq 
nu = planarpush.nu 
nz = nx + nu 
nθ = nx * (nz)

@variables fz[1:nx] fη[1:nx] η[1:nz] θ[1:nθ] 

function cost(fz, fη, η, θ) 
    M = reshape(θ, nx, nz) 
    r = fη - fz - M * η 
    return [transpose(r) * r]
end

c = cost(fz, fη, η, θ)
cθ = Symbolics.gradient(c[1], θ)
cθθ = Symbolics.hessian(c[1], θ)

c_func = eval(Symbolics.build_function(c, fz, fη, η, θ)[2])
cθ_func = eval(Symbolics.build_function(cθ, fz, fη, η, θ)[2])
cθθ_func = eval(Symbolics.build_function(cθθ, fz, fη, η, θ)[2])

fz0 = rand(nx)
fη0 = rand(nx) 
η0 = rand(nz) 
θ0 = rand(nθ)
θθ0 = rand(nθ, nθ)

c0 = zeros(1)
cθ0 = zeros(nθ)
cθθ0 = zeros(nθ, nθ)

c_func(c0, fz0, fη0, η0, θ0)
# @benchmark c_func($c0, $fz0, $fη0, $η0, $θ0)

cθ_func(cθ0, fz0, fη0, η0, θ0)
# @benchmark cθ_func($cθ0, $fz0, $fη0, $η0, $θ0)

cθθ_func(cθθ0, fz0, fη0, η0, θ0)
# @benchmark cθ_func($cθθ0, $fz0, $fη0, $η0, $θθ0)

N = 2 * nz
η = [zeros(nz) for i = 1:N]
ϵ = 0.1
for i = 1:nz 
    η[i][i] = ϵ 
    η[i+nz][i] = -ϵ
end

z0 = [q1; q2; u1]

function planarpush_ss(z) 
    q1 = z[1:planarpush.nq] 
    q2 = z[planarpush.nq .+ (1:planarpush.nq)] 
    v1 = (q2 - q1) ./ h
    u1 = z[2 * planarpush.nq .+ (1:planarpush.nu)] 
    q3 = RoboDojo.step!(sim, q2, v1, u1, 1) 
    [q2; q3] 
end

fz = planarpush_ss(z0) 
fη = [planarpush_ss(z0 + η[i]) for i = 1:N]

ls = LeastSquares(N, 
    fz, fη, η, 
    c_func, cθ_func, cθθ_func, 
    [0.0], zeros(nθ), zeros(nθ, nθ), 
    [0.0], zeros(nθ), zeros(nθ, nθ), 
    zeros(nθ), zeros(nθ),
    lu_solver(zeros(nθ, nθ)))
ls.θ
# eval_cost!(ls, z0)
# @benchmark eval_cost!($ls)

# eval_grad!(ls, z0)
# @benchmark eval_grad!($ls)

# eval_hess!(ls, z0)
# @benchmark eval_hess!($ls)

update!(ls)
reshape(ls.θ, nx, nz)

[sim.grad.∂q3∂q1[1] sim.grad.∂q3∂q2[1] sim.grad.∂q3∂u1[1]]

















# ## discrete-time state-space model
im_dyn = ImplicitDynamics(planarpush, h, eval(r_func), eval(rz_func), eval(rθ_func); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-2, nc=1, nb=9) 

nx = 2 * planarpush.nq
nu = planarpush.nu 
nw = planarpush.nw

# ## dynamics for iLQR
ilqr_dyn = IterativeLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
					(dx, x, u, w) -> fx(dx, im_dyn, x, u, w), 
					(du, x, u, w) -> fu(du, im_dyn, x, u, w), 
					nx, nx, nu)  

# ## model for iLQR
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

x1 = [q0; q1]
ū = MODE == :translate ? [t < 5 ? [1.0; 0.0] : [0.0; 0.0] for t = 1:T-1] : [t < 5 ? [1.0; 0.0] : t < 10 ? [0.5; 0.0] : [0.0; 0.0] for t = 1:T-1]
w = [zeros(nw) for t = 1:T-1]
x̄ = rollout(model, x1, ū)
q̄ = state_to_configuration(x̄)
visualize!(vis, planarpush, q̄, Δt=h)

prob = problem_data(model, obj, cons)
initialize_controls!(prob, ū)
initialize_states!(prob, x̄)

# ## solve
IterativeLQR.solve!(prob, verbose=true)

# ## solution
x_sol, u_sol = get_trajectory(prob)
q_sol = state_to_configuration(x_sol)
visualize!(vis, planarpush, q_sol, Δt=h)
