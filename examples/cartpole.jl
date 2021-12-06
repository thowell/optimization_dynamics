using Plots
using Random
Random.seed!(1)

# ## mode
MODE = :friction 
MODE = :no_friction

# ## cart-pole model 
include("../models/cartpole/model.jl")
path = @get_scratch!("cartpole")

# ## visualization 
include("../models/cartpole/visuals.jl")
include("../models/visualize.jl")
vis = Visualizer() 
render(vis)

# ## build implicit dynamics
h = 0.05
T = 51

if MODE == :friction 
	include("../models/cartpole/simulator_friction.jl")
	@load joinpath(path, "friction.jld2") r_func rz_func rθ_func rz_array rθ_array

	# ## discrete-time state-space model
	im_dyn = ImplicitDynamics(cartpole, h, eval(r_func), eval(rz_func), eval(rθ_func); 
		r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3, no_impact=true) 
	cartpole.friction .= 0.15
	# cartpole.friction .= 0.1
	# cartpole.friction .= 0.05
	# cartpole.friction .= 0.01 
else
	include("../models/cartpole/simulator_no_friction.jl")
	@load joinpath(path, "no_friction.jld2") r_no_friction_func rz_no_friction_func rθ_no_friction_func rz_no_friction_array rθ_no_friction_array
	im_dyn = ImplicitDynamics(cartpole, h, eval(r_no_friction_func), eval(rz_no_friction_func), eval(rθ_no_friction_func); 
    	r_tol=1.0e-8, κ_eval_tol=1.0, κ_grad_tol=1.0, no_impact=true, no_friction=true) 
end

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

	J += transpose(v1) * v1 
	J += transpose(x - xT) * (x - xT) 
	J += transpose(u) * u

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	J += transpose(x - xT) * (x - xT) 

	return J
end

ct = IterativeLQR.Cost(objt, nx, nu, nw)
cT = IterativeLQR.Cost(objT, nx, 0, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
function terminal_con(x, u, w) 
    [
    x - xT; # goal 
    ]
end

cont = Constraint()
conT = Constraint(terminal_con, nx, 0)
cons = [[cont for t = 1:T-1]..., conT]

# ## rollout
ū = [(t == 1 ? -1.0 : 0.0) * ones(nu) for t = 1:T-1]
w = [zeros(nw) for t = 1:T-1]
x̄ = rollout(model, x1, ū)
q̄ = state_to_configuration(x̄)
RoboDojo.visualize!(vis, cartpole, q̄, Δt=h)

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
    con_tol=0.005,
    ρ_init=1.0, 
    ρ_scale=10.0, 
	verbose=true)

@show prob.s_data.iter[1]

# ## benchmark 
@benchmark IterativeLQR.solve!($prob, x̄, ū,
	linesearch = :armijo,
    α_min=1.0e-5,
    obj_tol=1.0e-5,
    grad_tol=1.0e-5,
    max_iter=100,
    max_al_iter=10,
    con_tol=0.005,
    ρ_init=1.0, 
    ρ_scale=10.0, 
	verbose=false) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))

# ## solution
x_sol, u_sol = get_trajectory(prob)
q_sol = state_to_configuration(x_sol)
RoboDojo.visualize!(vis, cartpole, q_sol, Δt=h)

# ghost
limit_color = [0.0, 0.0, 0.0]
# limit_color = [0.0, 1.0, 0.0]

t = 1
id = t
tl = 0.05
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 5
id = t
tl = 0.15
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 10
id = t
tl = 0.25
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 15
id = t
tl = 0.35
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 20
id = t
tl = 0.45
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 25
id = t
tl = 0.55
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 30
id = t
tl = 0.65
_create_cartpole!(vis, cartpole;
tl = tl,
color = RGBA(limit_color..., tl),
i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 35
id = t
tl = 0.75
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 40
id = t
tl = 0.85
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

t = 45
id = t
tl = 0.95
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)


t = 51
id = t
tl = 1.0
_create_cartpole!(vis, cartpole;
        tl = tl,
        color = RGBA(limit_color..., tl),
        i = id)
_set_cartpole!(vis, cartpole, x_sol[t], i = id)

line_mat = LineBasicMaterial(color=color=RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0), linewidth=10.0)
# line_mat = LineBasicMaterial(color=color=RGBA(51.0 / 255.0, 1.0, 1.0, 1.0), linewidth=10.0)

points = Vector{Point{3,Float64}}()
for (i, xt) in enumerate(x_sol)
    k = kinematics(cartpole, xt)
	push!(points, Point(k[1], 0.0, k[2]))

    setobject!(vis["ee_vertex_$i"], Sphere(Point3f0(0),
        convert(Float32, 0.001)),
        MeshPhongMaterial(color = RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0)))
        settransform!(vis["ee_vertex_$i"], Translation(points[i]))
end
setobject!(vis[:ee_traj], MeshCat.Line(points, line_mat))
open(vis)
