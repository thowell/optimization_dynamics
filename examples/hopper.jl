using Plots
using Random
Random.seed!(1)

# ## visualize 
vis = Visualizer() 
render(vis)

# ## build implicit dynamics
T = 21
h = 0.05

hopper = RoboDojo.hopper

im_dyn1 = ImplicitDynamics(hopper, h, 
	eval(RoboDojo.residual_expr(hopper)), 
	eval(RoboDojo.jacobian_var_expr(hopper)), 
	eval(RoboDojo.jacobian_data_expr(hopper)); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3,
	nn=4 * hopper.nq, n=2 * hopper.nq, m=hopper.nu + 2 * hopper.nq) 

im_dynt = ImplicitDynamics(hopper, h, 
	eval(RoboDojo.residual_expr(hopper)), 
	eval(RoboDojo.jacobian_var_expr(hopper)), 
	eval(RoboDojo.jacobian_data_expr(hopper)); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3,
	nn=4 * hopper.nq, n=4 * hopper.nq, m=hopper.nu) 

function f1(d, model::ImplicitDynamics, x, u, w)
	nq = model.eval_sim.model.nq
	nu = model.eval_sim.model.nu 

	θ = u[nu .+ (1:(2 * nq))]
	q1 = θ[1:nq]
	q2 = θ[nq .+ (1:nq)]
	u1 = u[1:nu] 

	v1 = (q2 - q1) ./ model.eval_sim.h

	q3 = RoboDojo.step!(model.eval_sim, q2, v1, u1, 1)

	d[1:nq] .= q2 
	d[nq .+ (1:nq)] .= q3
	d[2 * nq .+ (1:(2 * nq))] = θ

	return d
end

function f1x(dx, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq
	nu = model.eval_sim.model.nu 

	θ = u[nu .+ (1:(2 * nq))]
	q1 = θ[1:nq]
	q2 = θ[nq .+ (1:nq)]
	u1 = u[1:nu] 

	v1 = (q2 - q1) ./ model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, v1, u1, 1)

	dx .= [zeros(nq, nq) zeros(nq, nq); zeros(nq, 2 * nq); zeros(2 * nq , 2 * nq)]

	return dx
end
	
function f1u(du, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq
	nu = model.eval_sim.model.nu 

	θ = u[nu .+ (1:(2 * nq))]
	q1 = θ[1:nq]
	q2 = θ[nq .+ (1:nq)]
	u1 = u[1:nu] 

	v1 = (q2 - q1) ./ model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, v1, u1, 1)

	∂q3∂q1 = model.grad_sim.grad.∂q3∂q1[1]
	∂q3∂q2 = model.grad_sim.grad.∂q3∂q2[1]
	∂q3∂u1 = model.grad_sim.grad.∂q3∂u1[1]
	
	du .= [zeros(nq, nu) zeros(nq, nq) I; ∂q3∂u1 ∂q3∂q1 ∂q3∂q2; zeros(2 * nq, nu) I]

	return du
end

function ft(d, model::ImplicitDynamics, x, u, w)
	nq = model.eval_sim.model.nq
	nu = model.eval_sim.model.nu 

	θ = x[2 * nq .+ (1:(2 * nq))]
	q1 = x[1:nq]
	q2 = x[nq .+ (1:nq)]
	u1 = u[1:nu] 

	v1 = (q2 - q1) ./ model.eval_sim.h

	q3 = RoboDojo.step!(model.eval_sim, q2, v1, u1, 1)

	d[1:nq] .= q2 
	d[nq .+ (1:nq)] .= q3
	d[2 * nq .+ (1:(2 * nq))] = θ

	return d
end

function ftx(dx, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq
	nu = model.eval_sim.model.nu 

	θ = x[2 * nq .+ (1:(2 * nq))]
	q1 = x[1:nq]
	q2 = x[nq .+ (1:nq)]
	u1 = u[1:nu] 

	v1 = (q2 - q1) ./ model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, v1, u1, 1)

	∂q3∂q1 = model.grad_sim.grad.∂q3∂q1[1]
	∂q3∂q2 = model.grad_sim.grad.∂q3∂q2[1]

	dx .= [zeros(nq, nq) I zeros(nq, 2 * nq); ∂q3∂q1 ∂q3∂q2 zeros(nq, 2 * nq); zeros(2 * nq, 2 * nq) I]

	return dx
end
	
function ftu(du, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq
	nu = model.eval_sim.model.nu 

	θ = x[2 * nq .+ (1:(2 * nq))]
	q1 = x[1:nq]
	q2 = x[nq .+ (1:nq)]
	u1 = u[1:nu] 

	v1 = (q2 - q1) ./ model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, v1, u1, 1)

	∂q3∂u1 = model.grad_sim.grad.∂q3∂u1[1]
	
	du .= [zeros(nq, nu); ∂q3∂u1; zeros(2 * nq, nu)]

	return du
end

ilqr_dyn1 = IterativeLQR.Dynamics((d, x, u, w) -> f1(d, im_dyn1, x, u, w), 
					(dx, x, u, w) -> f1x(dx, im_dyn1, x, u, w), 
					(du, x, u, w) -> f1u(du, im_dyn1, x, u, w), 
					4 * hopper.nq, 2 * hopper.nq, hopper.nu + 2 * hopper.nq)  

ilqr_dynt = IterativeLQR.Dynamics((d, x, u, w) -> ft(d, im_dynt, x, u, w), 
	(dx, x, u, w) -> ftx(dx, im_dynt, x, u, w), 
	(du, x, u, w) -> ftu(du, im_dynt, x, u, w), 
	4 * hopper.nq, 4 * hopper.nq, hopper.nu)  

model = [ilqr_dyn1, [ilqr_dynt for t = 2:T-1]...]

# ## initial conditions
q1 = [0.0; 0.5 + hopper.foot_radius; 0.0; 0.5]
qM = [0.5; 0.5 + hopper.foot_radius; 0.0; 0.5]
qT = [1.0; 0.5 + hopper.foot_radius; 0.0; 0.5]
q_ref = [0.5; 0.75 + hopper.foot_radius; 0.0; 0.25]

x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

# ## objective
function obj1(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x - x_ref) 
	J += 0.5 * transpose(u) * Diagonal([1.0e-1 * ones(hopper.nu); 1.0e-1 * ones(hopper.nq); 1.0e-5 * ones(hopper.nq)]) * u
	return J
end

function objt(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x - [x_ref; zeros(2 * hopper.nq)]) * 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0; zeros(2 * hopper.nq)]) * (x - [x_ref; zeros(2 * hopper.nq)]) 
	J += 0.5 * transpose(u) * Diagonal(1.0e-1 * ones(hopper.nu)) * u
	return J
end

function objT(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x - [x_ref; zeros(2 * hopper.nq)]) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; zeros(2 * hopper.nq)]) * (x - [x_ref; zeros(2 * hopper.nq)]) 
	return J
end

c1 = IterativeLQR.Cost(obj1, 2 * hopper.nq, hopper.nu + 2 * hopper.nq, 0)
ct = IterativeLQR.Cost(objt, 4 * hopper.nq, hopper.nu, 0)
cT = IterativeLQR.Cost(objT, 4 * hopper.nq, 0, 0)
obj = [c1, [ct for t = 2:T-1]..., cT]

# ## constraints
ul = [-10.0; -10.0]
uu = [10.0; 10.0]
 
function stage1_con(x, u, w) 
    [
    ul - u[1:hopper.nu]; # control limit (lower)
    u[1:hopper.nu] - uu; # control limit (upper)

	u[hopper.nu .+ (1:hopper.nq)] - x1[1:hopper.nq];

	RoboDojo.kinematics_foot(hopper, u[hopper.nu .+ (1:hopper.nq)]) - RoboDojo.kinematics_foot(hopper, x1[1:hopper.nq]);
	RoboDojo.kinematics_foot(hopper, u[hopper.nu + hopper.nq .+ (1:hopper.nq)]) - RoboDojo.kinematics_foot(hopper, x1[hopper.nq .+ (1:hopper.nq)])
    ]
end 

function staget_con(x, u, w) 
    [
    ul - u[collect(1:hopper.nu)]; # control limit (lower)
    u[collect(1:hopper.nu)] - uu; # control limit (upper)
    ]
end 

function terminal_con(x, u, w) 
	x_travel = 0.5
	θ = x[2 * hopper.nq .+ (1:(2 * hopper.nq))]
    [
	x_travel - (x[1] - θ[1])
	x_travel - (x[hopper.nq + 1] - θ[hopper.nq + 1])
	x[1:hopper.nq][collect([2, 3, 4])] - θ[1:hopper.nq][collect([2, 3, 4])]
	x[hopper.nq .+ (1:hopper.nq)][collect([2, 3, 4])] - θ[hopper.nq .+ (1:hopper.nq)][collect([2, 3, 4])]
    ]
end

con1 = Constraint(stage1_con, 2 * hopper.nq, hopper.nu + 2 * hopper.nq, idx_ineq=collect(1:4))
cont = Constraint(staget_con, 4 * hopper.nq, hopper.nu, idx_ineq=collect(1:4))
conT = Constraint(terminal_con, 4 * hopper.nq, 0, idx_ineq=collect(1:2))
cons = [con1, [cont for t = 2:T-1]..., conT]

# ## rollout
ū_stand = [t == 1 ? [0.0; hopper.gravity * hopper.mass_body * 0.5 * h; x1] : [0.0; hopper.gravity * hopper.mass_body * 0.5 * h] for t = 1:T-1]
w = [zeros(hopper.nw) for t = 1:T-1]
x̄ = rollout(model, x1, ū_stand)
q̄ = state_to_configuration(x̄)
RoboDojo.visualize!(vis, hopper, x̄, Δt=h)

prob = problem_data(model, obj, cons)
initialize_controls!(prob, ū_stand)
initialize_states!(prob, x̄)

# ## solve
IterativeLQR.solve!(prob, verbose=true)

# ## solution
x_sol, u_sol = get_trajectory(prob)
q_sol = state_to_configuration(x_sol)
RoboDojo.visualize!(vis, hopper, q_sol, Δt=h)

