function hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w) 
    model = RoboDojo.hopper

    # dimensions
    nq = model.nq
    nu = model.nu 

    # configurations
    
    q1⁻ = x[1:nq] 
    q2⁻ = x[nq .+ (1:nq)]
    q2⁺ = y[1:nq]
    q3⁺ = y[nq .+ (1:nq)]

    # control 
    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    # ψ = u[nu + 4 + 4 .+ (1:2)] 
    # η = u[nu + 4 + 4 + 2 .+ (1:4)] 
    # sϕ = u[nu + 4 + 4 + 2 + 4 .+ (1:4)]
    # sψ = u[nu + 4 + 4 + 2 + 4 + 4 .+ (1:2)]
    # sα = u[nu + 4 + 4 + 2 + 4 + 4 + 2 .+ (1:1)]
    
    E = [1.0 -1.0] # friction mapping 
    J = RoboDojo.contact_jacobian(model, q2⁺)
    λ = transpose(J) * [[E * β[1:2]; γ[1]];
                        [E * β[3:4]; γ[2]];
                         γ[3:4]]
    λ[3] += (model.body_radius * E * β[1:2])[1] # friction on body creates a moment

    [q2⁺ - q2⁻;
     RoboDojo.dynamics(model, mass_matrix, dynamics_bias, 
        h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)]
end

function hopper_dyn1(mass_matrix, dynamics_bias, h, y, x, u, w)
    [
     hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
     y[nx .+ (1:nx)] - x
    ]
end

function hopper_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
    [
     hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
     y[nx .+ (1:nx)] - x[nx .+ (1:nx)]
    ]
end


function contact_constraints_inequality(h, x, u, w) 
    model = RoboDojo.hopper

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    ψ = u[nu + 4 + 4 .+ (1:2)] 
    η = u[nu + 4 + 4 + 2 .+ (1:4)] 
    sϕ = u[nu + 4 + 4 + 2 + 4 .+ (1:4)]
    sψ = u[nu + 4 + 4 + 2 + 4 + 4 .+ (1:2)]
    sα = u[nu + 4 + 4 + 2 + 4 + 4 + 2 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3) 
   
    v = (q3 - q2) ./ h[1]
    vT_body = v[1] + model.body_radius * v[3]
    vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
    vT = [vT_body; -vT_body; vT_foot; -vT_foot]
    
    ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
    
    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    [
     γ .* sϕ .- sα;
     β .* η .- sα;
     ψ .* sψ  .- sα;
    ]
end

function contact_constraints_equality(h, x, u, w) 
    model = RoboDojo.hopper

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    ψ = u[nu + 4 + 4 .+ (1:2)] 
    η = u[nu + 4 + 4 + 2 .+ (1:4)] 
    sϕ = u[nu + 4 + 4 + 2 + 4 .+ (1:4)]
    sψ = u[nu + 4 + 4 + 2 + 4 + 4 .+ (1:2)]
    sα = u[nu + 4 + 4 + 2 + 4 + 4 + 2 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3) 
   
    v = (q3 - q2) ./ h[1]
    vT_body = v[1] + model.body_radius * v[3]
    vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
    vT = [vT_body; -vT_body; vT_foot; -vT_foot]
    
    ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
    
    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    [
     sϕ - ϕ;
     sψ - fc;
     η - vT - ψ_stack;
    ]
end

# ## horizon 
T = 21 
h = 0.05

# ## hopper 
nx = 2 * RoboDojo.hopper.nq
nu = RoboDojo.hopper.nu + 4 + 4 + 2 + 4 + 4 + 2 + 1
nw = RoboDojo.hopper.nw

# ## model
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.hopper)
d1 = DirectTrajectoryOptimization.Dynamics((y, x, u, w) -> hopper_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx, nx, nu)
dt = DirectTrajectoryOptimization.Dynamics((y, x, u, w) -> hopper_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx, 2 * nx, nu)

dyn = [d1, [dt for t = 2:T-1]...]
model = DynamicsModel(dyn)

# ## initial conditions
q1 = [0.0; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
qM = [0.5; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
qT = [1.0; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
q_ref = [0.5; 0.75 + RoboDojo.hopper.foot_radius; 0.0; 0.25]

x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

# ## objective
function obj1(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x - x_ref) 
	J += 0.5 * transpose(u) * Diagonal(1.0e-1 * ones(nu)) * u
    J += 1000.0 * u[nu]
	return J
end

function objt(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x[1:nx] - x_ref)
	J += 0.5 * transpose(u) * Diagonal(1.0e-1 * ones(nu)) * u
    J += 1000.0 * u[nu]
	return J
end

function objT(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
    return J
end

c1 = DirectTrajectoryOptimization.Cost(obj1, nx, nu, nw, [1])
ct = DirectTrajectoryOptimization.Cost(objt, 2 * nx, nu, nw, [t for t = 2:T-1])
cT = DirectTrajectoryOptimization.Cost(objT, 2 * nx, 0, 0, [T])
obj = [c1, ct, cT]

# ## constraints
function stage1_eq(x, u, w) 
    [
   	RoboDojo.kinematics_foot(RoboDojo.hopper, x[1:RoboDojo.hopper.nq]) - RoboDojo.kinematics_foot(RoboDojo.hopper, x1[1:RoboDojo.hopper.nq]);
	RoboDojo.kinematics_foot(RoboDojo.hopper, x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)]) - RoboDojo.kinematics_foot(RoboDojo.hopper, x1[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)])
    ]
end

function terminal_con_eq(x, u, w) 
	θ = x[nx .+ (1:nx)]
    [
	x[1:RoboDojo.hopper.nq][collect([2, 3, 4])] - θ[1:RoboDojo.hopper.nq][collect([2, 3, 4])]
	x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])] - θ[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])]
    ]
end

function terminal_con_ineq(x, u, w) 
	x_travel = 0.5
	θ = x[nx .+ (1:nx)]
    [
	x_travel - (x[1] - θ[1])
	x_travel - (x[RoboDojo.hopper.nq + 1] - θ[RoboDojo.hopper.nq + 1])
    ]
end

contact_ineq1 = StageConstraint((x, u, w) -> contact_constraints_inequality(h, x, u, w), nx, nu, nw, [1], :inequality)
contact_ineqt = StageConstraint((x, u, w) -> contact_constraints_inequality(h, x, u, w), 2 * nx, nu, nw, [t for t = 2:T-1], :inequality)
contact_eq1 = StageConstraint((x, u, w) -> contact_constraints_equality(h, x, u, w), nx, nu, nw, [1], :equality)
contact_eqt = StageConstraint((x, u, w) -> contact_constraints_equality(h, x, u, w), 2 * nx, nu, nw, [t for t = 2:T-1], :equality)

ql = [-Inf; 0; -Inf; 0.0]
qu = [Inf; Inf; Inf; 1.0]
xl1 = [q1; ql] 
xu1 = [q1; qu]
xlt = [ql; ql; -Inf * ones(nx)] 
xut = [qu; qu; Inf * ones(nx)]
ul = [-10.0; -10.0; zeros(nu - 2)]
uu = [10.0; 10.0; Inf * ones(nu - 2)]

bnd1 = Bound(nx, nu, [1], xl=xl1, xu=xu1, ul=ul, uu=uu)
bndt = Bound(2 * nx, nu, [t for t = 2:T-1], xl=xlt, xu=xut, ul=ul, uu=uu)
bndT = Bound(2 * nx, 0, [T], xl=xlt, xu=xut)

con_eq1 = StageConstraint(stage1_eq, nx, nu, nw, [1], :equality)
conT_eq = StageConstraint(terminal_con_eq, 2 * nx, nu, nw, [T], :equality)
conT_ineq = StageConstraint(terminal_con_ineq, 2 * nx, nu, nw, [T], :inequality)

cons = ConstraintSet([bnd1, bndt, bndT], [contact_ineq1, contact_ineqt, contact_eq1, contact_eqt, con_eq1, conT_eq, conT_ineq])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, options=Options(
    tol=1.0e-3,
    constr_viol_tol=1.0e-3,
))

# ## initialize
x_interpolation = [x1, [[x1; x1] for t = 2:T]...]
u_guess = [[0.0; RoboDojo.hopper.gravity * RoboDojo.hopper.mass_body * 0.5 * h[1]; 0.0 * rand(nu - 2)] for t = 1:T-1]
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = x_interpolation[t]
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
initialize!(s, z0)

# ## solve
@time DirectTrajectoryOptimization.solve!(s)

# ## solution
@show trajopt.x[1]
@show trajopt.x[T]
sum([u[nu] for u in trajopt.u[1:end-1]])
trajopt.x[1] - trajopt.x[T][1:nx]

# ## visualize 
vis = Visualizer() 
render(vis)
q_sol = state_to_configuration([x[1:nx] for x in trajopt.x])
RoboDojo.visualize!(vis, RoboDojo.hopper, q_sol, Δt=h)

maximum([norm(contact_constraints_equality(h, trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
maximum([norm(max.(0.0, contact_constraints_inequality(h, trajopt.x[t], trajopt.u[t], zeros(0))), Inf) for t = 1:T-1])
maximum([norm(contact_constraints_equality(h, trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
maximum([norm(hopper_dyn(mass_matrix, dynamics_bias, h, trajopt.x[t+1], trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
minimum([min.(0.0, u[2 .+ (1:nu-2)]) for u in trajopt.u[1:end-1]])