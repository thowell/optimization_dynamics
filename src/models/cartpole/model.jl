"""
    Cartpole w/ internal friction
"""
struct Friction end 
struct Frictionless end 

struct Cartpole{T,X} <: Model{T}
    # dimensions
    nq::Int # generalized coordinates
    nu::Int # controls
    nw::Int # parameters
	nc::Int # contact points

    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2

    friction::Vector{T} # friction coefficients for slider and arm joints
end

function kinematics(model::Cartpole, q)
    [q[1] + model.l * sin(q[2]); -model.l * cos(q[2])]
end

lagrangian(model::Cartpole, q, q̇) = 0.0

function M_func(model::Cartpole, x)
    H = [model.mc + model.mp model.mp * model.l * cos(x[2]);
		 model.mp * model.l * cos(x[2]) model.mp * model.l^2.0]
    return H
end

function B_func(model::Cartpole, x)
    [1.0; 0.0]
end

function P_func(model::Cartpole, x)
    [1.0 0.0;
     0.0 1.0]
end

function C_func(model::Cartpole, q, q̇)
    C = [0.0 -1.0 * model.mp * q̇[2] * model.l * sin(q[2]);
	 	 0.0 0.0]
    G = [0.0,
		 model.mp * model.g * model.l * sin(q[2])]
    return -C * q̇ + G
end

function dynamics(model::Cartpole{T,Friction}, mass_matrix, dynamics_bias, h, q0, q1, u1, w1, λ1, q2) where T
	# evalutate at midpoint
    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

    D1L1, D2L1 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm2, vm2)

    d = 0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2 # variational integrator (midpoint)

	return d + B_func(model, qm2) * u1[1] + transpose(P_func(model, q2)) * λ1
end

function dynamics(model::Cartpole{T,Frictionless}, mass_matrix, dynamics_bias, h, q0, q1, u1, w1, λ1, q2) where T
	# evalutate at midpoint
    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

    D1L1, D2L1 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm2, vm2)

    d = 0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2 # variational integrator (midpoint)

	return d + B_func(model, qm2) * u1[1] 
end

function residual(model::Cartpole{T,Friction}, z, θ, κ) where T
    nq = model.nq
    nu = model.nu
    nc = model.nc

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    μ_slider = θ[2nq + nu .+ (1:1)]
    μ_angle = θ[2nq + nu + 1 .+ (1:1)]
    h = θ[2nq + nu + 1 + 1 .+ (1:1)]

    q2 = z[1:nq]
    ψ = z[nq .+ (1:nc)]
    b = z[nq + nc .+ (1:nc)]
    sψ = z[nq + nc + nc .+ (1:nc)]
    sb = z[nq + nc + nc + nc .+ (1:nc)]

    vT1 = (q2[1] - q1[1]) / h
    vT2 = (q2[2] - q1[2]) / h

    λ1 = b

    [
     dynamics(model, a -> M_func(model, a), (a, b) -> C_func(model, a, b), 
        h, q0, q1, u1, zeros(model.nw), λ1, q2);
     sb[1] - vT1[1];
     ψ[1] .- μ_slider[1] * (model.mp + model.mc) * model.g * h[1];
     sb[2] - vT2[1];
     ψ[2] .- μ_angle[1] * (model.mp * model.g * model.l) * h[1];
     cone_product([ψ[1]; b[1]], [sψ[1]; sb[1]]) - [κ[1]; 0.0];
     cone_product([ψ[2]; b[2]], [sψ[2]; sb[2]]) - [κ[1]; 0.0];
    ]
end

function residual(model::Cartpole{T,Frictionless}, z, θ, κ) where T
    nq = model.nq
    nu = model.nu

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]

    return dynamics(model, a -> M_func(model, a), (a, b) -> C_func(model, a, b), 
        h, q0, q1, u1, zeros(model.nw), zeros(0), q2);
end

# models
cartpole_friction = Cartpole{Float64,Friction}(2, 1, 0, 2, 1.0, 0.2, 0.5, 9.81, [0.1; 0.1])
cartpole_frictionless = Cartpole{Float64,Frictionless}(2, 1, 0, 2, 1.0, 0.2, 0.5, 9.81, [0.0; 0.0])

