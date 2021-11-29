"""
    planar push block
        particle with contacts at each corner
"""
struct PlanarPush{T} <: Model{T}
	# dimensions
	nq::Int # generalized coordinates
	nu::Int # controls
	nw::Int # parameters
	nc::Int # contact points

    mass_block::T
	mass_pusher::T

    inertia
    μ_surface
	μ_pusher
    gravity

    contact_corner_offset
	block_dim
	block_rnd
end

# signed distance for a box
function sd_box(p, dim)
    q = abs.(p) - dim
	return norm(max.(q, 1.0e-32)) + min(maximum(q), 0.0)
end

function sd_2d_box(p, pose, dim, rnd)
	x, y, θ = pose
	R = rotation_matrix(-θ)
	p_rot = R * (p - pose[1:2])
	return sd_box(p_rot, dim) - rnd
end

# Kinematics
r_dim = 0.1

# contact corner
cc1 = [r_dim, r_dim]
cc2 = [-r_dim, r_dim]
cc3 = [r_dim, -r_dim]
cc4 = [-r_dim, -r_dim]

contact_corner_offset = [cc1, cc2, cc3, cc4]

# Parameters
μ_surface = 0.5  # coefficient of friction
μ_pusher = 0.5
gravity = 9.81
mass_block = 1.0   # mass
mass_pusher = 10.0
inertia = 1.0 / 12.0 * mass_block * ((2.0 * r_dim)^2 + (2.0 * r_dim)^2)

rnd = 0.01
dim = [r_dim, r_dim]
dim_rnd = [r_dim - rnd, r_dim - rnd]

# Methods
M_func(model::PlanarPush, q) = Diagonal([model.mass_block, model.mass_block,
	model.inertia, model.mass_pusher, model.mass_pusher])

function C_func(model::PlanarPush, q, q̇)
	[0.0, 0.0, 0.0, 0.0, 0.0]
end

function rotation_matrix(x)
	[cos(x) -sin(x); sin(x) cos(x)]
end

function ϕ_func(model::PlanarPush, q)
    p_block = q[1:3]
	p_pusher = q[4:5]

	sdf = sd_2d_box(p_pusher, p_block, model.block_dim, model.block_rnd)

    [sdf]
end

function B_func(model::PlanarPush, q)
	[0.0 0.0;
	 0.0 0.0;
	 0.0 0.0;
	 1.0 0.0;
	 0.0 1.0]
end

function N_func(model::PlanarPush, q)
	tmp(z) = ϕ_func(model, z)
    ForwardDiff.jacobian(tmp, q)
end

function p_func(model, x)
    pos = x[1:2]
    θ = x[3]
    R = rotation_matrix(θ)

    [(pos + R * model.contact_corner_offset[1])[1:2];
     (pos + R * model.contact_corner_offset[2])[1:2];
     (pos + R * model.contact_corner_offset[3])[1:2];
     (pos + R * model.contact_corner_offset[4])[1:2]]
end

function P_func(model::PlanarPush, q)

    P_block = ForwardDiff.jacobian(a -> p_func(model, a), q)

	# pusher block
	p_block = view(q, 1:3)
	p_pusher = view(q, 4:5)

	sd_p(x) = sd_2d_box(x, p_block, model.block_dim, model.block_rnd)
	Np = ForwardDiff.gradient(sd_p, p_pusher)

	n_dir = Np[1:2] ./ norm(Np[1:2])
	t_dir = rotation_matrix(0.5 * π) * n_dir

	r = p_pusher - p_block[1:2]
	m = cross([r; 0.0], [t_dir; 0.0])[3]

	P_pusherblock = [t_dir[1]; t_dir[2]; m; -t_dir[1]; -t_dir[2]]'

	return [P_block; P_pusherblock]
end

function dynamics(model::PlanarPush, mass_matrix, dynamics_bias, h, q0, q1, u1, w1, λ1, q2)
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm2, vm2)

    return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
            + B_func(model, qm2) * u1
            + transpose(N_func(model, q2)) * λ1[end:end]
            + transpose(P_func(model, q2)) * λ1[1:end-1])
end

function residual(model, z, θ, κ)
    nq = model.nq
    nu = model.nu
    nc = model.nc
    nc_impact = 1

    nb = 3 * 4 + 2 * 1

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    γ1 = z[nq .+ (1:nc_impact)]
	s1 = z[nq + nc_impact .+ (1:nc_impact)]

	ψ1 = z[nq + 2 * nc_impact .+ (1:5)]
	b1 = z[nq + 2 * nc_impact + 5 .+ (1:9)]
	sψ1 = z[nq + 2 * nc_impact + 5 + 9 .+ (1:5)]
	sb1 = z[nq + 2 * nc_impact + 5 + 9 + 5 .+ (1:9)]

	ϕ = ϕ_func(model, q2)

	λ1 = [b1; γ1]
    vT = P_func(model, q2) * (q2 - q1) / h[1]

    [
     dynamics(model, a -> M_func(model, a), (a, b) -> C_func(model, a, b),
	 	h, q0, q1, u1, zeros(model.nw), λ1, q2);
	
	 s1 .- ϕ;

	 vT[1:2] - sb1[1:2];
	 ψ1[1] .- model.μ_surface[1] * model.mass_block * model.gravity * h[1] * 0.25;

	 vT[3:4] - sb1[2 .+ (1:2)];
	 ψ1[2] .- model.μ_surface[2] * model.mass_block * model.gravity * h[1] * 0.25;

	 vT[5:6] - sb1[2 + 2 .+ (1:2)];
	 ψ1[3] .- model.μ_surface[3] * model.mass_block * model.gravity * h[1] * 0.25;

	 vT[7:8] - sb1[2 + 2 + 2 .+ (1:2)];
	 ψ1[4] .- model.μ_surface[4] * model.mass_block * model.gravity * h[1] * 0.25;

	 vT[9] - sb1[2 + 2 + 2 + 2 + 1];
	 ψ1[5] .- model.μ_pusher * γ1;

	 γ1 .* s1 .- κ;
	 cone_product([ψ1[1]; b1[1:2]], [sψ1[1]; sb1[1:2]]) - [κ; 0.0; 0.0];
	 cone_product([ψ1[2]; b1[2 .+ (1:2)]], [sψ1[2]; sb1[2 .+ (1:2)]]) - [κ; 0.0; 0.0];
	 cone_product([ψ1[3]; b1[4 .+ (1:2)]], [sψ1[3]; sb1[4 .+ (1:2)]]) - [κ; 0.0; 0.0];
	 cone_product([ψ1[4]; b1[6 .+ (1:2)]], [sψ1[4]; sb1[6 .+ (1:2)]]) - [κ; 0.0; 0.0];
	 cone_product([ψ1[5]; b1[8 .+ (1:1)]], [sψ1[5]; sb1[8 .+ (1:1)]]) - [κ; 0.0];
    ]
end


nq = 5 # configuration dimension
nu = 2 # control dimension
nc = 5 # number of contact points

planarpush = PlanarPush(nq, nu, 0, nc,
			mass_block, mass_pusher, inertia,
			[μ_surface for i = 1:nc], μ_pusher,
			gravity,
			contact_corner_offset,
			dim_rnd,
			rnd)

using ForwardDiff
z0 = rand(num_var(planarpush)) 
θ = rand(num_data(planarpush))
res = a -> residual(planarpush, a, θ, [1.0])

res(rand(num_var(planarpush)))
ForwardDiff.jacobian(res, rand(num_var(planarpush)))