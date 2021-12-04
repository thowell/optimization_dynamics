using Plots
using Random
Random.seed!(1)

# ## least squares 
include("ls.jl")

struct MInfo{T}
    idx_q1::Vector{Int} 
    idx_q2::Vector{Int} 
    idx_u1::Vector{Int}
    v1::Vector{T} 
end

function _step(sim::Simulator, info::MInfo, q1, q2, u1) 
    info.v1 .= q2 
    info.v1 .-= q1 
    info.v1 ./= sim.h
    RoboDojo.step!(sim, q2, info.v1, u1, 1) 
end

struct GradientBundle{T}
    ls::LeastSquares{T}
    q1η::Vector{T} 
    q2η::Vector{T} 
    u1η::Vector{T} 
    dz::Matrix{T} 
    ny::Int 
    nz::Int
    info::MInfo{T}
end

function GradientBundle(model; ϵ=1.0e-4) 
    nx = 2 * model.nq 
    ny = model.nq
    nu = model.nu 
    nz = nx + nu 
    nθ = ny * (nz)

    @variables fz[1:ny] fη[1:ny] η[1:nz] θ[1:nθ] 

    function cost(fz, fη, η, θ) 
        M = reshape(θ, ny, nz) 
        r = fη - fz - M * η 
        return [transpose(r) * r]
    end

    c = cost(fz, fη, η, θ)
    cθ = Symbolics.gradient(c[1], θ)
    cθθ = Symbolics.hessian(c[1], θ)

    c_func_ls = eval(Symbolics.build_function(c, fz, fη, η, θ)[2])
    cθ_func_ls = eval(Symbolics.build_function(cθ, fz, fη, η, θ)[2])
    cθθ_func_ls = eval(Symbolics.build_function(cθθ, fz, fη, η, θ)[2])

    N = 100# 2 * nz
    η = [zeros(nz) for i = 1:N]

    for i = 1:N 
        w = ϵ * randn(1)[1]
        η[i][rand([j for j = 1:nz], 1)[1]] = w 
    end

    # for i = 1:nz 
    #     η[i][i] = ϵ 
    #     η[i+nz][i] = -ϵ
    # end 

    info = MInfo(
        collect(1:model.nq), 
        collect(model.nq .+ (1:model.nq)),
        collect(2 * model.nq .+ (1:model.nu)),
        zeros(model.nq))

    fz = zeros(model.nq)
    fη = [zeros(model.nq) for i = 1:N]

    ls = LeastSquares(N, 
        fz, fη, η, 
        c_func_ls, cθ_func_ls, cθθ_func_ls, 
        [0.0], zeros(nθ), zeros(nθ, nθ), 
        [0.0], zeros(nθ), zeros(nθ, nθ), 
        zeros(nθ), zeros(nθ),
        lu_solver(zeros(nθ, nθ)))

    GradientBundle(ls,
        zeros(nq),
        zeros(nq),
        zeros(nu),
        zeros(ny, nz),
        ny, nz,
        info)
end

function gradient!(sim::Simulator, gb::GradientBundle, q1, q2, u1; verbose=false)
    gb.ls.fz .= _step(sim, gb.info, q1, q2, u1) 
    for i = 1:gb.ls.N
        ηq1 = @views gb.ls.η[i][gb.info.idx_q1] 
        ηq2 = @views gb.ls.η[i][gb.info.idx_q2] 
        ηu1 = @views gb.ls.η[i][gb.info.idx_u1]
        gb.q1η .= q1 
        gb.q1η .+= ηq1
        gb.q2η .= q2 
        gb.q2η .+= ηq2 
        gb.u1η .= u1
        gb.u1η .+= ηu1
        gb.ls.fη[i] .= _step(sim, gb.info, gb.q1η, gb.q2η, gb.u1η)
    end
    update!(gb.ls, verbose=verbose)
    gb.dz .= Base.ReshapedArray(gb.ls.θ, (gb.ny, gb.nz), ())
    return gb.dz
end

# gradient!(im_dyn.eval_sim, gb, q1, q2, u1)
# @benchmark gradient!($im_dyn.eval_sim, $gb, $q1, $q2, $u1)

function fx_gb(dx, model::ImplicitDynamics, x, u, w)
	q1 = @views x[model.idx_q1]
	q2 = @views x[model.idx_q2]
	model.v1 .= q2 
	model.v1 .-= q1 
	model.v1 ./= model.eval_sim.h
	RoboDojo.step!(model.eval_sim, q2, model.v1, u, 1)
	nq = model.eval_sim.model.nq
	for i = 1:nq
		dx[model.idx_q1[i], model.idx_q2[i]] = 1.0
	end
    gradient!(model.eval_sim, model.info, q1, q2, u)
    ∂q3∂q1 = @views model.info.dz[model.idx_q1, model.idx_q1] 
    ∂q3∂q2 = @views model.info.dz[model.idx_q1, model.idx_q2] 
	dx[model.idx_q2, model.idx_q1] = ∂q3∂q1
	dx[model.idx_q2, model.idx_q2] = ∂q3∂q2
	return dx
end

# dx = zeros(2 * planarpush.nq, 2 * planarpush.nq)
# x = rand(2 * planarpush.nq)
# u = rand(planarpush.nu)
# w = zeros(0)
# fx_gb(dx, im_dyn, x, u, w)
# @benchmark fx_gb($dx, $im_dyn, $x, $u, $w)
# @code_warntype fx_gb(dx, im_dyn, x, u, w)

function fu_gb(du, model::ImplicitDynamics, x, u, w)
	q1 = @views x[model.idx_q1]
	q2 = @views x[model.idx_q2]
	model.v1 .= q2 
	model.v1 .-= q1 
	model.v1 ./= model.eval_sim.h
	RoboDojo.step!(model.eval_sim, q2, model.v1, u, 1)
    gradient!(model.eval_sim, model.info, q1, q2, u)
    ∂q3∂u1 = @views model.info.dz[model.idx_q1, model.info.info.idx_u1] 
	du[model.idx_q2, :] = ∂q3∂u1
	return du
end

# du = zeros(2 * planarpush.nq, planarpush.nu)
# x = rand(2 * planarpush.nq)
# u = rand(planarpush.nu)
# w = zeros(0)
# fu_gb(du, im_dyn, x, u, w)
# @benchmark fu_gb($du, $im_dyn, $x, $u, $w)
# @code_warntype fu_gb(du, im_dyn, x, u, w)











