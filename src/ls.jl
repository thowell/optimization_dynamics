using Symbolics
using BenchmarkTools
using InteractiveUtils

struct LeastSquares{T}
    N::Int
    fz::Vector{T} 
    fη::Vector{Vector{T}}
    η::Vector{Vector{T}}
    cost::Function
    grad::Function 
    hess::Function 
    val_cost::Vector{T}
    val_grad::Vector{T} 
    val_hess::Matrix{T}
    cache_cost::Vector{T}
    cache_grad::Vector{T} 
    cache_hess::Matrix{T}
    θ::Vector{T} 
    Δ::Vector{T}
    solver::RoboDojo.LinearSolver
end

function eval_cost!(ls::LeastSquares) 
    fill!(ls.val_cost, 0.0)
    for i = 1:ls.N 
        ls.cost(ls.cache_cost, ls.fz, ls.fη[i], ls.η[i], ls.θ)
        ls.val_cost .+= ls.cache_cost
    end
end

function eval_grad!(ls::LeastSquares) 
    fill!(ls.val_grad, 0.0)
    for i = 1:ls.N 
        ls.grad(ls.cache_grad, ls.fz, ls.fη[i], ls.η[i], ls.θ)
        ls.val_grad .+= ls.cache_grad
    end
end

function eval_hess!(ls::LeastSquares) 
    fill!(ls.val_hess, 0.0)
    for i = 1:ls.N 
        ls.hess(ls.cache_hess, ls.fz, ls.fη[i], ls.η[i], ls.θ)
        ls.val_hess .+= ls.cache_hess
    end
end

function update!(ls::LeastSquares; tol=1.0e-8, verbose=true) 
    iter = 0
    eval_grad!(ls) 
    res = norm(ls.val_grad, Inf) 
    verbose && (@show res)
    while res > tol && iter < 100
        eval_hess!(ls)
        fill!(ls.Δ, 0.0)
        RoboDojo.linear_solve!(ls.solver, ls.Δ, ls.val_hess, ls.val_grad)
        ls.θ .-= ls.Δ
        eval_grad!(ls) 
        res = norm(ls.val_grad, Inf)
        verbose && (@show res) 
        verbose && (@show ls.θ)
        iter += 1
    end
end

# ## test 
# function f(z) 
#     x = z[1:2] 
#     u = z[2 .+ (1:1)] 
#     A = [1.0 1.0; 0.0 1.0] 
#     B = [0.0; 1.0] 

#     A * x + B * u[1]
# end

# nx = 2 
# nu = 1 
# nz = nx + nu 
# nθ = nx * (nz)

# @variables fz[1:nx] fη[1:nx] η[1:nz] θ[1:nθ] 

# function cost(fz, fη, η, θ) 
#     M = reshape(θ, nx, nz) 
#     r = fη - fz - M * η 
#     return [transpose(r) * r]
#     return [transpose(θ) * θ]
# end

# c = cost(fz, fη, η, θ)
# cθ = Symbolics.gradient(c[1], θ)
# cθθ = Symbolics.hessian(c[1], θ)

# c_func = eval(Symbolics.build_function(c, fz, fη, η, θ)[2])
# cθ_func = eval(Symbolics.build_function(cθ, fz, fη, η, θ)[2])
# cθθ_func = eval(Symbolics.build_function(cθθ, fz, fη, η, θ)[2])

# fz0 = rand(nx)
# fη0 = rand(nx) 
# η0 = rand(nz) 
# θ0 = rand(nθ)
# θθ0 = rand(nθ, nθ)

# c0 = zeros(1)
# cθ0 = zeros(nθ)
# cθθ0 = zeros(nθ, nθ)

# c_func(c0, fz0, fη0, η0, θ0)
# @benchmark c_func($c0, $fz0, $fη0, $η0, $θ0)

# cθ_func(cθ0, fz0, fη0, η0, θ0)
# @benchmark cθ_func($cθ0, $fz0, $fη0, $η0, $θ0)

# cθθ_func(cθθ0, fz0, fη0, η0, θ0)
# @benchmark cθ_func($cθθ0, $fz0, $fη0, $η0, $θθ0)

# N = 2 * nz
# η = [zeros(nz) for i = 1:N]
# ϵ = 0.1
# for i = 1:nz 
#     η[i][i] = ϵ 
#     η[i+nz][i] = -ϵ
# end

# z0 = rand(nz)
# fz = f(z0) 
# fη = [f(z0 + η[i]) for i = 1:N]

# ls = LeastSquares(N, 
#     fz, fη, η, 
#     c_func, cθ_func, cθθ_func, 
#     [0.0], zeros(nθ), zeros(nθ, nθ), 
#     [0.0], zeros(nθ), zeros(nθ, nθ), 
#     zeros(nθ), zeros(nθ),
#     lu_solver(zeros(nθ, nθ)))

# # eval_cost!(ls, z0)
# # @benchmark eval_cost!($ls)

# # eval_grad!(ls, z0)
# # @benchmark eval_grad!($ls)

# # eval_hess!(ls, z0)
# # @benchmark eval_hess!($ls)

# update!(ls)
# reshape(ls.θ, nx, nz)
# # @benchmark update!($ls, z) setup=(z=copy(z0))











