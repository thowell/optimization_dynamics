friction_coefficients(model::Cartpole{T}) where T = T[]

function RoboDojo.indices_z(model::Cartpole) 
    nq = model.nq 
    nc = model.nc
    q = collect(1:nq) 
    γ = collect(1:0)
    sγ = collect(1:0)
    ψ = collect(1:0) 
    b = collect(1:0) 
    sψ = collect(1:0) 
    sb = collect(1:0) 
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

RoboDojo.num_var(model::Cartpole) = model.nq


RoboDojo.nominal_configuration(model::Cartpole) = zeros(model.nq)

function RoboDojo.indices_optimization(model::Cartpole) 
    nq = model.nq
    nz = nq 
    IndicesOptimization(
        nz, 
        nz, 
        [collect(1:0), collect(1:0)],
        [collect(1:0), collect(1:0)],
        Vector{Vector{Vector{Int}}}(),
        Vector{Vector{Vector{Int}}}(),
        collect(1:(nq)),
        collect(1:0),
        collect(1:0),
        Vector{Vector{Int}}(),
        collect(1:0))
end

function RoboDojo.initialize_z!(z, model::Cartpole, idx::IndicesZ, q)
    z[idx.q] .= q
end