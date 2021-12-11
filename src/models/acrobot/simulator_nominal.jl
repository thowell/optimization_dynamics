function RoboDojo.indices_z(model::DoublePendulum{T,Nominal}) where T 
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

RoboDojo.num_var(model::DoublePendulum{T,Nominal}) where T = model.nq

function RoboDojo.indices_optimization(model::DoublePendulum{T,Nominal}) where T
    nq = model.nq
    nz = nq
    IndicesOptimization(
        nz, 
        nz, 
        [collect(1:0), collect(1:0)],
        [collect(1:0), collect(1:0)],
        Vector{Vector{Vector{Int}}}(),
        Vector{Vector{Vector{Int}}}(),
        collect(1:nq),
        collect(1:0),
        collect(1:0),
        Vector{Vector{Int}}(),
        collect(1:0))
end

function RoboDojo.initialize_z!(z, model::DoublePendulum{T,Nominal}, idx::IndicesZ, q) where T
    z[idx.q] .= q
end
