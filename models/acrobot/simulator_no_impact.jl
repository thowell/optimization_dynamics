function RoboDojo.indices_z(model::DoublePendulum) 
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

RoboDojo.num_var(model::DoublePendulum) = model.nq

RoboDojo.nominal_configuration(model::DoublePendulum) = zeros(model.nq)

function RoboDojo.indices_optimization(model::DoublePendulum) 
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

function RoboDojo.initialize_z!(z, model::DoublePendulum, idx::IndicesZ, q)
    z[idx.q] .= q
end

friction_coefficients(model::DoublePendulum{T}) where T = T[]
