function RoboDojo.indices_z(model::DoublePendulum{T,Impact}) where T 
    nq = model.nq 
    nc = model.nc
    q = collect(1:nq) 
    γ = collect(nq .+ (1:2)) 
    sγ = collect(nq + 2 .+ (1:2))
    ψ = collect(1:0) 
    b = collect(1:0)
    sψ = collect(1:0)
    sb = collect(1:0)
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

RoboDojo.num_var(model::DoublePendulum{T,Impact}) where T = model.nq + 2 * model.nc

function RoboDojo.indices_optimization(model::DoublePendulum{T,Impact}) where T 
    nq = model.nq
    nc = model.nc
    nz = nq + 2 * nc
    RoboDojo.IndicesOptimization(
        nz, 
        nz, 
        [collect(nq .+ (1:2)), collect(nq + 2 .+ (1:2))],
        [collect(nq .+ (1:2)), collect(nq + 2 .+ (1:2))],
        Vector{Vector{Vector{Int}}}(),
        Vector{Vector{Vector{Int}}}(),
        collect(1:(nq + nc)),
        collect(nq + nc .+ (1:nc)),
        collect(1:0),
        Vector{Vector{Int}}(),
        collect(nq + nc .+ (1:nc)))
end

function RoboDojo.initialize_z!(z, model::DoublePendulum{T,Impact}, idx::RoboDojo.IndicesZ, q) where T
    z[idx.q] .= q
    z[idx.γ] .= 1.0
    z[idx.sγ] .= 1.0
end

RoboDojo.nominal_configuration(model::DoublePendulum) = zeros(model.nq)
friction_coefficients(model::DoublePendulum{T}) where T = T[]
