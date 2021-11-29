friction_coefficients(model::Cartpole) = model.friction

function RoboDojo.indices_z(model::Cartpole) 
    nq = model.nq 
    nc = model.nc
    q = collect(1:nq) 
    γ = collect(nq .+ (1:0)) 
    sγ = collect(nq .+ (1:0))
    ψ = collect(nq .+ (1:nc)) 
    b = collect(nq + nc .+ (1:nc)) 
    sψ = collect(nq + nc + nc .+ (1:nc)) 
    sb = collect(nq + nc + nc + nc .+ (1:nc)) 
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

RoboDojo.num_var(model::Cartpole) = model.nq + 4 * model.nc

RoboDojo.nominal_configuration(model::Cartpole) = zeros(model.nq)

function RoboDojo.indices_optimization(model::Cartpole) 
    nq = model.nq
    nc = model.nc
    nz = nq + 4 * nc
    IndicesOptimization(
        nz, 
        nz, 
        [collect(1:0), collect(1:0)],
        [collect(1:0), collect(1:0)],
        [[collect([nq + i, nq + nc + i]), collect([nq + nc + nc + i, nq + nc + nc + nc + i])] for i = 1:nc], 
        [[collect([nq + i, nq + nc + i]), collect([nq + nc + nc + i, nq + nc + nc + nc + i])] for i = 1:nc], 
        collect(1:(nq + 2 * nc)),
        collect(1:0),
        collect(nq + 2 * nc .+ (1:(2 * nc))),
        [collect(nq + 2 * nc + (i - 1) * 2 .+ (1:2)) for i = 1:nc],
        collect(nq + 2 * nc .+ (1:(2 * nc))))
end

function RoboDojo.initialize_z!(z, model::Cartpole, idx::IndicesZ, q)
    z[idx.q] .= q
    z[idx.ψ] .= 1.0
    z[idx.b] .= 0.1
    z[idx.sψ] .= 1.0
    z[idx.sb] .= 0.1
end