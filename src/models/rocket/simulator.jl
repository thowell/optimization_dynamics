friction_coefficients(model::Rocket{T}) where T = T[]

function RoboDojo.indices_z(model::Rocket) 
    nq = model.nq 
    q = collect(1:nq) 
    γ = collect(1:0)
    sγ = collect(1:0)
    ψ = collect(1:0) 
    b = collect(1:0) 
    sψ = collect(1:0) 
    sb = collect(1:0) 
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

function RoboDojo.indices_θ(model::Rocket; nf=0) 
    nq = model.nq 
    nu = model.nu 
    nw = model.nw 

    q1 = collect(1:nq)
    q2 = collect(1:0)
    u = collect(nq .+ (1:nu))
    w = collect(1:0)
    f = collect(1:0)
    h = collect(nq + nu .+ (1:1))

    Indicesθ(q1, q2, u, w, f, h) 
end

RoboDojo.num_var(model::Rocket) = model.nq

RoboDojo.nominal_configuration(model::Rocket) = zeros(model.nq)

function RoboDojo.indices_optimization(model::Rocket) 
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

function RoboDojo.initialize_z!(z, model::Rocket, idx::IndicesZ, q)
    z[idx.q] .= q
end