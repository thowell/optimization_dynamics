function RoboDojo.indices_z(model::PlanarPush) 
    nq = model.nq 
    nc = model.nc
    q = collect(1:nq) 
    γ = collect(nq .+ (1:1)) 
    sγ = collect(nq + 1 .+ (1:1))
    ψ = collect(nq + 2 .+ (1:5)) 
    b = collect(nq + 2 + 5 .+ (1:9)) 
    sψ = collect(nq + 2 + 5 + 9 .+ (1:5)) 
    sb = collect(nq + 2 + 5 + 9 + 5 .+ (1:9)) 
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

RoboDojo.nominal_configuration(model::PlanarPush) = [-1.0; 0.0; 0.0; 0.0; 0.0]
function RoboDojo.indices_optimization(model::PlanarPush) 
    nq = model.nq
    nz = num_var(model)
    IndicesOptimization(
        nz, 
        nz, 
        [collect(nq .+ (1:1)), collect(nq + 1 .+ (1:1))],
        [collect(nq .+ (1:1)), collect(nq + 1 .+ (1:1))],
        [
         [collect([collect(nq + 2 .+ (1:1)); collect(nq + 2 + 5 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 .+ (1:2))])], 
         [collect([collect(nq + 2 + 1 .+ (1:1)); collect(nq + 2 + 5 + 2 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 1 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 2 .+ (1:2))])], 
         [collect([collect(nq + 2 + 2 .+ (1:1)); collect(nq + 2 + 5 + 4 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 2 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 4 .+ (1:2))])], 
         [collect([collect(nq + 2 + 3 .+ (1:1)); collect(nq + 2 + 5 + 6 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 3 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 6 .+ (1:2))])], 
         [collect([collect(nq + 2 + 4 .+ (1:1)); collect(nq + 2 + 5 + 8 .+ (1:1))]), collect([collect(nq + 2 + 5 + 9 + 4 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 8 .+ (1:1))])], 
        ],
        [
         [collect([collect(nq + 2 .+ (1:1)); collect(nq + 2 + 5 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 .+ (1:2))])], 
         [collect([collect(nq + 2 + 1 .+ (1:1)); collect(nq + 2 + 5 + 2 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 1 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 2 .+ (1:2))])], 
         [collect([collect(nq + 2 + 2 .+ (1:1)); collect(nq + 2 + 5 + 4 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 2 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 4 .+ (1:2))])], 
         [collect([collect(nq + 2 + 3 .+ (1:1)); collect(nq + 2 + 5 + 6 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 3 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 6 .+ (1:2))])], 
         [collect([collect(nq + 2 + 4 .+ (1:1)); collect(nq + 2 + 5 + 8 .+ (1:1))]), collect([collect(nq + 2 + 5 + 9 + 4 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 8 .+ (1:1))])], 
        ],
        collect(1:(nq + 15)),
        collect(nq + 15 .+ (1:1)),
        collect(nq + 15 + 1 .+ (1:14)),
        [
         collect(nq + 15 + 1 .+ (1:3)), 
         collect(nq + 15 + 4 .+ (1:3)), 
         collect(nq + 15 + 7 .+ (1:3)), 
         collect(nq + 15 + 10 .+ (1:3)), 
         collect(nq + 15 + 13 .+ (1:2)), 
        ],
        collect(nq + 15 .+ (1:15))
        )
end

function RoboDojo.initialize_z!(z, model::PlanarPush, idx::IndicesZ, q)
    z[idx.q] .= q 
	z[idx.γ] .= 1.0 
	z[idx.sγ] .= 1.0
    z[idx.ψ] .= 1.0
    z[idx.b] .= 0.1
    z[idx.sψ] .= 1.0
    z[idx.sb] .= 0.1
end

RoboDojo.indices_optimization(planarpush)