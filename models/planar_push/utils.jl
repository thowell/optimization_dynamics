# directly add methods instead of importing from RoboDojo.jl 

function lagrangian_derivatives(mass_matrix, dynamics_bias, q, v)
    D1L = -1.0 * dynamics_bias(q, v)
    D2L = mass_matrix(q) * v
    return D1L, D2L
end

cone_product(a, b) = [dot(a, b); a[1] * b[2:end] + b[1] * a[2:end]]
