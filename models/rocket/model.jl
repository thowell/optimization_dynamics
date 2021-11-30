struct Rocket{T} <: Model{T}
      nq::Int
      nu::Int
      nw::Int
      nc::Int

      mass::T                               # mass
      inertia::Diagonal{T, Vector{T}}       # inertia matrix
      inertia_inv::Diagonal{T, Vector{T}}   # inertia matrix inverse
      gravity::Vector{T}                    # gravity
      length::T                             # length (com to thruster)
end

function f(model::Rocket, z, u, w)
      # states
      x = z[1:3]
      r = z[4:6]
      v = z[7:9]
      ω = z[10:12]

      # force in body frame
      F = u[1:3]

      # torque in body frame
      τ = [model.length * u[2],
           -model.length * u[1],
           0.0]

      [v;
       0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0*(ω' * r) * r);
       model.gravity + (1.0 / model.mass) * MRP(r[1], r[2], r[3]) * F;
       model.inertia_inv * (τ - cross(ω, model.inertia * ω))]
end

nq, nu, nw = 12, 3, 0

mass = 1.0
len = 1.0

inertia = Diagonal([1.0 / 12.0 * mass * len^2.0, 1.0 / 12.0 * mass * len^2.0, 1.0e-5])
inertia_inv = Diagonal([1.0 / (1.0 / 12.0 * mass * len^2.0), 1.0 / (1.0 / 12.0 * mass * len^2.0), 1.0 / (1.0e-5)])

rocket = Rocket(nq, nu, nw, 0,
                  mass,
                  inertia,
                  inertia_inv,
                  [0.0, 0.0, -9.81],
                  len)

