
struct Rocket{T} <: Model{T}
      nq::Int
      mu::Int
      nw::Int

      mass::T                               # mass
      inertia::Diagonal{T, Vector{T}}       # inertia matrix
      inertia_inv::Diagonal{T, Vector{T}}   # inertia matrix inverse
      gravity::Vector{T}                    # gravity
      length::T                             # length (com to thruster)
end

function f(model::Rocket, z, u, w)
      # states
      x = view(z,1:3)
      r = view(z,4:6)
      v = view(z,7:9)
      ω = view(z,10:12)

      # force in body frame
      F = view(u, 1:3)

      # torque in body frame
      τ = @SVector [model.length * u[2],
                    -model.length * u[1],
                    0.0]

      SVector{12}([v;
                   0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0*(ω' * r) * r);
                   model.gravity + (1.0 / model.mass) * MRP(r[1], r[2], r[3]) * F;
                   model.inertia_inv * (τ - cross(ω, model.inertia * ω))])
end

n, m, d = 12, 3, 0

mass = 1.0
len = 1.0

inertia = Diagonal([1.0 / 12.0 * mass * len^2.0, 1.0 / 12.0 * mass * len^2.0, 1.0e-5])
inertia_inv = Diagonal([1.0 / (1.0 / 12.0 * mass * len^2.0), 1.0 / (1.0 / 12.0 * mass * len^2.0), 1.0 / (1.0e-5)])

model = Rocket(n, m, d,
                  mass,
                  inertia,
                  inertia_inv,
                  [0.0, 0.0, -9.81],
                  len)

