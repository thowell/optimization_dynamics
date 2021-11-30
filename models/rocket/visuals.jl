function visualize!(vis, p::Rocket, q; Δt = 0.1, mesh = true, T_off = length(q))
	default_background!(vis)

	if mesh
		obj_rocket = joinpath(pwd(), "models/starship/Starship.obj")
		mtl_rocket = joinpath(pwd(), "models/starship/Starship.mtl")
		ctm = ModifiedMeshFileObject(obj_rocket, mtl_rocket, scale=1.0)
		setobject!(vis["rocket"]["starship"], ctm)

		settransform!(vis["rocket"]["starship"],
			compose(Translation(0.0, 0.0, -p.length),
				LinearMap(0.25 * RotY(0.0) * RotZ(0.5 * π) * RotX(0.5 * π))))

        body = Cylinder(Point3f0(0.0, 0.0, -1.25),
          Point3f0(0.0, 0.0, 0.5),
          convert(Float32, 0.125))

        setobject!(vis["rocket"]["body"], body,
          MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
	else
		body = Cylinder(Point3f0(0.0, 0.0, -1.0 * model.length),
			Point3f0(0.0, 0.0, 1.0 * model.length),
			convert(Float32, 0.15))

		setobject!(vis["rocket"], body,
			MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
			anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
	end

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	for t = 1:length(q)
	    MeshCat.atframe(anim, t) do
            if t >= T_off
				setvisible!(vis["rocket"]["body"], false)
			else
				setvisible!(vis["rocket"]["body"], true)
			end
	        settransform!(vis["rocket"],
	              compose(Translation(q[t][1:3]),
	                    LinearMap(MRP(q[t][4:6]...) * RotX(0.0))))
	    end
	end

	MeshCat.setanimation!(vis, anim)
end