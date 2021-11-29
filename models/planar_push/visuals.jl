function _create_planar_push!(vis, model::PlanarPush;
        i = 1,
        r = 0.1,
        r_pusher = 0.025,
        tl = 1.0,
        box_color = Colors.RGBA(0.0, 0.0, 0.0, tl),
        pusher_color = Colors.RGBA(0.5, 0.5, 0.5, tl))

    r_box = r - r_pusher

    setobject!(vis["box_$i"], GeometryBasics.Rect(Vec(-1.0 * r_box,
		-1.0 * r_box,
		-1.0 * r_box),
		Vec(2.0 * r_box, 2.0 * r_box, 2.0 * r_box)),
		MeshPhongMaterial(color = box_color))

    setobject!(vis["pusher_$i"],
        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r_box), r_pusher),
        MeshPhongMaterial(color = pusher_color))
end

function _set_planar_push!(vis, model::PlanarPush, q;
    i = 1)
    settransform!(vis["box_$i"],
		compose(Translation(q[1], q[2], 0.01 * i), LinearMap(RotZ(q[3]))))
    settransform!(vis["pusher_$i"], Translation(q[4], q[5], 0.01 * i))
end

function RoboDojo.visualize!(vis, model::PlanarPush, q;
        i = 1,
        r = 0.1,
        r_pusher = 0.025,
        tl = 1.0,
        box_color = Colors.RGBA(0.0, 0.0, 0.0, tl),
        pusher_color = Colors.RGBA(0.5, 0.5, 0.5, tl),
        Δt = 0.1)

	default_background!(vis)

    _create_planar_push!(vis, model,
        i = i,
        r = r,
        r_pusher = r_pusher,
        tl = tl,
        box_color = box_color,
        pusher_color = pusher_color)

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T-1
        MeshCat.atframe(anim, t) do
            _set_planar_push!(vis, model, q[t])
        end
    end

	settransform!(vis["/Cameras/default"],
    compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
    setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)


    MeshCat.setanimation!(vis, anim)
end
