using Literate

function preprocess(str)
    str = replace(str, "# PREAMBLE" => "")
    str = replace(str, "# PKG_SETUP" => "")
    return str
end

exampledir = @__DIR__
Literate.notebook(joinpath(exampledir, "acrobot.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "cartpole.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "hopper.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "planar_push.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "rocket.jl"), exampledir, execute=false, preprocess=preprocess)

