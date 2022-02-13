# Examples

The `.jl` files in each subdirectory are meant to be processed using [Literate.jl](https://github.com/fredrikekre/Literate.jl).
During the build process, the `.jl` files are converted to notebooks. 

1. [install optimization_dynamics](https://github.com/thowell/optimization_dynamics)
2. [install IJulia](https://github.com/JuliaLang/IJulia.jl) (`add` it to the default project)
3. interact with notebooks
   ```
   using IJulia, OptimizationDynamics
   notebook(dir=joinpath(dirname(pathof(OptimizationDynamics)), "..", "examples"))
   ```