![PETLION_logo](docs/logo/PETLION_small.png)


# PETLION – Porous Electrode Theory for Li-ion Batteries

PETLION is an open-source, high-performance computing implementation of the porous electrode theory (PET) model in Julia. A typical runtime for a dynamic simulation of full charge or discharge with 301 DAEs is 3 ms on a laptop while allocating about 1 MB of total memory. PETLION is built for efficient parameter estimation, controls, and other complex battery simulations using the rigorous PET model.

# Installation
After [installing Julia](https://julialang.org/downloads/), run the following command to add the PETLION package
```julia
julia> import Pkg; Pkg.add("PETLION")
```

# Getting started
To [get started](examples/getting_started.ipynb), we recommend checking out the [list of examples](examples) (more being added soon). To simulate a simple [constant current-constant voltage (CC-CV) charge](examples/CC-CV.ipynb), run the following:
```julia
using PETLION

p = Params(LCO)

model = run_model(p, I=2, SOC=0, V_max=4.1)
run_model!(model, p, V=:hold, I_min=1/20)
```
<img src="https://raw.githubusercontent.com/MarcBerliner/PETLION.jl/master/docs/example_pictures/CCCV_I.png" width="50%"><img src="https://raw.githubusercontent.com/MarcBerliner/PETLION.jl/master/docs/example_pictures/CCCV_V.png" width="50%">
```julia
PETLION model
  --------
  Runs:    I → V
  Time:    2444.59 s
  Voltage: 4.1 V
  Current: 0.1921C
  Power:   23.0214 W
  SOC:     1.0
  Exit:    Above maximum SOC limit
```

# Citations
If you use PETLION in your work, please cite:
(coming soon)

# Credits
+ [Marc D. Berliner](https://marcberliner.com/) – Creating and maintaining the code.
+ [Richard D. Braatz](https://cheme.mit.edu/profile/richard-d-braatz/) – Technical oversight.
+ [Richard B. Canty](https://scholar.google.com/citations?user=MqAWccAAAAAJ&hl=en) – Designing the PETLION logo.

# See also
Check out these high-quality and open-source battery simulation tools
+ [LIONSIMBA](https://github.com/lionsimbatoolbox/LIONSIMBA)
+ [MPET](https://bitbucket.org/bazantgroup/mpet/)
+ [PyBaMM](https://github.com/pybamm-team/PyBaMM)
