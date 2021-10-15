![PETLION_logo](docs/logo/PETLION_medium.png)


# PETLION – Porous Electrode Theory for Li-ion Batteries

High-performance simulations of the pseudo-2D porous electrode theory (PET) model in Julia
+ Built for efficient controls, parameter estimation, and other complex battery simulations using the rigorous PET model
+ Runs a full charge or discharge with 301 DAEs in ~3 ms on a laptop with 1 MB total memory usage
+ Includes thermal and aging modes

# Installation
After [installing Julia](https://julialang.org/downloads/), run the following command to add the PETLION package
```julia
julia> import Pkg; Pkg.add("PETLION")
```

# Getting started
To [get started](examples/getting_started.ipynb), we recommend checking out the [list of examples](examples). To simulate a simple [constant current-constant voltage (CC-CV) charge](examples/CC-CV.ipynb), run the following:
```julia
using PETLION
p = Params(LCO)

model = run_model(p, I=2, SOC=0, V_max=4.1)
run_model!(model, p, 1800, V=:hold, I_min=1/20)
```
<img src="https://raw.githubusercontent.com/MarcBerliner/PETLION.jl/master/docs/example_pictures/CCCV_I.png" width="50%"><img src="https://raw.githubusercontent.com/MarcBerliner/PETLION.jl/master/docs/example_pictures/CCCV_V.png" width="50%">
```julia
PETLION model
  --------
  Runs:    I → V
  Time:    2445.64 s
  Voltage: 4.1 V
  Current: 0.1921C
  Power:   23.023 W
  SOC:     1.0
  Exit:    Above maximum SOC limit
```

# Credits
+ [Marc D. Berliner](https://marcberliner.com/) – Creating and maintaining the code
+ [Richard D. Braatz](https://cheme.mit.edu/profile/richard-d-braatz/) – Technical oversight
+ [Richard B. Canty](https://scholar.google.com/citations?user=MqAWccAAAAAJ&hl=en) – Designing the PETLION logo

# Citations
If you use PETLION in your work, please cite [the paper](https://iopscience.iop.org/article/10.1149/1945-7111/ac201c):
```bibtex
@article{berliner2021petlion,
  title={Methods---{PETLION}: Open-Source Software for Millisecond-Scale Porous Electrode Theory-Based Lithium-Ion Battery Simulations},
  author={Berliner, Marc D and Cogswell, Daniel A and Bazant, Martin Z and Braatz, Richard D},
  journal={Journal of The Electrochemical Society},
  volume={168},
  number={9},
  pages={090504},
  year={2021},
  publisher={IOP Publishing}
}
```

# See also
Check out these high-quality and open-source battery simulation tools
+ [LIONSIMBA](https://github.com/lionsimbatoolbox/LIONSIMBA)
+ [MPET](https://bitbucket.org/bazantgroup/mpet/)
+ [PyBaMM](https://github.com/pybamm-team/PyBaMM)
