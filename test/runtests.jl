using PorousElectrodes
using Test

@testset "PorousElectrodes.jl" begin

    opts = (
        cathode=LCO,
        anode=LiC6,
        temperature = false,
        N_p=10,
        N_n=10,
        N_s=10,
    )

    p_AD = Params(;jacobian = :AD, opts...)
    p_sym = Params(;jacobian = :symbolic, opts...)

    # AD matches symbolic
    @assert run_model(p_sym, 0:100:3600, I=-1,  SOC=1, outputs=(:t, :V)).V ≈ run_model(p_AD, 0:100:3600, I=-1,  SOC=1, outputs=(:t, :V)).V
    @assert run_model(p_sym, 0:100, P=-10, SOC=1).V ≈ run_model(p_AD, 0:100, P=-10, SOC=1).V
    @assert run_model(p_sym, 0:10,  V=3.5, SOC=1).V ≈ run_model(p_AD, 0:10,  V=3.5, SOC=1).V

    # all outputs work
    run_model(p_sym, I=-1, SOC=1, outputs=:all)

    # Cfunc is actually working
    @assert run_model(p_sym, I=1, SOC=0).V[end] ≠ run_model(p_sym, I=(u,p,t)->cos(t), SOC=0).V[end]

    # Cfunc matches CC
    @assert all(run_model(p_sym, I=1, SOC=0).V .=== run_model(p_sym, I=(x...)->1, SOC=0).V)

    # :hold and I_max stop conditions are working
    model = run_model(p, 100, I=-0.1, SOC=1, outputs=(:t, :V))
    run_model!(model, p, 100, I=-0.1)
    run_model!(model, p, 100, V=:hold)
    run_model!(model, p, 100, V=:hold, I_max=-0.05)
    @assert model.I[end] ≈ -0.05
    run_model!(model, p, 100, P=:hold)
end
