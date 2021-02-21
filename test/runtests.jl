using PETLION
using Test

@testset "PETLION.jl" begin

    opts = (
        cathode=LCO,
        anode=LiC6,
        temperature = false,
        N_p=10,
        N_n=10,
        N_s=10,
    )

    p_AD = Params(;jacobian = :AD, opts...)
    p = Params(;jacobian = :symbolic, opts...)

    # AD matches symbolic
    @test run_model(p, 0:100:3600, I=-1,  SOC=1, outputs=(:t, :V)).V ≈ run_model(p_AD, 0:100:3600, I=-1,  SOC=1, outputs=(:t, :V)).V
    @test run_model(p, 0:100, P=-10, SOC=1).V ≈ run_model(p_AD, 0:100, P=-10, SOC=1).V
    @test run_model(p, 0:10,  V=3.5, SOC=1).V ≈ run_model(p_AD, 0:10,  V=3.5, SOC=1).V

    # all outputs work
    run_model(p, I=-1, SOC=1, outputs=:all)

    # Cfunc is actually working
    @test run_model(p, I=1, SOC=0).V[end] ≠ run_model(p, I=(u,p,t)->cos(t), SOC=0).V[end]

    # Cfunc matches CC
    @test all(run_model(p, I=1, SOC=0).V .=== run_model(p, I=(x...)->1, SOC=0).V)

    # :hold and I_max stop conditions are working
    model = run_model(p, 100, I=-0.1, SOC=1, outputs=(:t, :V))
    run_model!(model, p, 100, I=-0.1)
    run_model!(model, p, 100, V=:hold)
    run_model!(model, p, 100, V=:hold, I_max=-0.05)
    @test model.I[end] ≈ -0.05
    run_model!(model, p, 100, P=:hold)
end
