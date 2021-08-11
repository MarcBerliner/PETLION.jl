using PETLION
using Test

@testset "PETLION.jl" begin
    SAVE_SYMBOLIC_FUNCTIONS = PETLION.options[:SAVE_SYMBOLIC_FUNCTIONS]
    PETLION.options[:SAVE_SYMBOLIC_FUNCTIONS] = false

    opts = (
        temperature = false,
        N_p=10,
        N_n=10,
        N_s=10,
    )

    p_AD = Params(LCO; jacobian=:AD, opts...)
    p    = Params(LCO; jacobian=:symbolic, opts...)

    # AD matches symbolic
    @test run_model(p, 0:100:3600, I=-1,  SOC=1, outputs=(:t, :V)).V ≈ run_model(p_AD, 0:100:3600, I=-1,  SOC=1, outputs=(:t, :V)).V
    @test run_model(p, 0:100, P=-10, SOC=1).V ≈ run_model(p_AD, 0:100, P=-10, SOC=1).V
    @test run_model(p, 0:10, V=3.5, SOC=1).V ≈ run_model(p_AD, 0:10,  V=3.5, SOC=1).V

    # all outputs work
    run_model(p, I=-1, SOC=1, outputs=:all)

    # Cfunc is actually working
    p.opts.SOC=0
    @test run_model(p, I=1).V[end] ≠ run_model(p, I=(t)->cos(t)).V[end]
    @test run_model(p, P=100).V[end] ≠ run_model(p, P=(t)->100cos(t)).V[end]

    # Cfunc matches CC
    p.opts.SOC=0
    @test all(run_model(p, I=1).V .=== run_model(p, I=(t)->1).V)

    # :hold and I_max stop conditions are working
    model = run_model(p, 100, I=-0.1, SOC=1, outputs=(:t, :V))
    run_model!(model, p, 100, I=-0.1)
    run_model!(model, p, 100, V=:hold)
    run_model!(model, p, 100, V=:hold, I_max=-0.05)
    @test model.I[end] ≈ -0.05
    run_model!(model, p, 100, P=:hold)

    PETLION.options[:SAVE_SYMBOLIC_FUNCTIONS] = SAVE_SYMBOLIC_FUNCTIONS
end
