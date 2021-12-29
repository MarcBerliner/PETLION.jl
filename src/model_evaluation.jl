@inline function get_run(
    I::current,
    V::voltage,
    P::power,
    t0, tf::Q, p::param, model::model_output) where {
    current  <: Union{Number,Symbol,Function,Nothing},
    voltage  <: Union{Number,Symbol,Function,Nothing},
    power    <: Union{Number,Symbol,Function,Nothing},
    Q,
    }

    if !( sum(!(method === Nothing) for method in (current,voltage,power)) === 1 )
        error("Cannot select more than one input")
    end

    if     !(current === Nothing)
        method, input, name = method_I(), I, "I"
    elseif !(voltage === Nothing)
        method, input, name = method_V(), V, "V"
    elseif !(power === Nothing)
        method, input, name = method_P(), P, "P"
    else
        error("Method not supported")
    end

    if input isa Function
        input = redefine_func(input)
    end

    value = [0.0]
    tf = (Q === Nothing ? 1e6 : (@inbounds Float64(tf[end])))
    
    run_type = run_determination(method, input)
    run      = run_type(input,value,method,t0,tf,name,run_info())

    return run
end

@inline run_determination(::AbstractMethod,::Any)      = run_constant
@inline run_determination(::AbstractMethod,::Function) = run_function

function custom_res!(p::param,x::T,func_RHS::Q,model::model_output;kw...) where {T<:Function,Q<:Function}
    func = redefine_func(x)
    p.θ[:_residual_val] = 0.0
    return (t,Y,YP,p) -> func_RHS(t,Y,YP,p) - func(t,Y,YP,p)
end
function custom_res!(p::param,x::T,func_RHS::Q,model::model_output;kw...) where {T<:Number,Q<:Function}
    p.θ[:_residual_val] = Float64(x)
    return func_RHS
end
function custom_res!(p::param,x::T,func_RHS::Q,model::model_output;hold_val::Number=0.0,kw...) where {T<:Symbol,Q<:Function}
    if check_is_hold(x,model)
        p.θ[:_residual_val] = hold_val
    end
    return func_RHS
end

@inline within_bounds(run::AbstractRun) = run.info.flag === -1

@inline function initialize_model!(model::model_struct, p::param{jac}, run::T, bounds::boundary_stop_conditions, opts::AbstractOptionsModel, res_I_guess=nothing) where {jac<:AbstractJacobian,method<:AbstractMethod,T<:AbstractRun{method,<:Any},model_struct<:Union{model_output,Vector{Float64}}}
    if !haskey(p.funcs,run)
        get_method_funcs!(p,run)
        funcs = p.funcs(run)
    end
    funcs = p.funcs(run)
    θ_tot = funcs.J_full.θ_tot
    update_θ!(θ_tot, funcs.J_full.θ_keys, p.θ)
    if jac === jacobian_AD; (@views @inbounds (p.cache.θ_tot .= θ_tot[1:length(p.cache.θ_tot)])) end
    
    cache = p.cache
    
    initial_guess! = p.funcs.initial_guess!
    keep_Y = opts.var_keep.Y
    
    # update the θ_tot vector from the dict p.θ
    check_errors_parameters_runtime(p, opts)
    
    # if this is a new model?
    new_run = model_struct === Array{Float64,1} || isempty(model)

    ## initializing the states vector Y and time t
    Y0 = keep_Y ? zeros(Float64,length(cache.Y0)) : cache.Y0
    YP0 = cache.YP0
    if model_struct === Array{Float64,1}
        Y0 = deepcopy(model)
        model = model_output()
        SOC = calc_SOC(Y0, p)
    elseif new_run
        SOC = opts.SOC
        if opts.reinit
            initial_guess!(Y0, SOC, θ_tot, res_I_guess)
        end
    else # continue from previous simulation
        Y0 .= @inbounds keep_Y ? copy(model.Y[end]) : model.Y[end]
        SOC = @inbounds model.SOC[end]
    end

    initial_current!(Y0,YP0,p,run,model,res_I_guess)
    
    ## getting the DAE integrator function
    initialize_states!(p,Y0,YP0,run,opts,funcs,SOC)
    
    int = retrieve_integrator(run,p,funcs,Y0,YP0,opts,new_run)
    
    set_vars!(model, p, Y0, YP0, int.t, run, opts, bounds; init_all=new_run, SOC=SOC)
    set_var!(model.Y, Y0, new_run || keep_Y)
    
    check_simulation_stop!(model, 0.0, Y0, YP0, run, p, bounds, opts)
    return int, funcs, model
end

@inline function retrieve_integrator(run::T, p::param, funcs::Jac_and_res{<:Sundials.IDAIntegrator}, Y0, YP0, opts::AbstractOptionsModel, new_run::Bool) where {method<:AbstractMethod,T<:AbstractRun{method,<:Any}}
    """
    If the model has previously been evaluated for a constant run simulation, you can reuse
    the integrator function with its cache instead of creating a new one
    """
    
    if isempty(funcs.int) || opts.calc_integrator
        int = create_integrator(run, p, funcs, Y0, YP0, opts)
    else
        # reuse the integrator cache
        int = @inbounds funcs.int[end]
        @inbounds int.p.value .= run.value

        mem = int.mem
        # reinitialize at t = 0 with new Y0/YP0 and tolerances
        Sundials.IDASStolerances(mem, opts.reltol, opts.abstol)
        Sundials.IDAReInit(mem, 0.0, Y0, YP0)
        int.t = 0.0
    end

    postfix_integrator!(int, run, opts, new_run)

    return int
end

@inline function create_integrator(run::T, p::param, funcs::Q, Y0, YP0, opts::AbstractOptionsModel) where {T<:AbstractRun,Q<:Jac_and_res}
    R_full = funcs.R_full
    J_full = funcs.J_full
    DAEfunc = DAEFunction(
        (res,YP,Y,run,t) -> R_full(res,t,Y,YP,p,run);
        jac = (J,YP,Y,run,γ,t) -> J_full(J,t,Y,YP,γ,p,run),
        jac_prototype = J_full.sp,
    )

    prob = DAEProblem(DAEfunc, YP0, Y0, (0.0, run.tf), run, differential_vars=p.cache.id)

    int = init(prob, Sundials.IDA(;linear_solver=:KLU, init_all=false), tstops=Float64[], abstol=opts.abstol, reltol=opts.reltol, save_everystep=false, save_start=false, verbose=false)

    if isempty(funcs.int)
        push!(funcs.int, int)
    end

    return int
end
@inline function postfix_integrator!(int::Sundials.IDAIntegrator, run::AbstractRun, opts::AbstractOptionsModel, new_run::Bool)
    tstops = int.opts.tstops.valtree
    empty!(tstops)
    
    append!(tstops,opts.tstops)
    append!(tstops,opts.tdiscon .- opts.reltol/2)
    push!(tstops,run.tf)
    if !new_run prepend!(tstops, 1.0) end

    sort!(tstops)

    # the model can fail is tstops includes 0
    if (@inbounds tstops[1]) ≤ 0.0 deleteat!(tstops, 1:findfirst(tstops .≤ 0)) end
    return nothing
end

@inline function solve!(model,int::R1,run::R2,p,bounds,opts::R3,funcs) where {R1<:Sundials.IDAIntegrator,R2<:AbstractRun,R3<:AbstractOptionsModel}
    keep_Y = opts.var_keep.Y
    Y  = int.u.v
    YP = int.du.v
    
    iter = 1
    status = within_bounds(run)
    @inbounds while status
        step!(int)
        iter += 1
        t = int.t

        set_vars!(model, p, Y, YP, t, run, opts, bounds)
        
        check_simulation_stop!(model, t, Y, YP, run, p, bounds, opts)
        
        status = check_solve(run, model, int, p, bounds, opts, funcs, keep_Y, iter, Y, t)
    end
    
    run.info.iterations = iter
    return nothing
end

@inline function exit_simulation!(p::R1, model::R2, run::R3, bounds::R4, int::R5, opts::R6; cancel_interp::Bool=false) where {R1<:param,R2<:model_output,R3<:AbstractRun,R4<:boundary_stop_conditions,R5<:Sundials.IDAIntegrator,R6<:AbstractOptionsModel}
    # if a stop condition (besides t = tf) was reached
    if !cancel_interp
        if opts.interp_final && !(run.info.flag === 0) && int.t > 1
            interp_final_points!(p, model, run, bounds, int, opts)
        else
            set_var!(model.Y, opts.var_keep.Y ? copy(int.u.v) : int.u.v, opts.var_keep.Y)
        end
    end

    iterations_start = isempty(model) ? 0 : (@inbounds @views model.results[end].run_index[end])

    run_index = (1:run.info.iterations) .+ iterations_start

    tspan = (run.t0, @inbounds model.t[end])

    results = run_results(
        run,
        tspan,
        run.info,
        run_index,
        int,
        opts,
        bounds,
        p.N,
        p.numerics,
        p,
    )

    push!(model.results, results)

    return nothing
end

@views @inbounds @inline function interp_final_points!(p::R1, model::R2, run::R3, bounds::R4, int::R5, opts::R6) where {R1<:param,R2<:model_output,R3<:AbstractRun,R4<:boundary_stop_conditions,R5<:Sundials.IDAIntegrator,R6<:AbstractOptionsModel}
    if opts.var_keep.YP
        YP = length(model.YP) > 1 ? bounds.t_final_interp_frac.*(model.YP[end] .- model.YP[end-1]) .+ model.YP[end-1] : model.YP[end]
    else
        YP = Float64[]
    end
    
    t = bounds.t_final_interp_frac*(int.t - int.tprev) + int.tprev
    
    set_var!(model.Y,  bounds.t_final_interp_frac.*(int.u.v .- model.Y[end]) .+ model.Y[end], opts.var_keep.Y)
    set_vars!(model, p, model.Y[end], YP, t, run, opts, bounds; modify! = set_var_last!)
    
    return nothing
end

@inline function save_start_init!(Y0::Vector{Float64}, run::AbstractRun, p::param, SOC::Float64)
    key = save_start_info(
        run.method,
        round(SOC, digits=4),
        round(value(run), digits=4),
    )

    save_start_dict = p.cache.save_start_dict
    
    key_exists = key ∈ keys(save_start_dict)
    if key_exists
        @inbounds Y0[(1:p.N.alg) .+ p.N.diff] .= save_start_dict[key]
    end
    
    return key, key_exists
end
@inline initialize_states!(p::param, Y0::T, YP0::T, run::AbstractRun, opts::AbstractOptionsModel, funcs::Jac_and_res; kw...) where {T<:Vector{Float64}} = newtons_method!(p,Y0,YP0,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg;kw...)
@inline function initialize_states!(p::param, Y0::T, YP0::T, run::AbstractRun, opts::AbstractOptionsModel, funcs::Jac_and_res, SOC::Float64;kw...) where {T<:Vector{Float64}}
    if opts.save_start
        key, key_exists = save_start_init!(Y0, run, p, SOC)
        
        newtons_method!(p,Y0,YP0,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg;kw...)
        
        if !key_exists
            p.cache.save_start_dict[key] = @inbounds Y0[(1:p.N.alg) .+ p.N.diff]
        end
    else
        newtons_method!(p,Y0,YP0,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg;kw...)
    end
    
    return nothing
end

@inline factorization!(L::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64},A::SparseMatrixCSC{Float64, Int64}) = LinearAlgebra.lu!(L,A)
@inline factorization!(L::KLUFactorization{Float64, Int64},A::SparseMatrixCSC{Float64, Int64}) = klu!(L,A)
@inline function newtons_method!(p::param,Y::R1,YP::R1,run,opts::AbstractOptionsModel,R_alg::T1,R_diff::T2,J_alg::T3;
    itermax::Int64=100, t::Float64=0.0
    ) where {R1<:Vector{Float64},T1<:residual_combined,T2<:residual_combined,T3<:jacobian_combined}

    res   = p.cache.res
    Y_old = p.cache.Y_alg
    Y_new = @views @inbounds Y[p.N.diff+1:end]
    YP   .= 0.0
    J     = J_alg.sp
    γ     = 0.0
    L     = J_alg.L # factorization
    
    # starting loop for Newton's method
    @inbounds for iter in 1:itermax
        # updating res, Y, and J
        R_alg(res,t,Y,YP,p,run)
        J_alg(t,Y,YP,γ,p,run)
        factorization!(L, J)
        
        Y_old .= Y_new
        Y_new .-= L\res
        if norm(Y_old .- Y_new) < opts.reltol_init # || maximum(abs, res) < opts.abstol_init
            break
        elseif iter === itermax
            error("Could not initialize DAE in $itermax iterations.")
        end
    end
    # calculate the differential equations for YP0
    R_diff(YP,t,Y,YP,p,run)
    
    return nothing
end

"""
Current
"""
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model, res_I_guess) where {method<:method_I,in<:Number}
    input = run.input
    @inbounds run.value .= Y0[p.ind.I[1]] = input
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:method_I,in<:Symbol}
    input = run.input
    if check_is_hold(input,model)
        @inbounds run.value .= Y0[p.ind.I[1]] = calc_I((@views @inbounds model.Y[end]), p)
    elseif input === :rest
        @inbounds run.value .= Y0[p.ind.I[1]] = 0.0
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0::Vector{Float64},p,run::run_function{method,func},model, res_I_guess) where {method<:method_I,func<:Function}
    run.value .= Y0[p.ind.I[1]] = run.func(0.0,Y0,YP0,p)
    return nothing
end

"""
Power
"""
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model, res_I_guess) where {method<:method_P,in<:Number}
    input = run.input
    @inbounds run.value .= Y0[p.ind.I[1]] = input/(calc_V(Y0,p)*p.θ[:I1C])
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:method_P,in<:Symbol}
    input = run.input
    if check_is_hold(input,model)
        @inbounds run.value .= Y0[p.ind.I[1]] = calc_P((@views @inbounds model.Y[end]), p)
    elseif input === :rest
        @inbounds run.value .= Y0[p.ind.I[1]] = 0.0
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0::Vector{Float64},p,run::run_function{method,func},model, res_I_guess) where {method<:method_P,func<:Function}
    run.value .= Y0[p.ind.I[1]] = run.func(0.0,Y0,YP0,p)/(calc_V(Y0,p)*p.θ[:I1C])
    return nothing
end

"""
Voltage and η_plating
"""
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:method_V,in<:Number}
    input = run.input
    @inbounds run.value .= input
    if !isempty(model)
        @inbounds Y0[p.ind.I[1]] = calc_I((@views @inbounds model.Y[end]), p)
    else
        OCV = calc_V(Y0,p)
        @inbounds Y0[p.ind.I[1]] = input > OCV ? +1.0 : -1.0
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:method_V,in<:Symbol}
    input = run.input
    if check_is_hold(input,model)
        Y = @views @inbounds model.Y[end]
        @inbounds run.value .= calc_V(Y, p)
        @inbounds Y0[p.ind.I[1]] = calc_V(Y, p)
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_function{method,func},model::model_output, res_I_guess) where {method<:method_V,func<:Function}
    @inbounds run.value .= run.func(0.0,Y0,YP0,p)
    if !isempty(model)
        @inbounds Y0[p.ind.I[1]] = calc_I((@views @inbounds model.Y[end]), p)
    else
        # Arbitrary guess for the initial current. 
        OCV = calc_V(Y0,p)
        @inbounds Y0[p.ind.I[1]] = value(run) > OCV ? +1.0 : -1.0
    end
    return nothing
end
