options = Dict{Symbol,Bool}(
    :SAVE_SYMBOLIC_FUNCTIONS => true
)

function load_functions(p::AbstractParam, methods::Tuple=(:I,))
    if     p.numerics.jacobian === :symbolic
        jac_type = jacobian_symbolic
        load_func = load_functions_symbolic
    elseif p.numerics.jacobian === :AD
        jac_type = jacobian_AD
        load_func = load_functions_forward_diff
    end

    ## Pre-allocation
    Y0_alg = zeros(Float64, p.N.alg)
    Y0_alg_prev = zeros(Float64, p.N.alg)
    res = zeros(Float64, p.N.alg)
    Y0_diff = zeros(Float64, p.N.diff)
    
    ## Begin loading functions based on method
    funcs = ImmutableDict{Symbol, functions_model{jac_type}}()
    
    @inbounds for method in methods
        initial_guess!, f!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys, θ_len = load_func(p, method, Y0_diff)

        update_θ! = update_θ_maker(θ_keys)
        append!(p.cache.θ_keys[method], θ_keys)
        append!(p.cache.θ_tot[method],  zeros(sum(θ_len)))

        initial_conditions = init_newtons_method(f_alg!, J_y_alg!, f_diff!, Y0_alg, Y0_alg_prev, Y0_diff, res)
        funcs = ImmutableDict(funcs, method => functions_model{jac_type}(f!, initial_guess!, J_y!, initial_conditions, update_θ!, Sundials.IDAIntegrator[]))
    end
    
    return funcs
end

function load_functions_symbolic(p::AbstractParam, method::Symbol, YP_cache=nothing)
    if options[:SAVE_SYMBOLIC_FUNCTIONS]
        dir = strings_directory_func(p) * "/"
        if !isdir(dir)
            generate_functions_symbolic(p)
        end

        ## residuals
        initial_guess! = include(dir * "initial_guess.jl")
        f!             = include(dir * "f.jl")
        f_alg!         = include(dir * "f_alg.jl")
        f_diff!        = include(dir * "f_diff.jl")
        J_y!_func      = include(dir * "J_y.jl")
        J_y_alg!_func  = include(dir * "J_y_alg.jl")
        
        ## Jacobian
        @load dir * "J_sp.jl" J_y_sp θ_keys θ_len
    else
        Y0Func,resFunc,res_algFunc,res_diffFunc,jacFunc,jac_algFunc,J_y_sp,θ_keys,θ_len = generate_functions_symbolic(p)
        
        initial_guess! = eval(Y0Func)
        f!             = eval(resFunc)
        f_alg!         = eval(res_algFunc)
        f_diff!        = eval(res_diffFunc)
        J_y!_func      = eval(jacFunc)
        J_y_alg!_func  = eval(jac_algFunc)
    end

    J_y!_sp     = sparse(J_y_sp...)
    J_y_alg!_sp = J_y!_sp[p.N.diff+1:end,p.N.diff+1:end]
    
    J_y!     = jacobian_symbolic(J_y!_func, J_y!_sp)
    J_y_alg! = jacobian_symbolic(J_y_alg!_func, J_y_alg!_sp)

    return initial_guess!, f!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys, θ_len
end

function generate_functions_symbolic(p::AbstractParam, method::Symbol=:I;
    verbose=true,
    )
    
    println_v(x...) = verbose ? println(x...) : nothing

    println_v("Creating the functions for $p")

    θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, I_current_sym, γ_sym, p_sym, θ_keys, θ_len = get_symbolic_vars(p)

    ## Y0 function
    println_v("Making initial guess function")
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, I_current_sym)

    ## batteryModel function
    println_v("Making symbolic model")
    res = _symbolic_residuals(p_sym, t_sym, Y_sym, YP_sym, I_current_sym, θ_sym, method)

    θ_sym_slim, θ_keys_slim, θ_len_slim = get_only_θ_used_in_model(θ_sym, θ_keys, θ_len, res, Y0_sym)
    
    Y0Func = build_function(Y0_sym, SOC_sym, θ_sym_slim, I_current_sym, fillzeros=false, checkbounds=false)[2]
    resFunc = build_function(res, t_sym, Y_sym, YP_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]

    ## jacobian
    println_v("Making symbolic Jacobian. May take a few mins")
    Jac, jacFunc, J_sp = _symbolic_jacobian(p, res, t_sym, Y_sym, YP_sym, γ_sym, θ_sym, θ_sym_slim, method, verbose=verbose)

    println_v("Making initial condition functions")
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, Y_sym, YP_sym, θ_sym, θ_sym_slim, method)

    jac_algFunc = _symbolic_initial_conditions_jac(p, Jac, Y_sym, YP_sym, θ_sym, θ_sym_slim, method)
    
    J_y_sp = (findnz(J_sp)..., p.N.tot-1, p.N.tot)
        
    θ_keys = θ_keys_slim
    θ_len = θ_len_slim

    if options[:SAVE_SYMBOLIC_FUNCTIONS]
        dir = strings_directory_func(p; create_dir=true) * "/"
        mkdir(dir)
        
        write(dir * "initial_guess.jl", string(Y0Func))
        write(dir * "f.jl",             string(resFunc))
        write(dir * "J_y.jl",           string(jacFunc))
        write(dir * "f_alg.jl",         string(res_algFunc))
        write(dir * "J_y_alg.jl",       string(jac_algFunc))
        write(dir * "f_diff.jl",        string(res_diffFunc))
            
        @save dir * "J_sp.jl" J_y_sp θ_keys θ_len
    end

    println_v("Finished\n")

    return Y0Func, resFunc, res_algFunc, res_diffFunc, jacFunc, jac_algFunc, J_y_sp, θ_keys, θ_len
end

function load_functions_forward_diff(p::AbstractParam, method::Symbol, YP_cache::Vector{Float64})

    θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, I_current_sym, γ_sym, p_sym, θ_keys, θ_len = get_symbolic_vars(p)

    ## Y0 function
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, I_current_sym)

    ## batteryModel function
    res = _symbolic_residuals(p_sym, t_sym, Y_sym, YP_sym, I_current_sym, θ_sym, method)
    
    θ_sym_slim, θ_keys_slim, θ_len_slim = get_only_θ_used_in_model(θ_sym, θ_keys, θ_len, res, Y0_sym)
    
    Y0Func = build_function(Y0_sym, SOC_sym, θ_sym_slim, I_current_sym, fillzeros=false, checkbounds=false)[2]
    resFunc = build_function(res, Y_sym, YP_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, Y_sym, YP_sym, θ_sym, θ_sym_slim, method)

    J_y_sp, sp_x, sp_xp = _Jacobian_sparsity_pattern(p, res, Y_sym, YP_sym)

    J_y_sp_alg = J_y_sp[p.N.diff+1:end,p.N.diff+1:end]

    function build_color_jacobian_struct(J, f!, N, _YP_cache=zeros(Float64,N))
        colorvec = SparseDiffTools.matrix_colors(J)

        Y_cache = zeros(Float64, N)

        func = res_FD(f!, _YP_cache, p.cache.θ_tot[method])
        
        jac_cache = SparseDiffTools.ForwardColorJacCache(
            func,
            Y_cache,
            dx = similar(Y_cache),
            colorvec = colorvec,
            sparsity = similar(J),
            )

        J_struct = jacobian_AD(func, J, jac_cache)
        return J_struct
    end

    initial_guess! = eval(Y0Func)
    f! = eval(resFunc)
    f_alg! = eval(res_algFunc)
    f_diff! = eval(res_diffFunc)

    J_y! = build_color_jacobian_struct(J_y_sp, f!, p.N.tot)
    J_y_alg! = build_color_jacobian_struct(J_y_sp_alg, f_alg!, p.N.alg, YP_cache)


    @assert size(J_y!.sp) === (p.N.tot,p.N.tot)
    @assert size(J_y_alg!.sp) === (p.N.alg,p.N.alg)

    return initial_guess!, f!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys_slim, θ_len_slim
end

function _symbolic_initial_guess(p::AbstractParam, SOC_sym, θ_sym, I_current_sym)
    
    Y0_sym = guess_init(p, +1.0)[1]

    deleteat!(Y0_sym, p.N.tot)

    return Y0_sym
end

function _symbolic_residuals(p::AbstractParam, t_sym, Y_sym, YP_sym, I_current_sym, θ_sym, method)
    ## symbolic battery model
    res = zeros(eltype(t_sym), size(p.cache.Y0))
    residuals_PET!(res, t_sym, Y_sym, YP_sym, method, p)

    deleteat!(res, p.N.tot)

    return res
end

function _Jacobian_sparsity_pattern(p, res, Y_sym, YP_sym)
    sp_x = ModelingToolkit.jacobian_sparsity(res, Y_sym)
    sp_xp = ModelingToolkit.jacobian_sparsity(res, YP_sym)
    
    I,J,_ = findnz(sp_x .+ sp_xp)

    J_sp = sparse(I, J, 1.0, p.N.tot-1, p.N.tot)
    
    return J_sp, sp_x, sp_xp
end

function _symbolic_jacobian(p::AbstractParam, res, t_sym, Y_sym, YP_sym, γ_sym, θ_sym, θ_sym_slim, method; verbose=false)

    J_sp, sp_x, sp_xp = _Jacobian_sparsity_pattern(p, res, Y_sym, YP_sym)

    ## symbolic jacobian
    Jac_x  = sparsejacobian_multithread(res, Y_sym;  show_progress=false, simplify=false)
    Jac_xp = sparsejacobian_multithread(res, YP_sym; show_progress = false, sp = sp_xp)
    
    # For some reason, Jac[ind_new] .= Jac_new doesn't work on linux. This if statement is a temporary workaround
    Jac = Jac_x .+ γ_sym.*Jac_xp

    # @assert length(Jac.nzval) === length(J_sp.nzval)
    
    # building the sparse jacobian
    jacFunc = build_function(Jac.nzval, t_sym, Y_sym, YP_sym, γ_sym, θ_sym_slim)[2]
    
    # _fix_sparse_matrix_build_function(jacFunc)

    return Jac, jacFunc, J_sp
end
function _symbolic_initial_conditions_res(p::AbstractParam, res, Y_sym, YP_sym, θ_sym, θ_sym_slim, method)
    
    res_diff = res[1:p.N.diff]
    res_alg = res[p.N.diff+1:end]
    
    res_diffFunc = build_function(res_diff, Y_sym, YP_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    res_algFunc = build_function(res_alg,  Y_sym[p.N.diff+1:end], Y_sym[1:p.N.diff], θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    
    return res_algFunc, res_diffFunc
end
function _symbolic_initial_conditions_jac(p::AbstractParam, Jac, Y_sym, YP_sym, θ_sym, θ_sym_slim, method)
    
    jac_alg = Jac[p.N.diff+1:end,p.N.diff+1:end]
    
    jac_algFunc = build_function(jac_alg.nzval,  Y_sym[p.N.diff+1:end], Y_sym[1:p.N.diff], θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    
    return jac_algFunc
end

function get_only_θ_used_in_model(θ_sym, θ_keys, θ_len, X...)
    """
    Some of the parameters in θ may not be used in the model. This returns
    a vector of the ones that are actually used and their sorted names
    """
    used_params_tot = eltype(θ_sym)[]
    @inbounds for _X in X, x in _X
        append!(used_params_tot, ModelingToolkit.get_variables(x))
    end
    index_params = Int64[]

    dummy = BitArray{1}(undef, length(θ_sym))
    @inbounds for x in unique(used_params_tot)
        dummy .= isequal.(x, θ_sym)
        if sum(dummy) === 1
            index_param = findfirst(dummy)
            push!(index_params, index_param)
        end
    end
    sort!(index_params)

    θ_sym_slim = θ_sym[index_params]

    # for some reason, sometimes the above script does not remove all duplicates
    unique_ind = indexin(unique(θ_sym_slim), θ_sym_slim)
    θ_sym_slim = θ_sym_slim[unique_ind]
    index_params = index_params[unique_ind]
    
    # remove additional entries due to parameters that are vectors
    θ_keys_extend = Symbol[]
    θ_len_extend = Int64[]
    @inbounds for (key,len) in zip(θ_keys, θ_len)
        append!(θ_keys_extend, repeat([key], len))
        append!(θ_len_extend, repeat([len], len))
    end
    
    θ_keys_slim = θ_keys_extend[index_params]
    θ_len_slim = θ_len_extend[index_params]
    
    # Vectors of length n will have n-1 duplicates. This removes them
    unique_ind = indexin(unique(θ_keys_slim), θ_keys_slim)
    θ_keys_slim = θ_keys_slim[unique_ind]
    θ_len_slim = θ_len_slim[unique_ind]
    
    return θ_sym_slim, θ_keys_slim, θ_len_slim
end
function update_θ_maker(θ_vars::AbstractVector)
    str = "((x,θ) -> (@inbounds begin;"

    @inbounds for (i, θ) in enumerate(θ_vars)
        str *= "x[$(i)]=θ[:$(θ)];"
    end
    str *= "end;nothing))"

    func = mk_function(Meta.parse(str));
    return func
end

function get_symbolic_vars(p::AbstractParam)
    θ_keys = sort!(Symbol.(keys(p.θ))) # sorted for convenience
    θ_len = Int64[length(p.θ[key]) for key in θ_keys]

    ModelingToolkit.@variables Y_sym[1:p.N.tot], YP_sym[1:p.N.tot], t_sym, SOC_sym, I_current_sym, γ_sym
    ModelingToolkit.@parameters θ_sym[1:sum(θ_len)]

    p_sym = param_no_funcs([convert(_type,getproperty(p,field)) for (field,_type) in zip(fieldnames(param_no_funcs),fieldtypes(param_no_funcs))]...)
    p_sym.opts.SOC = SOC_sym

    @inbounds for (i,key) in enumerate(θ_keys)
        p_sym.θ[key] = θ_sym[i]
    end

    return θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, I_current_sym, γ_sym, p_sym, θ_keys, θ_len
end

function sparsejacobian_multithread(ops::AbstractVector{<:Num}, vars::AbstractVector{<:Num};
    sp = ModelingToolkit.jacobian_sparsity(ops, vars),
    simplify = true,
    show_progress = true,
    multithread = true,
    )

    I,J,_ = findnz(sp)

    exprs = Vector{Num}(undef, length(I))

    iters = show_progress ? ProgressBar(1:length(I)) : 1:length(I)
    if multithread
        @inbounds Threads.@threads for iter in iters
            @inbounds exprs[iter] = expand_derivatives(Differential(vars[J[iter]])(ops[I[iter]]), simplify)
        end
    else
        @inbounds for iter in iters
            @inbounds exprs[iter] = expand_derivatives(Differential(vars[J[iter]])(ops[I[iter]]), simplify)
        end
    end

    jac = sparse(I,J, exprs, length(ops), length(vars))
    return jac
end
