SAVE_SYMBOLIC_FUNCTIONS = true

function load_functions(p::AbstractParam, methods::Tuple)
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

        update_θ! = update_θ_maker(θ_keys, θ_len)
        append!(p.cache.θ_keys[method], θ_keys)
        append!(p.cache.θ_tot[method],  zeros(sum(θ_len)))

        initial_conditions = init_newtons_method(f_alg!, J_y_alg!, f_diff!, Y0_alg, Y0_alg_prev, Y0_diff, res)
        funcs = ImmutableDict(funcs, method => functions_model{jac_type}(f!, initial_guess!, J_y!, initial_conditions, update_θ!, Sundials.IDAIntegrator[]))
    end

    if p.numerics.jacobian === :symbolic && !SAVE_SYMBOLIC_FUNCTIONS
        rm(strings_directory_func(p))
    end
    
    return funcs
end

function retrieve_functions_symbolic(p::AbstractParam, methods::Tuple)
    ## Checking if the main directory exists
    dir = strings_directory_func(p)
    
    ## Checking if the methods exist
    res_prev = Jac_prev = nothing
    @inbounds for method in methods
        if !isdir(dir * "/$method/")
            res_prev, Jac_prev = generate_functions_symbolic(p, method, res_prev=res_prev, Jac_prev=Jac_prev)
        end
    end

    funcs = load_functions(p, methods)
    
    return funcs
end

function load_functions_symbolic(p::AbstractParam, method::Symbol, YP_cache=nothing)
    dir = strings_directory_func(p) * "/$method/"

    ## residuals
    initial_guess! = include(dir * "initial_guess.jl")
    f! = include(dir * "f.jl")
    f_alg! = include(dir * "f_alg.jl")
    f_diff! = include(dir * "f_diff.jl")

    ## Jacobian
    @load dir * "J_sp.jl" J_y_sp θ_keys θ_len
    
    J_y!_func = include(dir * "J_y.jl")
    J_y_alg!_func = include(dir * "J_y_alg.jl")

    J_y!_sp = sparse(J_y_sp...)
    J_y_alg!_sp = J_y!_sp[p.N.diff+1:end,p.N.diff+1:end]
    
    J_y! = jacobian_symbolic(J_y!_func, J_y!_sp)
    J_y_alg! = jacobian_symbolic(J_y_alg!_func, J_y_alg!_sp)

    if !SAVE_SYMBOLIC_FUNCTIONS
        @inbounds for file in ("initial_guess.jl", "f.jl", "f_alg.jl", "f_diff.jl", "J_sp.jl", "J_y.jl", "J_y_alg.jl")
            rm(dir * file)
        end
        rm(dir)
    end

    return initial_guess!, f!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys, θ_len
end

function _fix_sparse_matrix_build_function(J)
    """
    The newest version of ModelingToolkit (5.6.4) causes my machine to use CartesianIndices
    instead of .nzval for the sparse matrix. This is a hacky workaround
    """
    
    eqns = J.args[2].args[3].args[2].args[2].args[3].args[2:end-2]

    @inbounds for x in eqns
        x.args[1].args[1] = Symbol(x.args[1].args[1], ".nzval")
    end
end

function generate_functions_symbolic(p::AbstractParam, method::Symbol;
    verbose=true,
    # if this function has previously been evaluated, this will make the next one quicker
    res_prev=nothing,
    Jac_prev=nothing,
    )
    
    println_v(x...) = verbose ? println(x...) : nothing

    if isnothing(res_prev) && isnothing(Jac_prev)
        println_v("Creating the functions for method $method, $p")
    else
        println_v("Continuing to create method $method")
    end

    θ_sym, x_sym, xp_sym, t_sym, SOC_sym, I_current_sym, p_sym, θ_keys, θ_len = get_symbolic_vars(p)

    ## Y0 function
    println_v("Making initial guess function")
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, I_current_sym)

    ## batteryModel function
    println_v("Making symbolic model")
    res, ind_res = _symbolic_residuals(p_sym, t_sym, x_sym, xp_sym, I_current_sym, θ_sym, method)

    θ_sym_slim, θ_keys_slim, θ_len_slim = get_only_θ_used_in_model(θ_sym, res, Y0_sym, θ_keys, θ_len)
    
    Y0Func = build_function(Y0_sym, SOC_sym, θ_sym_slim, I_current_sym, fillzeros=false, checkbounds=false)[2]
    res_build = zeros(eltype(res), p.N.tot)
    res_build[ind_res] .= res[ind_res]
    resFunc = build_function(res_build, x_sym, xp_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]

    ## jacobian
    println_v("Making symbolic Jacobian. May take a few mins")
    Jac, jacFunc, J_sp = _symbolic_jacobian(p, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method;
        res_prev=res_prev, Jac_prev=Jac_prev, verbose=verbose)

    println_v("Making initial condition functions")
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method, ind_res)

    jac_algFunc = _symbolic_initial_conditions_jac(p, Jac, x_sym, xp_sym, θ_sym, θ_sym_slim, method)
    
    dir = strings_directory_func(p; create_dir=true) * "/$method/"
    mkdir(dir)
    
    write(dir * "initial_guess.jl", string(Y0Func))
    write(dir * "f.jl",             string(resFunc))
    write(dir * "J_y.jl",           replace(string(jacFunc), ".nzval\"" => "\".nzval"))
    write(dir * "f_alg.jl",         string(res_algFunc))
    write(dir * "J_y_alg.jl",       replace(string(jac_algFunc), ".nzval\"" => "\".nzval"))
    write(dir * "f_diff.jl",        string(res_diffFunc))
        
    J_y_sp = (findnz(J_sp)..., p.N.tot, p.N.tot)
    
    θ_keys = θ_keys_slim
    θ_len = θ_len_slim
    @save dir * "J_sp.jl" J_y_sp θ_keys θ_len

    println_v("Finished\n")

    res_prev = res
    Jac_prev = Jac

    return res_prev, Jac_prev
end

function load_functions_forward_diff(p::AbstractParam, method::Symbol, YP_cache::Vector{Float64})

    θ_sym, x_sym, xp_sym, t_sym, SOC_sym, I_current_sym, p_sym, θ_keys, θ_len = get_symbolic_vars(p)

    ## Y0 function
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, I_current_sym)

    ## batteryModel function
    res = _symbolic_residuals(p_sym, t_sym, x_sym, xp_sym, I_current_sym, θ_sym, method)[1]

    ind_res = 1:length(res)
    
    θ_sym_slim, θ_keys_slim, θ_len_slim = get_only_θ_used_in_model(θ_sym, res, Y0_sym, θ_keys, θ_len)
    
    Y0Func = build_function(Y0_sym, SOC_sym, θ_sym_slim, I_current_sym, fillzeros=false, checkbounds=false)[2]
    resFunc = build_function(res, x_sym, xp_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method, ind_res)

    J_y_sp, sp_x, sp_xp = _Jacobian_sparsity_pattern(p, res, x_sym, xp_sym)

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
    Y0_sym[p.ind.I] .= I_current_sym

    return Y0_sym
end

function _symbolic_residuals(p::AbstractParam, t_sym, x_sym, xp_sym, I_current_sym, θ_sym, method)
    ## symbolic battery model
    res = zeros(eltype(t_sym), size(p.cache.Y0))
    residuals_PET!(res, t_sym, x_sym, xp_sym, method, p)

    if method ∈ (:I, :V, :P)
        ind_res = 1:length(res)-1
    else
        ind_res = 1:length(res)
    end

    return res, ind_res
end

function _Jacobian_sparsity_pattern(p, res, x_sym, xp_sym)
    sp_x = ModelingToolkit.jacobian_sparsity(res, x_sym)
    sp_xp = ModelingToolkit.jacobian_sparsity(res, xp_sym)
    
    I,J,_ = findnz(sp_x .+ sp_xp)

    J_sp = sparse(I, J, 1.0, p.N.tot, p.N.tot)
    
    return J_sp, sp_x, sp_xp
end

function _symbolic_jacobian(p::AbstractParam, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method;
    res_prev=nothing, Jac_prev=nothing, verbose=false)

    J_sp, sp_x, sp_xp = _Jacobian_sparsity_pattern(p, res, x_sym, xp_sym)

    # if this function has previously been evaluated, this will make the next one quicker.
    # Don't bother recomputing the Jacobian for lines of the residual that are identical
    flag_prev = !isnothing(res_prev) && !isnothing(Jac_prev)
    if flag_prev
        ind_new = Int64[]
        @inbounds Threads.@threads for i in 1:length(res)
            if @views @inbounds !isequal(res[i], res_prev[i])
                push!(ind_new, i)
                SparseArrays.fkeep!(Jac_prev, (I,J,K) -> I ≠ i)
            end
        end

    else
        ind_new = 1:length(res)
    end

    ## symbolic jacobian
    Jacxp = sparsejacobian_multithread(res, xp_sym; show_progress = false, sp = sp_xp)
    function check_semi_explicit(Jacxp)
        I,J,K = findnz(Jacxp)
        @inbounds for (i,j,k) in zip(I,J,K)
            @assert i === j # only along the diagonal
            @assert isequal(k,1) || isequal(k,-1) # only ones or negative ones on diagonal
        end
    end
    check_semi_explicit(Jacxp)

    Jac_new = @inbounds @views sparsejacobian_multithread(res[ind_new], x_sym;  show_progress=false&&!flag_prev, simplify=false)
    
    # For some reason, Jac[ind_new] .= Jac_new doesn't work on linux. This if statement is a temporary workaround
    if !flag_prev
        Jac = Jac_new
    else
        Jac = Jac_prev
        Jac[ind_new,:] .= Jac_new
    end

    """
    Sometimes, the differential terms do not appear in the row corresponding to their value.
    this line ensures that there is still a spot for the ∂x/∂t term in the jacobian sparsity pattern/
    functon, otherwise there will be a mismatch
    """
    @inbounds for i in 1:p.N.diff
        if !sp_x[i,i]
            Jac[i,i] = 1e-100 # some arbitrarily small number so it doesn't get overwritten
        end
    end
    
    @assert length(Jac.nzval) === length(J_sp.nzval)
    
    # building the sparse jacobian
    jacFunc = build_function(Jac, x_sym, θ_sym_slim)[2]
    
    # _fix_sparse_matrix_build_function(jacFunc)

    return Jac, jacFunc, J_sp
end
function _symbolic_initial_conditions_res(p::AbstractParam, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method, ind_res)
    
    res_diff = res[1:p.N.diff]
    res_alg = zeros(eltype(res), p.N.alg)
    res_alg[ind_res[p.N.diff+1:end] .- p.N.diff] .= res[ind_res[p.N.diff+1:end]]
    
    res_diffFunc = build_function(res_diff, x_sym, xp_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    res_algFunc = build_function(res_alg,  x_sym[p.N.diff+1:end], x_sym[1:p.N.diff], θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    
    return res_algFunc, res_diffFunc
end
function _symbolic_initial_conditions_jac(p::AbstractParam, Jac, x_sym, xp_sym, θ_sym, θ_sym_slim, method)
    
    jac_alg = Jac[p.N.diff+1:end,p.N.diff+1:end]
    
    jac_algFunc = build_function(jac_alg,  x_sym[p.N.diff+1:end], x_sym[1:p.N.diff], θ_sym_slim, fillzeros=false, checkbounds=false)[2]

    # _fix_sparse_matrix_build_function(jac_algFunc)
    
    return jac_algFunc
end

function get_only_θ_used_in_model(θ_sym, res, Y0_sym, θ_keys, θ_len)
    """
    Some of the parameters in θ may not be used in the model. This returns
    a vector of the ones that are actually used and their sorted names
    """
    used_params_tot = eltype(θ_sym)[]
    @inbounds for (Y,r) in zip(Y0_sym,res)
        append!(used_params_tot, ModelingToolkit.get_variables(Y))
        append!(used_params_tot, ModelingToolkit.get_variables(r))
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
function update_θ_maker(θ_vars::AbstractVector, θ_len::Vector{Int64})
    str = "((x,θ) -> (@inbounds @views begin;"

    start = 0
    @inbounds for (θ, l) in zip(θ_vars, θ_len)
        if l === 1
            str *= "x[$(1+start)]=θ[:$(θ)]::Float64;"
        else
            str *= "x[$((1:l).+start)].=θ[:$(θ)]::Vector{Float64};"
        end
        start += l
    end
    str *= "end;nothing))"

    func = mk_function(Meta.parse(str));
    return func
end

function get_symbolic_vars(p::AbstractParam)
    θ_keys = sort!(Symbol.(keys(p.θ))) # sorted for convenience
    θ_len = Int64[length(p.θ[key]) for key in θ_keys]

    ModelingToolkit.@variables x_sym[1:p.N.tot], xp_sym[1:p.N.tot], t_sym, SOC_sym, I_current_sym
    ModelingToolkit.@parameters θ_sym[1:sum(θ_len)]

    p_sym = deepcopy(p)
    p_sym.opts.SOC = SOC_sym

    start = 0
    @inbounds for (i,key) in enumerate(θ_keys)
        if θ_len[i] === 1
            p_sym.θ[key] = θ_sym[1+start]
        else
            p_sym.θ[key] = θ_sym[(1:θ_len[i]) .+ start]
        end

        start += θ_len[i]
    end

    return θ_sym, x_sym, xp_sym, t_sym, SOC_sym, I_current_sym, p_sym, θ_keys, θ_len
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
