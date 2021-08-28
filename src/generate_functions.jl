options = Dict{Symbol,Bool}(
    :SAVE_SYMBOLIC_FUNCTIONS => true,
)

function load_functions(p::AbstractParam)
    if     p.numerics.jacobian === :symbolic
        jac_type = jacobian_symbolic
        load_func = load_functions_symbolic
    elseif p.numerics.jacobian === :AD
        jac_type = jacobian_AD
        load_func = load_functions_forward_diff
    end
    
    ## Begin loading functions
    initial_guess!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys = load_func(p)

    append!(p.cache.θ_keys, θ_keys)
    append!(p.cache.θ_tot,  zeros(length(θ_keys)))

    funcs = model_funcs(initial_guess!, f_diff!, f_alg!, J_y!, J_y_alg!)
    
    return funcs
end

function load_functions_symbolic(p::AbstractParam)
    dir = strings_directory_func(p) * "/"
    files_exist = isdir(dir)

    if files_exist && options[:SAVE_SYMBOLIC_FUNCTIONS]
        ## residuals
        initial_guess! = include(dir * "initial_guess.jl")
        f_alg!         = include(dir * "f_alg.jl")
        f_diff!        = include(dir * "f_diff.jl")
        J_y!_func      = include(dir * "J_y.jl")
        J_y_alg!_func  = include(dir * "J_y_alg.jl")
        
        ## Jacobian
        @load dir * "J_sp.jl" J_y_sp θ_keys
    else
        Y0Func,res_algFunc,res_diffFunc,jacFunc,jac_algFunc,J_y_sp,θ_keys = generate_functions_symbolic(p)
        
        initial_guess! = eval(Y0Func)
        f_alg!         = eval(res_algFunc)
        f_diff!        = eval(res_diffFunc)
        J_y!_func      = eval(jacFunc)
        J_y_alg!_func  = eval(jac_algFunc)
    end

    J_y!_sp     = sparse(J_y_sp...)
    J_y_alg!_sp = @inbounds J_y!_sp[p.N.diff+1:end,p.N.diff+1:end]
    
    J_y!     = jacobian_symbolic(J_y!_func, J_y!_sp)
    J_y_alg! = jacobian_symbolic(J_y_alg!_func, J_y_alg!_sp)

    return initial_guess!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys
end

function generate_functions_symbolic(p::AbstractParam; verbose=true)
    
    println_v(x...) = verbose ? println(x...) : nothing

    println_v("Creating the functions for $p\n\nMay take a few minutes...")

    θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, X_applied, γ_sym, p_sym, θ_keys = get_symbolic_vars(p)

    ## Y0 function
    println_v("1/4: Making initial guess function")
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, X_applied)

    ## batteryModel function
    println_v("2/4: Making symbolic model")
    res = _symbolic_residuals(p_sym, t_sym, Y_sym, YP_sym)

    θ_sym_slim, θ_keys_slim = get_only_θ_used_in_model(θ_sym, θ_keys, res, Y0_sym)
    
    Y0Func = build_function(Y0_sym, SOC_sym, θ_sym_slim, X_applied, fillzeros=false, checkbounds=false)[2]

    ## jacobian
    println_v("3/4: Making symbolic Jacobian")
    Jac, jacFunc, J_sp = _symbolic_jacobian(p, res, t_sym, Y_sym, YP_sym, γ_sym, θ_sym, θ_sym_slim, verbose=verbose)

    println_v("4/4: Making initial condition functions")
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, t_sym, Y_sym, YP_sym, θ_sym, θ_sym_slim)

    jac_algFunc = _symbolic_initial_conditions_jac(p, Jac, t_sym, Y_sym, YP_sym, γ_sym, θ_sym_slim)
    
    J_y_sp = (findnz(J_sp)..., p.N.tot-1, p.N.tot)
        
    θ_keys = θ_keys_slim

    if options[:SAVE_SYMBOLIC_FUNCTIONS]
        dir = strings_directory_func(p; create_dir=true) * "/"
        
        write(dir * "initial_guess.jl", string(Y0Func))
        write(dir * "J_y.jl",           string(jacFunc))
        write(dir * "f_alg.jl",         string(res_algFunc))
        write(dir * "J_y_alg.jl",       string(jac_algFunc))
        write(dir * "f_diff.jl",        string(res_diffFunc))
            
        @save dir * "J_sp.jl" J_y_sp θ_keys
    end

    println_v("Finished\n")

    return Y0Func, res_algFunc, res_diffFunc, jacFunc, jac_algFunc, J_y_sp, θ_keys
end

function load_functions_forward_diff(p::AbstractParam)

    θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, X_applied, γ_sym, p_sym, θ_keys = get_symbolic_vars(p)

    ## Y0 function
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, X_applied)

    ## batteryModel function
    res = _symbolic_residuals(p_sym, t_sym, Y_sym, YP_sym)
    
    θ_sym_slim, θ_keys_slim = get_only_θ_used_in_model(θ_sym, θ_keys, res, Y0_sym)
    
    Y0Func = build_function(Y0_sym, SOC_sym, θ_sym_slim, X_applied, fillzeros=false, checkbounds=false)[2]
    
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, t_sym, Y_sym, YP_sym, θ_sym, θ_sym_slim)

    J_y_sp = _Jacobian_sparsity_pattern(p, res, Y_sym, YP_sym)[1]

    """
    Sometimes, the differential terms do not appear in the row corresponding to their value.
    this line ensures that there is still a spot for the ∂x/∂t term in the jacobian sparsity pattern/
    functon, otherwise there will be a mismatch in the sparse matrix
    """
    @inbounds for i in 1:p.N.diff
        if iszero(J_y_sp[i,i])
            J_y_sp[i,i] = 1e-100 # some arbitrarily small number so it doesn't get overwritten
        end
    end

    J_y_sp_alg = @inbounds J_y_sp[p.N.diff+1:end,p.N.diff+1:end]

    function build_color_jacobian_struct(J, f!, N, )
        colorvec = matrix_colors(J)

        Y_cache = zeros(Float64, N+1)

        _Y_cache  = zeros(Float64,p.N.tot)
        _YP_cache = zeros(Float64,p.N.tot)
        func = res_FD(f!, _Y_cache, _YP_cache, p.cache.θ_tot, p.N)
        
        jac_cache = ForwardColorJacCache(
            func,
            Y_cache,
            dx = zeros(N),
            colorvec = colorvec,
            sparsity = similar(J),
            )

        J_struct = jacobian_AD(func, J, jac_cache)
        return J_struct
    end

    initial_guess! = eval(Y0Func)
    f_alg!         = eval(res_algFunc)
    f_diff!        = eval(res_diffFunc)
    f! = function (res,t,Y,YP,θ_tot)
        f_diff!((@views @inbounds res[1:p.N.diff]),       t, Y, YP, θ_tot)
        f_alg!( (@views @inbounds res[p.N.diff+1:end-1]), t, Y, YP, θ_tot)
    end

    J_y!     = build_color_jacobian_struct(J_y_sp, f!, p.N.tot-1)
    J_y_alg! = build_color_jacobian_struct(J_y_sp_alg, f_alg!, p.N.alg-1)

    @assert size(J_y!.sp) === (p.N.tot-1,p.N.tot)
    @assert size(J_y_alg!.sp) === (p.N.alg-1,p.N.alg)

    return initial_guess!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys_slim
end

function _symbolic_initial_guess(p::AbstractParam, SOC_sym, θ_sym, X_applied)
    
    Y0_sym = guess_init(p, +1.0)[1]

    deleteat!(Y0_sym, p.N.tot)

    return Y0_sym
end

function _symbolic_residuals(p::AbstractParam, t_sym, Y_sym, YP_sym)
    ## symbolic battery model
    res = similar(Y_sym)
    residuals_PET!(res, t_sym, Y_sym, YP_sym, p)

    deleteat!(res, p.N.tot)

    return res
end

function _Jacobian_sparsity_pattern(p, res, Y_sym, YP_sym)
    sp_x = jacobian_sparsity(res, Y_sym)
    sp_xp = jacobian_sparsity(res, YP_sym)
    
    I,J,_ = findnz(sp_x .+ sp_xp)

    J_sp = sparse(I, J, 1.0, p.N.tot-1, p.N.tot)
    
    return J_sp, sp_x, sp_xp
end

function _symbolic_jacobian(p::AbstractParam;inds::T=1:p.N.tot) where T<:UnitRange{Int64}
    θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, X_applied, γ_sym, p_sym, θ_keys = get_symbolic_vars(p)
    res = [_symbolic_residuals(p_sym, t_sym, Y_sym, YP_sym);Y_sym[end]]

    res    = @inbounds @views res[inds]
    Y_sym  = @inbounds @views Y_sym[inds]
    YP_sym = @inbounds @views YP_sym[inds]

    ## symbolic jacobian
    Jac_x  = sparsejacobian_multithread(res, Y_sym;  simplify=false)
    Jac_xp = sparsejacobian_multithread(res, YP_sym; simplify=false)
    
    Jac = Jac_x
    @inbounds for (i,j) in zip(Jac_xp.rowval,Jac_xp.colptr)
        Jac[i,j] += γ_sym*Jac_xp[i,j]
    end

    return Jac
end
function _symbolic_jacobian(p::AbstractParam, res, t_sym, Y_sym, YP_sym, γ_sym, θ_sym, θ_sym_slim; verbose=false)
    J_sp, sp_x, sp_xp = _Jacobian_sparsity_pattern(p, res, Y_sym, YP_sym)

    ## symbolic jacobian
    Jac_x  = sparsejacobian_multithread(res, Y_sym;  sp = sp_x,  simplify=false)
    Jac_xp = sparsejacobian_multithread(res, YP_sym; sp = sp_xp, simplify=false)
    
    Jac = Jac_x
    @inbounds for (i,j) in zip(Jac_xp.rowval,Jac_xp.colptr)
        Jac[i,j] += γ_sym*Jac_xp[i,j]
    end

    # building the sparse jacobian
    jacFunc = build_function(Jac.nzval, t_sym, Y_sym, YP_sym, γ_sym, θ_sym_slim)[2]

    return Jac, jacFunc, J_sp
end
function _symbolic_initial_conditions_res(p::AbstractParam, res, t_sym, Y_sym, YP_sym, θ_sym, θ_sym_slim)
    
    res_diff = res[1:p.N.diff]
    res_alg = res[p.N.diff+1:end]
    
    res_diffFunc = build_function(res_diff, t_sym, Y_sym, YP_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    res_algFunc = build_function(res_alg,  t_sym, Y_sym, YP_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    
    return res_algFunc, res_diffFunc
end
function _symbolic_initial_conditions_jac(p::AbstractParam, Jac, t_sym, Y_sym, YP_sym, γ_sym, θ_sym_slim)
    
    jac_alg = @inbounds Jac[p.N.diff+1:end,p.N.diff+1:end]
    
    jac_algFunc = build_function(jac_alg.nzval, t_sym, Y_sym, YP_sym, γ_sym, θ_sym_slim, fillzeros=false, checkbounds=false)[2]
    
    return jac_algFunc
end

function get_only_θ_used_in_model(θ_sym, θ_keys, X...)
    """
    Some of the parameters in θ may not be used in the model. This returns
    a vector of the ones that are actually used and their sorted names
    """
    used_params_tot = eltype(θ_sym)[]
    @inbounds for _X in X, x in _X
        append!(used_params_tot, get_variables(x))
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

    θ_keys_slim = θ_keys[index_params]
    
    # Vectors of length n will have n-1 duplicates. This removes them
    unique_ind = indexin(unique(θ_keys_slim), θ_keys_slim)
    θ_keys_slim = θ_keys_slim[unique_ind]
    
    return θ_sym_slim, θ_keys_slim
end
update_θ!(p::param) = update_θ!(p.cache.θ_tot,p.cache.θ_keys,p.θ)
function update_θ!(θ::Vector{Float64},keys::Vector{Symbol},θ_Dict::Dict{Symbol,Float64})
    @inbounds for i in 1:length(θ)
        θ[i] = θ_Dict[keys[i]]
    end
    θ_Dict[:I1C] = calc_I1C(θ_Dict)
    return nothing
end

function get_symbolic_vars(p::AbstractParam)
    θ_keys = sort!(Symbol.(keys(p.θ))) # sorted for convenience

    @variables Y_sym[1:p.N.tot], YP_sym[1:p.N.tot], t_sym, SOC_sym, X_applied, γ_sym
    @variables θ_sym[1:length(θ_keys)]

    Y_sym  = collect(Y_sym)
    YP_sym = collect(YP_sym)
    θ_sym  = collect(θ_sym)

    p_sym = param_skeleton([convert(_type,deepcopy(getproperty(p,field))) for (field,_type) in zip(fieldnames(param_skeleton),fieldtypes(param_skeleton))]...)
    p_sym.opts.SOC = SOC_sym

    @inbounds for (i,key) in enumerate(θ_keys)
        p_sym.θ[key] = θ_sym[i]
    end

    return θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, X_applied, γ_sym, p_sym, θ_keys
end

function sparsejacobian_multithread(ops::AbstractVector{<:Num}, vars::AbstractVector{<:Num};
    sp = jacobian_sparsity(ops, vars),
    simplify = true,
    multithread = true,
    )

    I,J,_ = findnz(sp)

    exprs = Vector{Num}(undef, length(I))

    if multithread
        @inbounds Threads.@threads for iter in 1:length(I)
            @inbounds exprs[iter] = expand_derivatives(Differential(vars[J[iter]])(ops[I[iter]]), simplify)
        end
    else
        @inbounds for iter in 1:length(I)
            @inbounds exprs[iter] = expand_derivatives(Differential(vars[J[iter]])(ops[I[iter]]), simplify)
        end
    end

    jac = sparse(I,J, exprs, length(ops), length(vars))
    return jac
end
