const PETLION_VERSION = (0,2,0)
const options = Dict{Symbol,Any}(
    :SAVE_SYMBOLIC_FUNCTIONS => true,
    :FILE_DIRECTORY => pwd(),
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

function get_saved_model_version(p::AbstractParam)
    """
    Gets the version of PETLION used to generate the saved model
    """
    str = strings_directory_func(p)
    str *= "/info.txt"
    
    out = readline(str)
    out = replace(out, "PETLION version: v" => "")
    # convert this to a tuple
    out = (Meta.parse.(split(out,"."))...,)
end

function remove_model_files(p::AbstractParam)
    """
    Removes symbolic files for the model
    """
    str = strings_directory_func(p) * "/"
    for x in readdir(str)
        rm(str*x)
    end
end

function load_functions_symbolic(p::AbstractParam)
    dir = strings_directory_func(p) * "/"
    files_exist = isdir(dir)

    if files_exist
        file_version = get_saved_model_version(p)
        
        # have there been any breaking changes since creating the functions?
        no_breaking_changes = (file_version[1] == PETLION_VERSION[1]) && (file_version[2] == PETLION_VERSION[2])

        if !no_breaking_changes
            @warn "Breaking updates encountered: re-evaluating model..."
            remove_model_files(p)
        end

    else
        no_breaking_changes = false
    end

    if files_exist && no_breaking_changes && options[:SAVE_SYMBOLIC_FUNCTIONS]
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

function generate_functions_symbolic(p::AbstractParam; verbose=options[:SAVE_SYMBOLIC_FUNCTIONS])
    
    if verbose println("Creating the functions for $p\n\nMay take a few minutes...") end

    θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, X_applied, γ_sym, p_sym, θ_keys = get_symbolic_vars(p)

    ## Y0 function
    if verbose println("1/4: Making initial guess function") end
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, X_applied)

    ## batteryModel function
    if verbose println("2/4: Making symbolic model") end
    res = _symbolic_residuals(p_sym, t_sym, Y_sym, YP_sym)

    θ_sym_slim, θ_keys_slim = get_only_θ_used_in_model(θ_sym, θ_keys, res, Y0_sym)
    
    Y0Func = build_function(Y0_sym, SOC_sym, θ_sym_slim, X_applied, fillzeros=false, checkbounds=false)[2]

    ## jacobian
    if verbose println("3/4: Making symbolic Jacobian") end
    Jac, jacFunc, J_sp = _symbolic_jacobian(p, res, t_sym, Y_sym, YP_sym, γ_sym, θ_sym, θ_sym_slim, verbose=verbose)

    if verbose println("4/4: Making initial condition functions") end
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, t_sym, Y_sym, YP_sym, θ_sym, θ_sym_slim)

    jac_algFunc = _symbolic_initial_conditions_jac(p, Jac, t_sym, Y_sym, YP_sym, γ_sym, θ_sym_slim)
    
    J_y_sp = (findnz(J_sp)..., p.N.tot-1, p.N.tot)
        
    θ_keys = θ_keys_slim

    if options[:SAVE_SYMBOLIC_FUNCTIONS]
        dir = strings_directory_func(p; create_dir=true) * "/"
        
        ## file info
        str = model_info(p)
 
        open(dir * "info.txt", "w") do f
            write(f, str)
        end

        save_string(x) = (string(x) |> remove_comments |> rearrange_if_statements |> join)
        
        write(dir * "initial_guess.jl", save_string(Y0Func))
        write(dir * "J_y.jl",           save_string(jacFunc))
        write(dir * "f_alg.jl",         save_string(res_algFunc))
        write(dir * "J_y_alg.jl",       save_string(jac_algFunc))
        write(dir * "f_diff.jl",        save_string(res_diffFunc))
            
        @save dir * "J_sp.jl" J_y_sp θ_keys
    end

    if verbose println("Finished\n") end

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
    
    Y0_sym = guess_init(p, X_applied)[1]

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

function _symbolic_jacobian(p::AbstractParam=Params(LCO);inds::T=1:p.N.tot) where T<:UnitRange{Int64}
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

function get_symbolic_vars(p::AbstractParam;
    original_keys=nothing)
    if isnothing(original_keys)
        θ_keys = sort!(Symbol.(keys(p.θ)))
    else
        θ_keys = deepcopy(original_keys)
        all_keys = sort!(Symbol.(keys(p.θ)))
        
        @inbounds for key in all_keys
            if !(key ∈ θ_keys)
                push!(θ_keys, key)
            end
        end
    end

    @variables Y[1:p.N.tot], YP[1:p.N.tot], t, SOC, X_applied, γ
    @variables θ[1:length(θ_keys)]

    Y  = collect(Y)
    YP = collect(YP)
    θ  = collect(θ)

    p = param_skeleton([convert(_type,deepcopy(getproperty(p,field))) for (field,_type) in zip(fieldnames(param_skeleton),fieldtypes(param_skeleton))]...)
    p.opts.SOC = SOC

    @inbounds for (i,key) in enumerate(θ_keys)
        p.θ[key] = θ[i]
    end

    return θ, Y, YP, t, SOC, X_applied, γ, p, θ_keys
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

remove_comments(str::String,x...) = remove_comments(collect(str),x...)
function remove_comments(str::Vector{Char},first::String="#=",last::String="=#")
    """
    An annoying aspect of Symbolics is that saving the output of `build_function` has tons of comments,
    severely adding to the memory size of the stored functions. This removes all comments
    """
    
    ind_first = find_in_string(str, first, 1)
    ind_last = find_in_string(str, last, length(last))

    @assert length(ind_first) === length(ind_last)

    ind_first = reverse(ind_first)
    ind_last  = reverse(ind_last)

    @inbounds for (l,u) in zip(ind_first,ind_last)
        index = l:u
        deleteat!(str,index)
    end
    return str
end

find_in_string(str::String,x...) = find_in_string(collect(str),x...)
function find_in_string(str::AbstractVector,x::String,I::T=1:length(x)) where T
    N = length(str)

    m = length(x)
    
    ind_vec = T[]
    @inbounds for i in 1:N - (m-1)
        index = i:(i+(m-1))
        y = @views @inbounds join(str[index])
        if y == x
            push!(ind_vec, (@inbounds index[I]))
        end
    end
    return ind_vec
end

function replace_repeated(str,x...)
    len_prev = length(str)
    len_new = 0
    @inbounds while len_prev != len_new
        len_prev = length(str)
        str = replace(str, x...)
        len_new = length(str)
    end
    return str
end

find_next(str::String,x...;kw...) = find_next(collect(str),x...;kw...)
function find_next(str,first,x::AbstractArray;itermax=900000)
    ind = first .+ (0:length(x)-1)
    iter = 0
    while (@views @inbounds str[ind]) != x
        ind = ind .+ 1
        iter += 1
        if ind[end] > length(str) error("Couldn't find the string") end
    end
    return ind
end
function find_next(str,first,x;itermax=900000)
    ind = first
    iter = 0
    while (@inbounds str[ind]) != x
        ind = ind .+ 1
        iter += 1
        if iter === itermax error(str[ind]) end
    end
    return ind
end

rearrange_if_statements(str::String) = rearrange_if_statements(collect(str))
function rearrange_if_statements(str::Vector{Char})
    inds = find_in_string(str, "if")
    @inbounds for ind_start in reverse(inds), _ in 1:4
        ind = find_next(str,ind_start[1] .+ (-1:0),['\n', ' '])
        while str[ind] == ['\n', ' ']
            deleteat!(str, ind[end])
        end

        str[ind[1]] = ';'
    end
    str = join(str)

    return str
end

convert_to_ifelse(str::String) = convert_to_ifelse(collect(str))
function convert_to_ifelse(str::Vector{Char})
    
    str = join(str)
    str = replace(str, "; else; " => ", ")
    str = replace(str, "; end" => ")")
    str = replace(str, ";" => ",")
    str = replace(str, "if" => "IfElse.ifelse(")

    return str
end
