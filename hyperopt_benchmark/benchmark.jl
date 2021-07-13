using Hyperopt, Nonconvex, NLopt
using ChainRulesCore, ForwardDiff
using Plots
using CUTEst

function ChainRulesCore.rrule(::typeof(CUTEst.obj), nlp::CUTEstModel, x::AbstractVector)
    val = CUTEst.obj(nlp, x)
    grad = CUTEst.grad(nlp, x)
    val, Δ -> (NoTangent(), NoTangent(), Δ * grad)
end

function multistart(options)
    start_time = time()
    epoch_time, epoch_best = [], []
    function callback(results)
        t = time()-start_time
        push!(epoch_best, minimum(results))
        push!(epoch_time, t)
    end
    function f(nlp)
        if options == BOHB_options
            options = HyperoptOptions(sub_options = max_iter -> IpoptOptions(first_order = true, max_iter = max_iter), 
                        sampler = Nonconvex.Hyperband(R=100, η=3, inner=BOHB(dims=[Continuous() for _ in 1:length(nlp.meta.x0)])))
        end
        model = Model(x -> CUTEst.obj(nlp, x))
        addvar!(model, nlp.meta.lvar, nlp.meta.uvar)
        alg = HyperoptAlg(IpoptAlg())
        result = Nonconvex.optimize(model, alg, nlp.meta.x0, options = options, callback=callback)
        result, epoch_time, epoch_best
    end
    return f
end

function get_fn(dims)
    function counting_ones(x::AbstractVector)
        target = 0.
        for (i, _x, dim) in zip(1:length(dims), x, dims)
            if dim isa Continuous
                target += _x
            else
                target += _x == 2 ? 1 : 0
            end
        end
        target
    end
    counting_ones
end

function multistart_counting_ones()
    dims = Array{Hyperopt.DimensionType}([Continuous() for _ in 1:50])
    append!(dims, [Categorical(2) for _ in 1:25])
    append!(dims, [UnorderedCategorical(2) for _ in 1:25])
    fn = get_fn(dims)
    options = HyperoptOptions(sub_options=max_iter->IpoptOptions(first_order=true, max_iter=max_iter), 
                                sampler=Nonconvex.Hyperband(R=100, η=3, inner=BOHB(dims=dims)))    
    model = Model(fn)
    addvar!(model, [0 for _ in 1:100], [1 for _ in 1:100])
    alg = HyperoptAlg(IpoptAlg())
    # define callback
    start_time = time()
    epoch_time, epoch_best = [], []
    function callback(results)
        t = time()-start_time
        push!(epoch_best, minimum(results))
        push!(epoch_time, t)
    end
    # optim
    result = Nonconvex.optimize(model, alg, [0 for _ in 1:100], options=options, callback=callback)
    result, epoch_time, epoch_best
end

function time_vs_regret(algorithm, problem)
    println("problem: $(problem)")
    if problem == "counting_ones"
        result, epoch_time, epoch_best = multistart_counting_ones()
    else
        nlp = CUTEstModel(problem)
        result, epoch_time, epoch_best = multistart(algorithm)(nlp)
        finalize(nlp)
    end
    return (result, epoch_time, epoch_best)
end

problems =
["counting_ones",
"DALLASL",
"HIER13",
"HIER16",
"READING3",
"READING1",
"SOSQP2",
"SREADIN3",
"SOSQP1",
"ZAMB2",]


random_options = HyperoptOptions(sub_options = IpoptOptions(), sampler = Hyperopt.RandomSampler())
LH_options = HyperoptOptions(sub_options = IpoptOptions(), sampler = Nonconvex.LHSampler())
hyperband_options = HyperoptOptions(sub_options = max_iter -> IpoptOptions(first_order = true, max_iter = max_iter), 
                            sampler = Hyperopt.Hyperband(R=100, η=3, inner=Hyperopt.RandomSampler()))
BOHB_options = HyperoptOptions(sub_options = max_iter -> IpoptOptions(first_order = true, max_iter = max_iter), sampler = Hyperopt.Hyperband(R=100, η=3, inner=Hyperopt.RandomSampler()))

algorithms = Dict( "hyperband" => hyperband_options, "BOHB" => BOHB_options, "random" => random_options, "LH" => LH_options)

results = Dict()
epoch_times = Dict()
epoch_bests = Dict()
for problem in problems
    plot(title=problem, xaxis=:log, xlabel="time elapsed", ylabel="regret")
    for algorithm in keys(algorithms)
        println("algorithm: $(algorithm), problem $(problem): ")
        linewidth = (problem ∈ ["random", "LH"] ? 10 : 2)
        results[algorithm, problem], epoch_times[algorithm, problem], epoch_bests[algorithm, problem] = time_vs_regret(algorithms[algorithm], problem)
        plot!(epoch_times[algorithm, problem], epoch_bests[algorithm, problem], label=algorithm, lw=linewidth)
    end
    savefig("$(problem).svg")
end
