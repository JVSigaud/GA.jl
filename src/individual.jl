using Random,StatsBase


# mutable struct chromossome
#     bit::BitVector
#     fen::Int64
#     fitness::Float64
# end




function BIN(data::BitVector)
    return sum(data .* (2 .^ reverse(collect(range(0,length(data)-1)))))
end


Roulette!(fit::Vector{Float64},R::Vector{Float64},F::Function) = F == max  ? (R .= fit ./ sum(fit) ) : (R .= (1 .- ( fit ./ sum(fit))) ./( length(fit) - 1)) 



Base.@kwdef mutable struct Population
    populationSize::Int
    constraint::Int
    
    gen::Vector{BitVector}
    fen::Vector{Float64}

    roulette::Vector{Float64}
    fitness::Vector{Float64}
    fit::Tuple
    RNG::AbstractRNG
    
    
    function Population(constraint::T, populationSize::T, ObjFunction::Function,opt::Function, decoder::Function = BIN,RNG::AbstractRNG = Random.default_rng(),selection::Function = Roulette!) where T<:Int
        gen::Vector{BitVector} = [rand(RNG,constraint) .> 0.5 for i in 1:populationSize]
        fen::Vector{Float64} = decoder.(gen)
        fitness::Vector{Float64} = ObjFunction.(fen)
        
        fit::Tuple = opt == max  ? findmax(fitness) : findmin(fitness)
        
        roulette::Vector{Float64} = Vector{Float64}(undef, populationSize)
        Roulette!(fitness,roulette,opt) 
        
        
        new(populationSize,constraint,
        gen,fen,roulette,fitness,
        fit,RNG
        )
    end
    
end
function crossover!(state::Population,crossoverRate::Float64,RNG::AbstractRNG)
    childs::Vector{BitVector} = similar(state.gen)
    var1::BitVector = similar(state.gen[1])
    var2::BitVector = similar(state.gen[1])

    for i in collect(range(1,state.populationSize,step = 2 ))
        if rand() < crossoverRate
            var1,var2 = sample(RNG,state.gen,Weights(state.roulette),2)
            SPX!(var1,var2,RNG)

            childs[i] = var1
            childs[i+1] = var2
        else
            childs[i] = var1
            childs[i+1] = var2
        end

    state.gen = childs
    

    end

end

function SPX!(var1::BitVector,var2::BitVector,RNG::AbstractRNG)
    point::Int = rand(RNG,collect(range(1,length(var1))))
    var1 = vcat(var1[1:point-1],var2[point:end])
    var2 = vcat(var2[1:point-1],var1[point:end])
end

function mutation()
end

struct GA
    Pop_size::Int
    mutationRate::Float64
    crossoverRate::Float64

    Obj_function::Function
    decoder::Function
    
    Id::Int
    # history::Dict{Int,Population}
    opt::Function
    RNG::AbstractRNG

    state::Population 

    function GA(Pop_size::Int,constraint::Int,
        mutationRate::Float64, crossoverRate::Float64,
        Obj_function::Function, decoder::Function,opt::Function,
        Id::Int=1,RNG::AbstractRNG = Random.default_rng(),
        state::Population = Population(constraint,populationSize,Obj_function,opt))

        new(Pop_size,mutationRate,crossoverRate,Obj_function,decoder,Id,opt,RNG,state)
    end

end


function Optimize!(model::GA)
    
    crossover!(model.state,model.crossoverRate,model.RNG)
    
   


end

RNG = MersenneTwister(42)
constraint = 5
populationSize = 10
mutationRate = 0.1
crossoverRate = 0.8


f(x) = x^2


x = GA(populationSize,constraint,mutationRate,crossoverRate,f,BIN,min)

x.state.roulette

x.state.roulette
x.state 

Optimize!(x)

for i in collect(range(1,x.state.populationSize,step = 2 ))
    println(i)
    println(i+1)
end


v1,v2 = sample(x.state.gen,Weights(x.state.roulette),2)

v1
v1[2:end]
v2[2:end]
v1[1:2-1]
v1

v2

vcat(v1[1:2-1],v2[2:end])

vcat(v2[1:2-1],v1[2:end])

for i in 1:10
    println(rand(RNG,[1,2,3,4,5,6]))
end
# b=[rand(rng,constraint) .> 0.5 for i in 1:populationSize]

# x.Pop
# b[1]
# length(b[1])
# sum(b[1] .* (2 .^ reverse(collect(range(0,4)))))

# b[1]
# map(x->BIN,b)
# x=BIN.(b)



# y = f.(x)

# typeof(findmax(y))

