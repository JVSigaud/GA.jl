using Random,StructArrays
using Plots
using BenchmarkTools

struct Interval{T}  
    inicio::T
    final::T

end

s = StructArray{Chromossome}(undef, 2)

typeof(s)




function rand!(Obj::StructArray{Chromossome},constraint::Int64,num_var::Int64,interval::Interval,decoder::Function,ObjFunction::Function)

    @inbounds @simd for i in eachindex(Obj.gen)
        Obj.gen[i] = rand(constraint*numvar) .>= 0.5
        Obj.fen[i] = decoder(Obj.gen[i],fen,interval,constraint,num_var)
        Obj.fitness[i] = ObjFunction(fen) 
    end
end

function BIN1D(data::AbstractArray)
    expoentes::Int32 = length(data) - 1
    sum = zero(Float64)
    @inbounds for i in eachindex(data)
        sum += data[i] * (2 ^ expoentes)
        expoentes -= 1

    end

    return sum
end

function BIN(data::BitVector,constraint::Int64,num_var::Int64)
    fen::Vector{Float64} = Vector{Float64}(undef,num_var)
    @inbounds for i in 0:num_var-1
        fen[i+1] = BIN1D(@view data[i*constraint + 1:(i+1)*constraint])
    end
    return fen
end



rand!(s,10,1)
b = BitVector((0,0,0,1,1,1,1,1,1,1))
@btime BIN1D(b)

c = BitVector((1,1,1,1,1,1,1,1,1,1))

@btime BIN(c,5,2)




struct Chromossome
    
    gen::BitVector
    fen::Vector{Float64}
    fitness::Float64

    function Chromossome(gen::BitVector,constraint::Int64,num_var::Int64,interval::Interval,decoder::Function,ObjFunction::Function)

        fen::Vector{Float64} = zeros(Float64,num_var)
        decoder(gen,fen,interval,constraint,num_var)
        fitness::Float64 = ObjFunction(fen)

        new(gen,fen,fitness)

    end

    function Chromossome(gen::BitVector,fen::Vector{Float64},fitness::Float64)
        new(gen,fen,fitness)
    end

end

function BIN(data::BitVector)
    return sum(data .* (2 .^ reverse(collect(range(0,length(data)-1)))))
end

function BIN(data::BitVector,constraint::Int,num_var::Int)
    vec::Vector{BitVector} = [data[i*constraint + 1:(i+1)*constraint] for i in 0:num_var-1]
    fen::Vector{Float64} = zeros(Float64,num_var)
    fen .= BIN.(vec)

end


Roulette!(fit::Vector{Float64},R::Vector{Float64},F::Function) = F == max  ? (R .= fit ./ sum(fit) ) : (R .= (1 .- ( fit ./ sum(fit))) ./( length(fit) - 1)) 
# Roulette!(fit::Vector{Float64},R::Vector{Float64}) = (R .= fit ./ sum(fit) )



Base.@kwdef mutable struct Population
    populationSize::Int
    constraint::Int
    
    gen::Vector{BitVector}
    fen::Vector

    roulette::Vector{Float64}
    fitness::Vector{Float64}
    fit::Tuple
    RNG::AbstractRNG
    
    
    function Population(constraint::T, populationSize::T,num_var::T, ObjFunction::Function,opt::Function, decoder::Function = BIN,RNG::AbstractRNG = Random.default_rng(),selection::Function = Roulette!) where T<:Int
        gen::Vector{BitVector} = [rand(RNG,constraint*num_var) .> 0.5 for i in 1:populationSize]
        fen::Vector = num_var == 1 ? decoder.(gen) : decoder.(gen,constraint,num_var)
        
        fitness::Vector{Float64} = ObjFunction.(fen)
        # fitness::Vector{Float64} = opt == max ? ObjFunction.(fen) : 1 ./ ObjFunction.(fen) 
        
        fit::Tuple = opt == max  ? findmax(fitness) : findmin(fitness)
        # fit::Tuple = findmax(fitness)
        
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

    pos::Vector{Int64} = collect(range(1,state.populationSize))
    rand_pos1::Int64 = one(Int64)
    rand_pos2::Int64 = one(Int64)

    sample!(RNG,state.gen,Weights(state.roulette),childs)

    # println(childs)

    for i in collect(range(1,state.populationSize,step = 2 ))
        # println(pos)
        rand_pos1 = rand(RNG,pos)
        deleteat!(pos,findall(x->x==rand_pos1,pos))
        rand_pos2 = rand(RNG,pos)
        deleteat!(pos,findall(x->x==rand_pos2,pos))
        # println(rand_pos2)
        # println(rand_pos1)
        var1 = childs[rand_pos1]
        var2 = childs[rand_pos2]
        if rand() < crossoverRate
            var1,var2 = SPX(var1,var2,RNG)

            state.gen[i] = var1
            state.gen[i+1] = var2
        else
            state.gen[i] = var1
            state.gen[i+1] = var2
        end
    end

end;


function SPX(var1::T,var2::T,RNG::AbstractRNG=Random.default_rng()) where T <: BitVector
    point::Int = rand(RNG,collect(range(1,length(var1))))
    var3::T = vcat(var1[1:point-1],var2[point:end])
    var4::T = vcat(var2[1:point-1],var1[point:end])
    return (var3,var4)
end

function mutation!(state::Population,num_var::Int,mutationRate::Float64,RNG::AbstractRNG)
    point::Int32 = one(Int32)
    vec::Vector{Int64} = collect(range(1,state.constraint*num_var))
    for i in 1:state.populationSize
        if rand() < mutationRate
            point = rand(RNG,vec)
            state.gen[i][point] = state.gen[i][point] == 1 ? 0 : 1 
        end
    end
end

mutable struct GA
    Pop_size::Int
    num_var::Int

    mutationRate::Float64
    crossoverRate::Float64

    ObjFunction::Function
    decoder::Function
    
    Id::Int
    opt::Function
    RNG::AbstractRNG
    
    state::Population 
    history::Dict{Int,Tuple}
    best::Float64

    function GA(Pop_size::Int,constraint::Int,num_var::Int,
        mutationRate::Float64, crossoverRate::Float64,
        ObjFunction::Function, decoder::Function,opt::Function,
        Id::Int=1,RNG::AbstractRNG = Random.default_rng(),
        state::Population = Population(constraint,populationSize,num_var,ObjFunction,opt))
        

        new(Pop_size,num_var,mutationRate,crossoverRate,ObjFunction,decoder,Id,opt,RNG,state,Dict(Id=>(state.fit,mean(state.fitness),state.fit)))
    end

end


function Optimize!(model::GA)
    
    crossover!(model.state,model.crossoverRate,model.RNG)
    
    mutation!(model.state,model.num_var,model.mutationRate,model.RNG)

    model.state.fen .= model.num_var == 1 ? model.decoder.(model.state.gen) : model.decoder.(model.state.gen,model.state.constraint,model.num_var)
    # model.state.fitness .= model.opt == max ? model.ObjFunction.(model.state.fen) : 1 ./ model.ObjFunction.(model.state.fen) 
        
    model.state.fitness .= model.ObjFunction.(model.state.fen) 

    Roulette!(model.state.fitness,model.state.roulette,model.opt)

    model.state.fit = model.opt == max  ? findmax(model.state.fitness) : findmin(model.state.fitness)
    # model.state.fit = findmax(model.state.fitness)

    model.Id += 1
    model.history[model.Id] = (model.state.fit,mean(model.state.fitness))
    return nothing

end




RNG = MersenneTwister(42)
constraint = 10
populationSize = 500

num_var = 10

mutationRate = 0.01
crossoverRate = 0.8


f(x) = x == 0.0 ? 1/(sum([x[i]^2 for i in 1:10]/(10^(-256)))) : 1/(sum([x[i]^2 for i in 1:10]))


g(x) = sum([x[i]^2 for i in 1:10])
x = GA(populationSize,constraint,num_var,mutationRate,crossoverRate,f,BIN,max)


for i in 1:1000
    Optimize!(x)
end



z = collect(1:x.Id)
y = [x.history[i][1][1] for i in z]
m = [x.history[i][2] for i in z]
plot(z,y)
plot!(z,m)


A = [1,2,3,4,5]

B = [0.1,0.1,0.1,0.1,0.6]
x = similar(A)
# function Sample!(vec::Vector{Int},w::Vector{Float64},alloc::Vector{Int})
#     alloc.= sample(vec,Weights(w),length(vec))
# end

# @time [sample(A,Weights(B)) for i in 1:10000000]
# @time sample!(A,Weights(B),x)
# @time Sample!(A,B,x)


# sample(A,Weights(B))
# A
# sample!(
# xx.==5
# sum(xx.==5)


# a = [rand(collect(range(1,100000))) for i in 1:10000]




a = rand(10) .< 0.5
b = rand(10) .< 0.5
println(a,b)

SPX(a,b)
