using Agents
using StaticArrays
using PyFormattedStrings
using GLMakie

################################################################################
# Define the main and somewhat general building blocks
################################################################################
@agent struct Cell(GridAgent{2})
    size::Float64 # starts at 0 and growing to 1. will reset to 0 and replicate itself
    growth_per_consumed_unit::Float64 # increases size by this * the amount of consumed resource
    consumes::Int # specifies which resource is consumed/produced
    produces::Int
    consumption_rate::Float64 # specifies the rate or consumption/production,
    production_rate::Float64  # aka this much is cons./prod. per 1 unit of time
    random_move_rate::Float64 # the rate at which the cell randomly moves perunit time
end

struct ModelProperties{N} # N is the number of resources
    dt::Float64 # the finite difference spacings
    dx::Float64
    resource_grid_factor::Int # specifies how many space grid cells make one resource one
    resource_spacesize::Tuple{Int,Int} # the resource grid dimensions
    resource_Ds::SVector{N,Float64} # the diffusion parameters for each resource
    resources::SVector{N,Matrix{Float64}} # matrices of available resources
    # internals
    res_grid_dx::Float64
    resources_temp::SVector{N,Matrix{Float64}} # just a preallocated temp var
    resource_alphas::SVector{N,Float64} # master equation diffusion coeffs
end
function make_model(space, dt, dx,
    resource_grid_factor, resource_Ds, resources;
    (agent_step!)=cell_step!,
    (model_step!)=mstep_diffusion!
)
    if !all(x -> x == 0, mod.(spacesize(space), resource_grid_factor))
        throw(ArgumentError("space dimensions are not all divisible by resource_grid_factor"))
    end
    res_spacesize = div.(spacesize(space), resource_grid_factor)

    resources_ = map(resources) do res
        if isa(res, Matrix)
            if size(res) == res_spacesize
                res
            else
                throw(ArgumentError(f"passed resource matrix {res} does not have the correct shape"))
            end
        elseif isa(res, AbstractFloat)
            fill(res, res_spacesize)
        else
            throw(ArgumentError(f"cannot interpret {res} as a resource"))
        end
    end

    res_grid_dx = resource_grid_factor * dx
    alphas = resource_Ds ./ ((res_grid_dx^2) / dt)
    properties = ModelProperties(dt, dx, resource_grid_factor, res_spacesize, resource_Ds, resources_,
        res_grid_dx, map(similar, resources_), alphas
    )

    StandardABM(Cell, space;
        agent_step!,
        model_step!,
        properties
    )
end

"""
For a given agent space grid position returns the coarser resource grid position
"""
real_to_res_pos(rgf::Integer, pos) = div.(pos, rgf, RoundUp)
real_to_res_pos(model, pos) = real_to_res_pos(model.resource_grid_factor, pos)

################################################################################
# The evolution steps
################################################################################
function cell_step!(cell, model)
    res_pos = real_to_res_pos(model, cell.pos)

    # produce
    model.resources[cell.produces][res_pos...] += cell.production_rate * model.dt

    # consume and grow
    consumed = min(cell.consumption_rate * model.dt, model.resources[cell.consumes][res_pos...])
    model.resources[cell.consumes][res_pos...] -= consumed
    cell.size += consumed * cell.growth_per_consumed_unit

    # maybe replicate
    if cell.size >= 1.0
        cell.size = 0.0
        replicate!(cell, model)
    end

    # randomly move
    if rand(abmrng(model)) < (cell.random_move_rate * model.dt)
        dir = rand(abmrng(model), SA[(1, 0), (-1, 0), (0, 1), (0, -1)])
        walk!(cell, dir, model)
    end
end

"""
Only does diffusion of each resource according to its coefficient on the coarser
resource grid.
"""
function mstep_diffusion!(model)
    for (res, temp, alpha) in zip(model.resources, model.resources_temp, model.resource_alphas)
        for x in 1:model.resource_spacesize[1]
            for y in 1:model.resource_spacesize[2]
                temp[x, y] = res[x, y]
                if x != 1
                    temp[x, y] += alpha * (res[x-1, y] - res[x, y])
                end
                if x != model.resource_spacesize[1]
                    temp[x, y] += alpha * (res[x+1, y] - res[x, y])
                end
                if y != 1
                    temp[x, y] += alpha * (res[x, y-1] - res[x, y])
                end
                if y != model.resource_spacesize[2]
                    temp[x, y] += alpha * (res[x, y+1] - res[x, y])
                end
                if temp[x, y] < 0.0
                    @error f"Getting a negative resource value of {temp[x,y]}! This is very likely a finite difference issue from dt being too big or dx too small"
                end
            end
        end
        res .= temp
    end
end

################################################################################
# Some running examples
################################################################################
"""
This makes the basic 2 interdependent species model
"""
function mmtest(dt, dx;
    space_size=100,
    rgf=5,
    num_s1=10,
    num_s2=10,
    res1_D=1.0,
    res2_D=1.0,
    pp=10
)
    dims = (space_size, space_size)
    space = GridSpace(dims; periodic=false, metric=:chebyshev)

    model = make_model(space, dt, dx, rgf, SA[res1_D, res2_D], SA[0.0, 0.0])

    for _ in 1:num_s1
        add_agent!(model, 0.0, 1000.0, 1, 2, 1.0, 1.0, pp)
    end

    for _ in 1:num_s2
        add_agent!(model, 0.0, 1.0, 2, 1, 1.0, 1.0, pp)
    end

    model
end

function runmmtest()
    m = mmtest(0.001, 1.0; space_size=200, pp=50, res1_D=100.0, res2_D=100.0)
    # xx = abmexploration(m; add_controls=true, agent_color=c -> c.consumes)
    xx = abmexploration(m; add_controls=true, agent_color=c -> c.consumes,
        adata=[(c -> c.consumes == 1, count), (c -> c.consumes == 2, count)],
        mdata=[m -> sum(m.resources[1]), m -> sum(m.resources[2])],
        alabels=["#consumers of 1", "#consumers of 2"],
        mlabels=["total res 1", "total res 2"],
    )
    display(xx[1])
end

# function custom_abmexplore(m)
# end
