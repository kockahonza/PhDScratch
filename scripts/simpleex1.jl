using Agents
using StaticArrays
using PyFormattedStrings
using GLMakie

################################################################################
# Define the main and somewhat general building blocks
################################################################################
@agent struct Cell(GridAgent{3})
    size::Vector{Float64}
    type::Int
    willbud::Bool
    dead::Bool
end

struct ModelProperties{N} # N is the number of resources
    dt::Float64 # the finite difference spacings
    dx::Float64
    resource_grid_factor::Int # specifies how many space grid cells make one resource one
    # physics info
    dependencies::Array{Bool,3} # this is num types by num resources by (uptake, release)
    type_deathrates::Vector{Float64}
    type_reproductionrates::Vector{Float64}
    resource_vmaxs::Vector{Float64}
    resource_kms::Vector{Float64}
    resource_alphas::Vector{Float64}
    resource_gammas::Vector{Float64}
    resource_spacesize::Tuple{Int,Int,Int} # the resource grid dimensions
    resource_Ds::SVector{N,Float64} # the diffusion parameters for each resource
    resources::SVector{N,Array{Float64,3}} # matrices of available resources
    # internals
    res_grid_dx::Float64
    resources_temp::SVector{N,Array{Float64,3}} # just a preallocated temp var
    resources_temp2::SVector{N,Array{Float64,3}} # just a preallocated temp var
    resource_betas::SVector{N,Float64} # master equation diffusion coeffs
end
function make_model(space_size, dt, dx,
    dependencies, type_deathrates, type_reproductionrates,
    resource_vmaxs, resource_kms, resource_alphas, resource_gammas,
    resource_grid_factor, resource_Ds, resources;
    (step!)=step1!
)
    dims = (space_size, space_size, space_size)
    space = GridSpace(dims; periodic=false, metric=:chebyshev)

    if !all(x -> x == 0, mod.(spacesize(space), resource_grid_factor))
        throw(ArgumentError("space dimensions are not all divisible by resource_grid_factor"))
    end
    res_spacesize = div.(spacesize(space), resource_grid_factor)

    resources_ = map(resources) do res
        if isa(res, Array)
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
    betas = resource_Ds ./ ((res_grid_dx^2) / dt)
    properties = ModelProperties(dt, dx, dependencies,
        type_deathrates, type_reproductionrates, resource_vmaxs, resource_kms, resource_alphas, resource_gammas,
        resource_grid_factor, res_spacesize, resource_Ds, resources_,
        res_grid_dx, map(similar, resources_), map(similar, resources_), betas
    )

    StandardABM(Cell, space;
        (model_step!)=step!,
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
"""
Only does diffusion of each resource according to its coefficient on the coarser
resource grid.
"""
function mprep_diffusion!(model)
    for (res, temp, beta) in zip(model.resources, model.resources_temp, model.resource_betas)
        for x in 1:model.resource_spacesize[1]
            for y in 1:model.resource_spacesize[2]
                for z in 1:model.resource_spacesize[3]
                    temp[x, y, z] = res[x, y, z]
                    if x != 1
                        temp[x, y, z] += beta * (res[x-1, y, z] - res[x, y, z])
                    end
                    if x != model.resource_spacesize[1]
                        temp[x, y, z] += beta * (res[x+1, y, z] - res[x, y, z])
                    end
                    if y != 1
                        temp[x, y, z] += beta * (res[x, y-1, z] - res[x, y, z])
                    end
                    if y != model.resource_spacesize[2]
                        temp[x, y, z] += beta * (res[x, y+1, z] - res[x, y, z])
                    end
                    if z != 1
                        temp[x, y, z] += beta * (res[x, y, z-1] - res[x, y, z])
                    end
                    if x != model.resource_spacesize[3]
                        temp[x, y, z] += beta * (res[x, y, z+1] - res[x, y, z])
                    end
                    if temp[x, y, z] < 0.0
                        @error f"Getting a negative resource value of {temp[x,y]}! This is very likely a finite difference issue from dt being too big or dx too small"
                    end
                end
            end
        end
        res .= temp
    end
end

const inplane_offsets = [(0, -1, 0), (-1, -1, 0), (-1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, -1, 0)]
const offplane_offsets = [(0, 0, 1), (0, -1, 1), (-1, -1, 1), (-1, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1), (1, -1, 1)]

function step1!(model)
    mprep_diffusion!(model)

    deps = model.dependencies

    # do uptake, release
    for nn in 1:size(model.dependencies)[2]
        for cell in allagents(model)
            if cell.dead
                continue
            end
            cell_pos = real_to_res_pos(cell.pos, model)
            if deps[cell.type, nn, 1]
                uptake = (model.resource_vmaxs[nn] * model.resources[cell_pos] * model.dt) / (model.resources[cell_pos] + model.resource_kms[nn])
                cell.size[nn] += uptake
                model.resources[nn][cell_pos] -= uptake
            end
            if deps[cell.type, nn, 2]
                release = model.resource_gammas[nn] * model.dt
                model.resources[nn][cell_pos] += release
            end
        end
    end

    # do life/death
    for cell in allagents(model)
        if cell.dead
            continue
        end
        if rand(abmrng(model)) < model.type_deathrates[cell.type] * model.dt
            cell.dead = true
        else
            cell.willbud = true
            for nn in 1:size(model.dependencies)[2]
                if deps[cell.type, nn, 1] && cell.size[nn] < model.resource_alphas[nn]
                    cell.willbud = false
                    break
                end
            end
        end
    end

    for cell in allagents(model)
        if cell.willbud
            new_pos = nothing
            for offset in shuffle(inplane_offsets)
                pos = cell.pos .+ offset
                if isempty(pos, model)
                    new_pos = pos
                    break
                end
            end
            for offset in shuffle(offplane_offsets)
                pos = cell.pos .+ offset
                if isempty(pos, model)
                    new_pos = pos
                    break
                end
            end
            if !isnothing(new_pos)
                cell.size .= 0.0
                replicate!(cell, model; pos=new_pos)
            else
                cell.dead = true
            end
        end
    end
end

################################################################################
# Some running examples
################################################################################
function mmtest()
    dependencies = zeros(2, 2, 2)
    dependencies[1, 1, 1] = true
    dependencies[1, 2, 2] = true

    dependencies[2, 2, 1] = true
    dependencies[2, 1, 2] = true

    alphas = [5.4, 3.1]
    km = [2.1e6, 1.3e6]
    gamma = [0.4, 0.26]

    reproduction = [0.51, 0.44]
    death = [0.021, 0.015]

    vm = reproduction .* alphas

    model = make_model(100, 0.001, 1.0, dependencies, death, reproduction, vm, km, alphas, gamma, 5, [0.1, 0.1], [0.0, 0.0])

    # for _ in 1:num_s2
    #     add_agent!(model, 0.0, 1.0, 2, 1, 1.0, 1.0, pp)
    # end

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
