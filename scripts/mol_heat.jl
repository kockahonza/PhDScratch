using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
# Method of Manufactured Solutions: exact solution
u_exact = (x, t) -> exp.(-t) * cos.(x)

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ cos(x),
    u(t, 0) ~ exp(-t),
    u(t, 1) ~ exp(-t) * cos(1)]

function make_pde(tmax=1.0, L=1.0)
    # Space and time domains
    domains = [
        t ∈ Interval(0.0, tmax),
        x ∈ Interval(0.0, L)
    ]

    # PDE system
    PDESystem(eq, bcs, domains, [t, x], [u(t, x)]; name=:pdesys)
end

# Method of lines discretization
function heat_discretize(pdesys, dx=0.1)
    discretization = MOLFiniteDifference([x => dx], t)
    #
    # Convert the PDE problem into an ODE problem
    discretize(pdesys, discretization)
end

# Solve ODE problem
using OrdinaryDiffEq
heat_solve(dsc) = solve(dsc, Tsit5(), saveat=0.2)

# Plot results and compare with exact solution
using GLMakie
function plotsol(sol)
    discrete_x = sol[x]
    discrete_t = sol[t]
    solu = sol[u(t, x)]

    f = Figure()
    ax = Axis(f[1, 1])

    for i in eachindex(discrete_t)
        scatterlines!(ax, discrete_x, solu[i, :]; label="t=$(discrete_t[i])")
        # scatter!(ax, discrete_x, u_exact(discrete_x, discrete_t[i]), label="Exact, t=$(discrete_t[i])")
    end
    axislegend(ax)

    f
end
