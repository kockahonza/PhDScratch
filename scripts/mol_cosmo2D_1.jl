using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets

using GLMakie

################################################################################
# The most physicsy bit
################################################################################
# Space/time params
@parameters t x y
# Physical params
@parameters Ds mu1 mu2 nu1 nu2 # strain only params
@parameters Dn d1 d2 # nutrient only params
@parameters r11 r22 v21 v12 K21 K12 t21 t12 # the non-zero "interaction" params
params_list = [Ds, mu1, mu2, nu1, nu2, Dn, d1, d2, r11, r22, v21, v12, K21, K12, t21, t12]
# Fields
@variables N1(..) N2(..) S1(..) S2(..) c21(..) c12(..)
fields = [N1, N2, S1, S2, c21, c12]

Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
lap(u) = Dxx(u) + Dyy(u)

# 1D PDE and boundary conditions
eqN1 = Dt(N1(t, x, y)) ~ Ds * lap(N1(t, x, y)) + mu1 * N1(t, x, y) - nu1 * N1(t, x, y)
eqN2 = Dt(N2(t, x, y)) ~ Ds * lap(N2(t, x, y)) + mu2 * N2(t, x, y) - nu2 * N2(t, x, y)
eqS1 = Dt(S1(t, x, y)) ~ Dn * lap(S1(t, x, y)) + r11 * N1(t, x, y) - N2(t, x, y) * v12 * (S1(t, x, y) / (S1(t, x, y) + K12)) - d1 * S1(t, x, y)
eqS2 = Dt(S2(t, x, y)) ~ Dn * lap(S2(t, x, y)) + r22 * N2(t, x, y) - N1(t, x, y) * v21 * (S2(t, x, y) / (S2(t, x, y) + K21)) - d2 * S2(t, x, y)
eqc21 = Dt(c21(t, x, y)) ~ v21 * (S2(t, x, y) / (S2(t, x, y) + K21)) - mu1 * N1(t, x, y)
eqc12 = Dt(c12(t, x, y)) ~ v12 * (S1(t, x, y) / (S1(t, x, y) + K12)) - mu2 * N2(t, x, y)

eqs = [eqN1, eqN2, eqS1, eqS2, eqc21, eqc12]

function heaviside(x)
    if x >= 0
        1.0
    else
        0.0
    end
end
@register_symbolic heaviside(x)

################################################################################
# Make and discretize the model
################################################################################
function make_cosmo_pde(;
    T=Inf, L=1.0,
    init_radius=L / 3,
    init_density=0.5,
    params=Dict(),
)
    # Periodic boundary conditions
    bcs = []
    for f in fields
        push!(bcs, f(t, 0, y) ~ f(t, L, y))
        push!(bcs, f(t, x, 0) ~ f(t, x, L))
    end
    # intial conditions
    for f in [S1, S2, c21, c12] # nutrients start at 0.
        push!(bcs, f(0, x, y) ~ 0.0)
    end
    for f in [N1, N2]
        # strains start at the value of init_density within a circle of radius
        # init_radius positioned at the middle
        m = L / 2
        push!(bcs, f(0, x, y) ~ heaviside(init_radius - sqrt((x - m)^2 + (y - m)^2)) * init_density)
    end

    # Space and time domains
    domains = [
        t ∈ Interval(0.0, T),
        x ∈ Interval(0.0, L),
        y ∈ Interval(0.0, L)
    ]

    # PDE system
    dvs = [f(t, x, y) for f in fields]
    default_params = Dict(
        Ds => 1.0, mu1 => 0.0, mu2 => 0.0, nu1 => 0.0, nu2 => 0.0,
        Dn => 1.0, d1 => 0.0, d2 => 0.0,
        r11 => 1.0, r22 => 1.0, v21 => 1.0, v12 => 1.0, K21 => 1.0, K12 => 1.0, t21 => 1.0, t12 => 1.0
    )
    PDESystem(eqs, bcs, domains, [t, x, y], dvs, params_list;
        defaults=default_params, name=:pdesys
    )
end
function make_params_from_default()
    final_ps = Vector{Pair{Num,Float64}}(undef, 0)
    for k in keys(default_params)
        push!(final_ps, k => default_params[k])
    end
    final_ps
end

# Method of lines discretization
# TODO: 

################################################################################
# Ploting
################################################################################
function plotcosmosol(sol)
    fig = Figure()
    sg = SliderGrid(fig[1, 1:6], (label="time", range=sol.t))

    ti_obs = sg.sliders[1].selected_index

    axN1 = Axis(fig[2, 1])
    axN1.title = "N1"
    hm = heatmap!(axN1, sol[x], sol[y], lift(ti -> sol[N1(t, x, y)][ti, :, :], ti_obs))
    Colorbar(fig[2, 2], hm)

    axN2 = Axis(fig[3, 1])
    axN2.title = "N2"
    hm = heatmap!(axN2, sol[x], sol[y], lift(ti -> sol[N2(t, x, y)][ti, :, :], ti_obs))
    Colorbar(fig[3, 2], hm)

    axS1 = Axis(fig[2, 3])
    axS1.title = "S1"
    hm = heatmap!(axS1, sol[x], sol[y], lift(ti -> sol[S1(t, x, y)][ti, :, :], ti_obs))
    Colorbar(fig[2, 4], hm)

    axS2 = Axis(fig[3, 3])
    axS2.title = "S2"
    hm = heatmap!(axS2, sol[x], sol[y], lift(ti -> sol[S2(t, x, y)][ti, :, :], ti_obs))
    Colorbar(fig[3, 4], hm)

    axc21 = Axis(fig[2, 5])
    axc21.title = "c21"
    hm = heatmap!(axc21, sol[x], sol[y], lift(ti -> sol[c21(t, x, y)][ti, :, :], ti_obs))
    Colorbar(fig[2, 6], hm)

    axc12 = Axis(fig[3, 5])
    axc12.title = "c12"
    hm = heatmap!(axc12, sol[x], sol[y], lift(ti -> sol[c12(t, x, y)][ti, :, :], ti_obs))
    Colorbar(fig[3, 6], hm)

    fig
end

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
