
"""
TM-ABC-MCMC with Lotka_Volterra model as an example using synthetic data
Morgan_Craig_Lab
Programming language : Julia v"1.10.4"
"""




# ==============================
# Required packages
# ==============================


using XLSX
using DataFrames
using LinearAlgebra
using Statistics
using Distributions
using Distributions: Normal, Uniform
using DelayDiffEq
using StatsBase
using Sundials
using LaTeXStrings
using Measures
using Colors
using JLD2
using PlotlyJS
using CSV, DataFrames
using Base64
using LaTeXStrings
using Interpolations
using Random



# ==============================
# Define the mathematical model
# ==============================


function model!(dydt, y, pp, t)
    #prey = x, predator = y
    prey, predator = y
    a,b = pp
    dydt[1] = a * prey - prey * predator
    dydt[2] = b * prey * predator - predator
end



#= 
==============================
Validation:
Solve the model on [t_start, t_end] with num_points time points, using initial_conditions (y0) and best_fit_params. 
Plot/inspect the state variables of interest and confirm their dynamics are reasonable.
==============================
=# 

function solve_ode(theta, t_start, t_end, num_points)
    y0 = [1, 0.5]
    tspan = (t_start, t_end)
    prob = ODEProblem(model!, y0, tspan, theta)
    t_range = range(t_start, t_end, length=num_points)
    sol = solve(prob, Rosenbrock23(), saveat=range(t_start, t_end, length=num_points), reltol=1e-6, abstol=1e-6)
    return sol.t, sol[1,:], sol[2,:]
end
t_start = 0
t_end = 15
num_points = 1500
best_params = [1,1]
t, X, Y = solve_ode(best_params, t_start, t_end, num_points)

# Uniform time grid
Δt = t_end/num_points
t_sim = Δt .* (1:length(X))


p = plot(
    scatter(x=t_sim, y=X, mode="lines", name="X", line=attr(width=3, color="black"), showlegend=true),
Layout(
    xaxis=attr(title="Time", range=[-0.5, 16], tickvals=0:2:16, ticktext=string.(0:2:16), showline=true, linecolor="black"),
    yaxis=attr(range=[0, 2.5], tickvals=[0, 1, 2], ticktext=["0", "1", "2"], showline=true, linecolor="black"),
    font=attr(family="Arial", size=25),
    width=600, height=500,plot_bgcolor="white", legend=attr(x=0.5, y=1.1, xanchor="center", font=attr(size=16), orientation="h")
))
display(p)
savefig(p, "X-prey.png") 




#= 
==============================
Synthetic data generation

Time grid
- Simulate on [t_start, t_end] with num_points = 1500 → t_sim.
- Observations at 8 time points: take every 200th index → t_sampled (8 points).

Population-level data
- Mean  = Simulation
- Standard error: SE = 0.2 × Mean
- Report Mean ± SE → upper = 1.2*Mean, lower = 0.8*Mean

Individual-level data
- At each sampled time, draw K=5 individuals uniformly in [lower, upper]
  where lower = 0.8*Mean and upper = 1.2*Mean.
- Max = 1.2*Mean, Min = 0.8*Mean at each sampled time;
==============================
=# 

t_step = 200
t_sampled = t_sim[1:t_step:end]


#Population level : Mean
X_mean = [1.0
 1.644753555848245
 0.5237364303531032
 0.7888045519296935
 1.752034252979167
 0.6268522002311171
 0.6385899774360522
 1.5864256673717205]

#Population level/upper bound : Mean + SE
X_upper_bound = [ 1.2
 1.973704267017894
 0.6284837164237239
 0.9465654623156321
 2.1024411035750004
 0.7522226402773405
 0.7663079729232626
 1.9037108008460646]

#Population level/lower bound : Mean - SE
X_lower_bound = [0.8
 1.3158028446785959
 0.4189891442824826
 0.6310436415437548
 1.4016274023833335
 0.5014817601848937
 0.5108719819488418
 1.2691405338973765]






# Plot of population level data
min_max_lines = [
    scatter(
        x = [t, t],
        y = [X_lower_bound[i], X_upper_bound[i]],
        mode = "lines",
        line = attr(color="#5F747A", dash="dash", width=1.5),
        showlegend = false,
        hoverinfo = "skip"
    )
    for (i, t) in enumerate(t_sampled)
]
p = plot(
    vcat(
        [scatter(x=t_sampled, y=X_upper_bound, mode="markers", name="Mean ± SE",
                    marker=attr(color="#5F747A", symbol="line-ew-open", size=10)),
            scatter(x=t_sampled, y=X_lower_bound, mode="markers", showlegend=false,
                    marker=attr(color="#5F747A", symbol="line-ew-open", size=10))],
        min_max_lines,  
        [scatter(x=t_sampled, y=X_mean,
                    mode="markers", name="Mean",
                    marker=attr(color="red", symbol="circle", size=10))]
    ),
    Layout(
        xaxis=attr(title="Time", range=[-0.5, 16], tickvals=0:2:16, ticktext=string.(0:2:16), showline=true, linecolor="black"),
        yaxis=attr(range=[0, 2.5], tickvals=[0, 1, 2], ticktext=["0", "1", "2"], showline=true, linecolor="black"),
        font=attr(family="Arial", size=25),
        width=600, height=500,
        plot_bgcolor="white",
        legend=attr(x=0.5, y=1.1, xanchor="center", font=attr(size=16), orientation="h")
    )
)
display(p)
savefig(p, "Pop_data.png")





#Individual level data
Ind_Data = [ 0.974223  0.813132  1.05383   0.951812  1.04637
 1.84554   1.7868    1.75268   1.47408   1.60357
 0.475012  0.545214  0.478659  0.568133  0.552999
 0.884418  0.877249  0.824552  0.906342  0.702517
 1.58203   1.99343   1.70428   1.47866   1.41655
 0.602427  0.630376  0.527107  0.746579  0.57132
 0.682578  0.659807  0.678128  0.69571   0.726133
 1.43      1.29182   1.7293    1.40829   1.31505]






# Plot of Individual level data
min_max_lines = [
    scatter(
        x=[t, t], y=[X_lower_bound[i], X_upper_bound[i]],
        mode="lines",
        line=attr(color="#5F747A", dash="dash", width=1.5),
        showlegend=false, hoverinfo="skip"
    )
    for (i, t) in enumerate(t_sampled)
]

sample_traces = [
    scatter(
        x=t_sampled, y=Ind_Data[:, j], mode="markers",
        name = (j == 1 ? "Individuals(n=7)" : ""),
        showlegend = (j == 1),
        marker = attr(symbol="circle", size=10, color="red", opacity=0.75)
    )
    for j in 1:5
]

p = plot(
    vcat(
        [scatter(x=t_sampled, y=X_upper_bound, mode="markers", name="Max/Min",
                    marker=attr(color="#5F747A", symbol="circle", size=10)),
            scatter(x=t_sampled, y=X_lower_bound, mode="markers", showlegend=false,
                    marker=attr(color="#5F747A", symbol="circle", size=10))],
        min_max_lines,  
        sample_traces 
    ),
    Layout(
        xaxis=attr(title="Time", range=[-0.5, 16], tickvals=0:2:16, ticktext=string.(0:2:16), showline=true, linecolor="black"),
        yaxis=attr(range=[0, 2.5], tickvals=[0, 1, 2], ticktext=["0", "1", "2"], showline=true, linecolor="black"),
        font=attr(family="Arial", size=25),
        width=600, height=500,
        plot_bgcolor="white",
        legend=attr(x=0.5, y=1.1, xanchor="center", font=attr(size=16), orientation="h")
    )
)

display(p)
savefig(p, "Ind_data.png")







# Plot synthetic data and simulation
p = plot(
    [
        scatter(x=t_sim, y=X, mode="lines", name="X", line=attr(width=3, color="black")),
        scatter(x=t_sampled, y=X_upper_bound, mode="markers", name="Data", marker=attr(color="#5F747A", symbol="circle", size=10)),
        scatter(x=t_sampled, y=X_lower_bound, mode="markers", showlegend = false, marker=attr(color="#5F747A", symbol="circle", size=10))
    ],
    Layout(
        xaxis=attr(title="Time", range=[-0.5, 16], tickvals=0:2:16, ticktext=string.(0:2:16), showline=true, linecolor="black"),
        yaxis=attr(range=[0, 2.5], tickvals=[0, 1, 2], ticktext=["0", "1", "2"], showline=true, linecolor="black"),
        font=attr(family="Arial", size=25),
        width=600, height=500,
        plot_bgcolor="white", legend=attr(x=0.5, y=1.1, xanchor="center", font=attr(size=16), orientation="h")
    )
)
display(p)
savefig(p, "Sim and data.png") 




# Model definition reused in the loop (N iterations)

function model!(dydt, y, pp, t)
    prey, predator = y
    a,b = pp
    dydt[1] = a * prey - prey * predator
    dydt[2] = b * prey * predator - predator
end
function solve_ode(theta, t_start, t_end, num_points, i)
    y0 = [1, 0.5]
    tspan = (t_start, t_end)
    prob = ODEProblem(model!, y0, tspan, theta)
    t_range = range(t_start, t_end, length=num_points)
    sol = solve(prob, Rosenbrock23(), saveat=range(t_start, t_end, length=num_points), reltol=1e-6, abstol=1e-6)
    return sol.t, sol[1,:], sol[2,:]
end
t_start = 0
t_end = 15
num_points = 1500
t, X_i, Y_i = solve_ode(best_params, t_start, t_end, num_points,1)


p = plot(
    scatter(x=t_sim, y=X_i, mode="lines",name = "X", line=attr(width=3, color="black"), showlegend = true),
    Layout(
        xaxis=attr(title="Time", range=[-0.5, 16], tickvals=0:2:16, ticktext=string.(0:2:16), showline=true, linecolor="black"),
        yaxis=attr( range=[0, 2.5], tickvals=[0,1,2], ticktext=["0", "1", "2"], showline=true, linecolor="black"),
        font=attr(family="Arial", size=25),
        width=600, height=500, plot_bgcolor="white", showgrid=false,
        legend=attr(x=0.5, y=1.1, xanchor="center", font=attr(size=16), orientation="h")
    )
)
display(p)



#=
==============================
Standard deviation assumptions

- Use a common SD for all parameters (σ is identical across params).
- Draw samples from Normal(μ, σ) on the linear scale; no log transform is applied.

From 95% CIs:
SD_parameter = (upper bound - lower bound) * C, (See paper for details)
==============================
=#


SD_a = 0.1
SD_b = 0.1

#= 
==============================
Total number of iterations: N
Number of repeats : M
==============================
=#
N = 5000
M = 1  # Use M = 1: the model is simple and converges quickly; repeated runs are unnecessary.



# Initialize storage for parameter values
a = zeros(M, N)
b = zeros(M, N)

a_acc = zeros(M, N)
b_acc = zeros(M, N)


# Initialize Normal(μ, σ) with μ = best_params for the first iteration
for i in 1:M
    a_acc[i, 1] = 1
    b_acc[i, 1] = 1
end



# Initialize storage for accepted trajectories
X = zeros(M, N)
Y = zeros(M, N)
X_Acc = [[] for _ in 1:M]
X_Rej = [[] for _ in 1:M]



#= 
=================================================================================================================================
                                                           TM-ABC-MCMC
=================================================================================================================================
=#
using Dates 
start_time = now()
for m in 1:M
    println("Starting iteration $m")
    for i in 2:N

        a[m, i] = rand(Normal(a_acc[m, i-1], abs(SD_a)))
        b[m, i] = rand(Normal(b_acc[m, i-1], abs(SD_b)))

        theta = [a[m, i], b[m, i]]
        t, X, Y  = solve_ode(theta, t_start, t_end, num_points, i)
        s = []

        for j in 1:length(X[1:t_step:end])
            if X_lower_bound[j] <= X[1:t_step:end][j] <= X_upper_bound[j]
                push!(s, j)
            end
        end
        
        if length(s) >= 8 &&  0 <= a[m, i] && 0 <= b[m, i] 
            a_acc[m, i] = a[m, i]
            b_acc[m, i] = b[m, i]
            push!(X_Acc[m], X)
        else
            a_acc[m, i] = a_acc[m, i-1]
            b_acc[m, i] = b_acc[m, i-1]
            push!(X_Rej[m], X)
        end
    end
end
end_time = now()
elapsed = end_time - start_time
println("Execution time: ", elapsed)



for i in 1:M
    println("Length of X_Acc: ", length(X_Acc[i]))
end




# Plot accepted trajectories and data min–max at observation times
for i in 1:M
    all_trajectories = !isempty(X_Acc[i]) ? 
        [scatter(x=(t_end/num_points) .* (1:length(vec)), y=vec, mode="lines",
                 line=attr(color="#e4eef1"), showlegend=false) for vec in X_Acc[i]] : []

    selected_plots = [
        scatter(x=(t_end/num_points) .* (1:length(X_Acc[i][5])), y=X_Acc[i][5], mode="lines",
                line=attr(color="#868CA2", width=3), showlegend=false),
        scatter(x=(t_end/num_points) .* (1:length(X_Acc[i][10])), y=X_Acc[i][10], mode="lines",
                line=attr(color="#8b5058", width=3), showlegend=false),
        scatter(x=(t_end/num_points) .* (1:length(X_Acc[i][11])), y=X_Acc[i][11], mode="lines",
                line=attr(color="#ec9f79", width=3), showlegend=false)
    ]

    max_min_scatter = [
        scatter(x=t_sampled, y=X_upper_bound, mode="markers", marker=attr(color="#5F747A", symbol="circle", size=10),showlegend=false),
        scatter(x=t_sampled, y=X_lower_bound, mode="markers", showlegend = false, marker=attr(color="#5F747A", symbol="circle", size=10))
    ]

    min_max_lines = [
        scatter(x=[t, t],
                y=[X_lower_bound[i], X_upper_bound[i]],
                mode="lines",
                line=attr(color="#5F747A", dash="dash", width=1),
                showlegend=false, hoverinfo="skip")
        for (i, t) in enumerate(t_sampled)
    ]

    Dummy_1 = scatter(x=[NaN], y=[NaN], mode="markers",marker=attr(color="#5F747A", size = 10),name="Data")
    Dummy_2 = scatter(x=[NaN], y=[NaN], mode="markers",marker=attr(color="#e4eef1", symbol="square", size=14),name="Acc traj",showlegend=true)
    Dummy_3 = scatter(x=[NaN], y=[NaN], mode="lines",line=attr(color="black"),name="Rep traj",showlegend=true)

    ann = attr(
        text = "M = $(i)",
        x = 0.99, y = 1.14, xref = "paper", yref = "paper",
        xanchor = "right", yanchor = "bottom",
        showarrow = false,
        font = attr(family="Arial", size=18),
        bgcolor = "rgba(255,255,255,0.85)",  
        bordercolor = "black", 
        borderwidth = 2,
        borderpad = 3 
    )
    layout = Layout(
        xaxis=attr(title="Time", range=[-0.5, 16], tickvals=0:2:16, ticktext=string.(0:2:16), showline=true, linecolor="black"),
        yaxis=attr(range=[0, 2.5], tickvals=[0,1,2], ticktext=["0", "1", "2"], showline=true, linecolor="black"),
        font=attr(family="Arial", size=25),
        width=600, height=500, plot_bgcolor="white", showgrid=false
        ,legend=attr(x=0.5, y=1.1, xanchor="center", font=attr(size=16), orientation="h")
    )

    p = plot(vcat(Dummy_1, Dummy_2, Dummy_3,all_trajectories, max_min_scatter, min_max_lines, selected_plots), layout)
    relayout!(p, annotations=[ann], margin=attr(t=90))
    display(p)
    savefig(p, "X_Acc_$i.png")
    println("Plotted iteration $i with $(length(X_Acc[i])) trajectories.")
end
CSV.write("X_Acc.csv", vcat([DataFrame(Iteration=i, Trajectory=j, Time=collect(1:length(vec)), Value=vec) 
          for i in 1:M for (j, vec) in enumerate(X_Acc[i])]...))



# Define parameters and their corresponding labels
param_names = ["a", "b"]
param_labels = ["a", "b"]


# Storage for all unique values
all_unique_lists = Dict(param => Vector{Vector{Float64}}(undef, M) for param in param_names)


# Compute unique values for each parameter and save as CSV
for i in 1:M
    unique_lists = [
        unique(a_acc[i, :]), unique(b_acc[i, :])]
    
    for (param, unique_vals) in zip(param_names, unique_lists)
        all_unique_lists[param][i] = unique(vcat(unique_vals...))
        df = DataFrame(Iteration=i, Value=all_unique_lists[param][i])
        CSV.write("all_unique_$(param)_iteration_$(i).csv", df)
    end
end




# Convergence diagnostics
ann_M(i) = attr(
    text = "M = $(i)",
    x = 0.98, y = 1.0, xref = "paper", yref = "paper",
    xanchor = "right", yanchor = "top",
    showarrow = false,
    font = attr(family="Arial", size=16),
    bgcolor = "rgba(255,255,255,0.85)",
    bordercolor = "#5F747A", borderwidth = 2, borderpad = 3
)

for (param, label) in zip(param_names, param_labels)
    for i in 1:M
        all_unique_param = all_unique_lists[param][i]
        log_values = all_unique_param
        num = length(log_values)

        common_layout = Layout(
            xaxis=attr(showline=true, linecolor="black"),
            yaxis=attr(showline=true, linecolor="black"),
            width=600, height=500, font=attr(family="Arial", size=25),
            showlegend=false, showgrid=false, plot_bgcolor="white",
            margin=attr(l=10, r=10, t=10, b=10)
        )

        x_min, x_max = minimum(log_values), maximum(log_values)
        range_low, range_high = abs(0.1*x_min), abs(0.1*x_max)
        extended_range = (x_min - range_low, x_max + range_high)

        # Histogram
        p_hist = plot(
            histogram(x=log_values, nbinsx=10, marker_color="#525b7a"),
            merge(common_layout, Layout(
                xaxis_title=label, yaxis_title="Frequency",
                xaxis=attr(range=[extended_range[1], extended_range[2]]),
                annotations=[ann_M(i)]
            ))
        )
        display(p_hist); savefig(p_hist, "$(param)_histogram_iteration_$i.png")

        # Trace
        p_trace = plot(
            scatter(x=1:num, y=log_values, mode="lines", line=attr(color="black")),
            merge(common_layout, Layout(yaxis_title=label, annotations=[ann_M(i)],xaxis_title="Iteration"))
        )
        display(p_trace); savefig(p_trace, "$(param)_trace_iteration_$i.png")

        # ACF
        acf_values = autocor(log_values, 0:num-1)
        p_acf = plot(
            [
                scatter(x=1:num, y=acf_values, mode="lines", line=attr(color="black")),
                scatter(x=[1, num], y=[0, 0], mode="lines", line=attr(color="black", dash="dash"))
            ],
            merge(common_layout, Layout(yaxis_title=label, annotations=[ann_M(i)],xaxis_title="Lag"))
        )
        display(p_acf); savefig(p_acf, "ACF_$(param)_iteration_$i.png")
    end
end
