#!/usr/bin/env julia

using CSV, DataFrames, Plots, TOML
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

function load_station_points(run_dir::String)
    cfg = TOML.parsefile(joinpath(run_dir, "config.toml"))
    station_file = joinpath(PROJECT_ROOT, cfg["data"]["station_file"])
    stations = AnalysisHelpers.read_candidate_stations(station_file)
    sort!(stations, :id)
    return stations
end

function load_period_activations(run_dir::String)
    df = CSV.read(joinpath(run_dir, "variable_exports", "scenario_activation.csv"), DataFrame)
    by_period = Dict{String, Set{Int}}()
    for row in eachrow(df)
        Float64(row.value) >= 0.5 || continue
        label = String(row.scenario_label)
        push!(get!(by_period, label, Set{Int}()), Int(row.station_id))
    end
    return by_period
end

function plot_pair(nominal, robust, out_dir::String)
    stations = load_station_points(String(nominal["run_dir"]))
    nom_by_period = load_period_activations(String(nominal["run_dir"]))
    rob_by_period = load_period_activations(String(robust["run_dir"]))
    periods = ("period_1", "period_2", "period_3", "period_4")

    lons = Float64.(stations.lon)
    lats = Float64.(stations.lat)
    xpad = (maximum(lons) - minimum(lons)) * 0.05
    ypad = (maximum(lats) - minimum(lats)) * 0.05

    plt = plot(layout=(2, 4), size=(1600, 800), legend=false)
    for (col, period) in enumerate(periods)
        for (row_idx, (label, active_ids, color)) in enumerate((
            ("Nominal", get(nom_by_period, period, Set{Int}()), :blue),
            ("Robust", get(rob_by_period, period, Set{Int}()), :darkorange),
        ))
            idx = (row_idx - 1) * 4 + col
            scatter!(plt[idx], lons, lats;
                color=:lightgray, alpha=0.9, markersize=3.5,
                markerstrokewidth=0, label="")
            active_rows = [i for i in 1:nrow(stations) if Int(stations.id[i]) in active_ids]
            if !isempty(active_rows)
                scatter!(plt[idx], lons[active_rows], lats[active_rows];
                    color=color, markersize=5.5, markerstrokecolor=:black,
                    markerstrokewidth=0.5, label="")
            end
            plot!(plt[idx];
                title="$(label) $(period)",
                xlims=(minimum(lons) - xpad, maximum(lons) + xpad),
                ylims=(minimum(lats) - ypad, maximum(lats) + ypad),
                xticks=false, yticks=false, aspect_ratio=:equal, grid=false)
        end
    end

    plot!(plt; plot_title="z_js activations | k=$(nominal["k"]) | lambda=$(nominal["in_vehicle_time_weight"]) | nominal job $(nominal["job_id"]) vs robust job $(robust["job_id"]) | q=$(robust["comparison_quantile"])")
    out_path = joinpath(out_dir, safe_slug("k$(nominal["k"])_lambda$(nominal["in_vehicle_time_weight"])_nom$(nominal["job_id"])_rob$(robust["job_id"])_q$(robust["comparison_quantile"])") * ".png")
    savefig(plt, out_path)
    return (
        k = nominal["k"],
        in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
        quantile = robust["comparison_quantile"],
        nominal_job_id = nominal["job_id"],
        robust_job_id = robust["job_id"],
        figure_path = out_path,
    )
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/plot_activation_comparisons.jl <experiment_dir>")
    exp_dir = normpath(ARGS[1])
    out_dir = joinpath(exp_dir, "figures", "activation_comparisons")
    mkpath(out_dir)
    rows = [plot_pair(n, r, out_dir) for (n, r) in pair_runs(exp_dir)]
    CSV.write(joinpath(out_dir, "manifest.csv"), DataFrame(rows))
    println("Wrote $(length(rows)) activation comparison figures to $out_dir")
end

main()
