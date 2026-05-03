#!/usr/bin/env julia

using CSV
using DataFrames
using JSON
using Plots
using TOML

push!(LOAD_PATH, joinpath(@__DIR__, "..", "StationSelection.jl", "src"))
using StationSelection

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

function load_metrics(metrics_path::String)
    return JSON.parsefile(metrics_path)
end

function iter_metric_files(exp_dir::String)
    runs_dir = joinpath(exp_dir, "runs")
    isdir(runs_dir) || return String[]
    files = String[]
    for run_name in sort(readdir(runs_dir))
        metrics_path = joinpath(runs_dir, run_name, "metrics.json")
        isfile(metrics_path) && push!(files, metrics_path)
    end
    return files
end

function collect_runs(exp_dir::String)
    nominal_by_key = Dict{Tuple{Int, Float64}, Vector{Dict{String, Any}}}()
    robust_by_key = Dict{Tuple{Int, Float64}, Vector{Dict{String, Any}}}()
    for metrics_path in iter_metric_files(exp_dir)
        metrics = load_metrics(metrics_path)
        model_type = get(metrics, "model_type", nothing)
        k = get(metrics, "k", nothing)
        lam = get(metrics, "in_vehicle_time_weight", nothing)
        (isnothing(model_type) || isnothing(k) || isnothing(lam)) && continue
        record = Dict{String, Any}(
            "job_id" => get(metrics, "job_id", nothing),
            "run_dir" => dirname(metrics_path),
            "timestamp" => get(metrics, "timestamp", nothing),
            "k" => Int(k),
            "in_vehicle_time_weight" => Float64(lam),
            "demand_quantile" => get(metrics, "demand_quantile", nothing),
            "Q_cap_quantile" => get(metrics, "Q_cap_quantile", nothing),
            "q_high_quantile" => get(metrics, "q_high_quantile", nothing),
        )
        key = (Int(k), Float64(lam))
        if model_type == "NominalModel"
            push!(get!(nominal_by_key, key, Dict{String, Any}[]), record)
        elseif model_type == "RobustTotalDemandCapModel"
            quantile = get(record, "demand_quantile", nothing)
            if isnothing(quantile)
                q_cap = get(record, "Q_cap_quantile", nothing)
                q_high = get(record, "q_high_quantile", nothing)
                (isnothing(q_cap) || isnothing(q_high) || q_cap != q_high) && continue
                quantile = q_cap
            end
            record["comparison_quantile"] = Float64(quantile)
            push!(get!(robust_by_key, key, Dict{String, Any}[]), record)
        end
    end
    return nominal_by_key, robust_by_key
end

function pair_runs(exp_dir::String)
    nominal_by_key, robust_by_key = collect_runs(exp_dir)
    pairs = Tuple{Dict{String, Any}, Dict{String, Any}}[]
    for key in sort(collect(keys(robust_by_key)))
        nominal_runs = sort(get(nominal_by_key, key, Dict{String, Any}[]); by=r -> (get(r, "job_id", -1), string(get(r, "timestamp", ""))))
        robust_runs = sort(robust_by_key[key]; by=r -> (get(r, "comparison_quantile", -1.0), get(r, "job_id", -1), string(get(r, "timestamp", ""))))
        isempty(nominal_runs) && continue
        for nominal in nominal_runs, robust in robust_runs
            push!(pairs, (nominal, robust))
        end
    end
    return pairs
end

function safe_slug(value::AbstractString)
    io = IOBuffer()
    for ch in value
        if isletter(ch) || isnumeric(ch) || ch == '-' || ch == '_'
            print(io, ch)
        else
            print(io, '_')
        end
    end
    return String(take!(io))
end

function figure_name(nominal::Dict{String, Any}, robust::Dict{String, Any})
    return safe_slug("k$(nominal["k"])_lambda$(nominal["in_vehicle_time_weight"])_nom$(nominal["job_id"])_rob$(robust["job_id"])_q$(robust["comparison_quantile"])") * ".png"
end

function load_active_schedule(run_dir::String)
    schedule_file = joinpath(run_dir, "backtest", "active_station_schedule.csv")
    df = CSV.read(schedule_file, DataFrame)
    return [(String(row.scenario_label), sort(parse.(Int, split(String(row.active_station_ids))))) for row in eachrow(df)]
end

function build_cost_inputs(run_dir::String)
    cfg = JSON.parsefile(joinpath(run_dir, "metrics.json"))
    lambda_val = Float64(get(cfg, "in_vehicle_time_weight", 0.0))

    # use the original data inputs to match optimization geometry exactly
    toml_cfg = TOML.parsefile(joinpath(run_dir, "config.toml"))
    data_cfg = toml_cfg["data"]
    params = toml_cfg["parameters"]
    station_file = joinpath(PROJECT_ROOT, data_cfg["station_file"])
    segment_file = joinpath(PROJECT_ROOT, data_cfg["segment_file"])
    stations = read_candidate_stations(station_file)
    walking_costs = compute_station_pairwise_costs(stations)
    routing_costs = read_routing_costs_from_segments(segment_file, stations)
    station_ids = sort(Int.(stations.id))
    max_walking_distance = Float64(get(params, "max_walking_distance", Inf))
    return station_ids, walking_costs, routing_costs, max_walking_distance, lambda_val
end

function best_feasible_pair_cost(
    origin_id::Int,
    dest_id::Int,
    active_ids::Vector{Int},
    walking_costs::Dict{Tuple{Int, Int}, Float64},
    routing_costs::Dict{Tuple{Int, Int}, Float64},
    lambda_val::Float64,
    max_walking_distance::Float64,
)
    best_cost = Inf
    for pickup_id in active_ids
        get(walking_costs, (origin_id, pickup_id), Inf) <= max_walking_distance || continue
        for dropoff_id in active_ids
            get(walking_costs, (dropoff_id, dest_id), Inf) <= max_walking_distance || continue
            route_cost = pickup_id == dropoff_id ? 0.0 : get(routing_costs, (pickup_id, dropoff_id), Inf)
            isfinite(route_cost) || continue
            total_cost = get(walking_costs, (origin_id, pickup_id), Inf) +
                         get(walking_costs, (dropoff_id, dest_id), Inf) +
                         lambda_val * route_cost
            best_cost = min(best_cost, total_cost)
        end
    end
    return isfinite(best_cost) ? best_cost : nothing
end

function theoretical_costs_by_scenario(run_dir::String)
    station_ids, walking_costs, routing_costs, max_walking_distance, lambda_val = build_cost_inputs(run_dir)
    result = Dict{String, Vector{Float64}}()
    for (label, active_ids) in load_active_schedule(run_dir)
        costs = Float64[]
        for origin_id in station_ids, dest_id in station_ids
            origin_id == dest_id && continue
            best = best_feasible_pair_cost(origin_id, dest_id, active_ids, walking_costs, routing_costs, lambda_val, max_walking_distance)
            !isnothing(best) && push!(costs, best)
        end
        result[label] = sort(costs)
    end
    return result
end

function ecdf(values::Vector{Float64})
    n = length(values)
    n == 0 && return Float64[], Float64[]
    return values, collect(1:n) ./ n
end

function plot_pair(nominal::Dict{String, Any}, robust::Dict{String, Any}, out_dir::String)
    nom = theoretical_costs_by_scenario(String(nominal["run_dir"]))
    rob = theoretical_costs_by_scenario(String(robust["run_dir"]))
    scenario_labels = sort!(collect(union(keys(nom), keys(rob))))

    plt = plot(layout=(2, 2), size=(1200, 800))
    for (idx, label) in enumerate(scenario_labels)
        nom_costs = get(nom, label, Float64[])
        rob_costs = get(rob, label, Float64[])
        if !isempty(nom_costs)
            x_nom, y_nom = ecdf(nom_costs)
            plot!(plt[idx], x_nom, y_nom; color=:blue, linewidth=2, label="Nominal (n=$(length(nom_costs)))")
        end
        if !isempty(rob_costs)
            x_rob, y_rob = ecdf(rob_costs)
            plot!(plt[idx], x_rob, y_rob; color=:orange, linewidth=2, label="Robust (n=$(length(rob_costs)))")
        end
        plot!(plt[idx]; title=label, xlabel="Theoretical OD weighted cost", ylabel="Empirical CDF", gridalpha=0.2)
    end

    plot!(plt; plot_title="Theoretical OD CDFs by scenario | k=$(nominal["k"]) | lambda=$(nominal["in_vehicle_time_weight"]) | nominal job $(nominal["job_id"]) vs robust job $(robust["job_id"]) | q=$(robust["comparison_quantile"])")
    out_path = joinpath(out_dir, figure_name(nominal, robust))
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
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/plot_theoretical_od_cdfs_by_scenario.jl <experiment_dir>")
    exp_dir = normpath(ARGS[1])
    out_dir = joinpath(exp_dir, "figures", "theoretical_od_cdfs_by_scenario")
    mkpath(out_dir)

    rows = NamedTuple[]
    for (nominal, robust) in pair_runs(exp_dir)
        push!(rows, plot_pair(nominal, robust, out_dir))
    end

    manifest_path = joinpath(out_dir, "manifest.csv")
    CSV.write(manifest_path, DataFrame(rows))
    println("Wrote $(length(rows)) theoretical OD CDF figures to $out_dir")
    println("Manifest: $manifest_path")
end

main()
