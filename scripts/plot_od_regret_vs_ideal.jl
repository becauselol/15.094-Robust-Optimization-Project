#!/usr/bin/env julia

using TOML
using CSV
using DataFrames
using JSON
using Plots
using Statistics

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
    cfg = TOML.parsefile(joinpath(run_dir, "config.toml"))
    data_cfg = cfg["data"]
    params = cfg["parameters"]

    station_file = joinpath(PROJECT_ROOT, data_cfg["station_file"])
    segment_file = joinpath(PROJECT_ROOT, data_cfg["segment_file"])
    stations = read_candidate_stations(station_file)
    walking_costs = compute_station_pairwise_costs(stations)
    routing_costs = read_routing_costs_from_segments(segment_file, stations)
    station_ids = sort(Int.(stations.id))
    max_walking_distance = Float64(get(params, "max_walking_distance", Inf))

    return station_ids, walking_costs, routing_costs, max_walking_distance
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

function compute_pair_results(nominal::Dict{String, Any}, robust::Dict{String, Any})
    lambda_val = Float64(nominal["in_vehicle_time_weight"])
    base_run_dir = String(nominal["run_dir"])
    station_ids, walking_costs, routing_costs, max_walking_distance = build_cost_inputs(base_run_dir)

    ideal_costs = Dict{Tuple{Int, Int}, Float64}()
    for o in station_ids, d in station_ids
        o == d && continue
        best = best_feasible_pair_cost(o, d, station_ids, walking_costs, routing_costs, lambda_val, max_walking_distance)
        !isnothing(best) && (ideal_costs[(o, d)] = best)
    end

    detail_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    scatter_data = Dict("Nominal" => Dict{String, Vector{Tuple{Float64, Float64}}}(),
                        "Robust" => Dict{String, Vector{Tuple{Float64, Float64}}}())

    for (model_label, record) in [("Nominal", nominal), ("Robust", robust)]
        for (scenario_label, active_ids) in load_active_schedule(String(record["run_dir"]))
            regrets = Float64[]
            points = Tuple{Float64, Float64}[]
            total_ods = 0
            feasible_ods = 0
            for o in station_ids, d in station_ids
                o == d && continue
                total_ods += 1
                ideal_cost = get(ideal_costs, (o, d), nothing)
                isnothing(ideal_cost) && continue
                actual_cost = best_feasible_pair_cost(o, d, active_ids, walking_costs, routing_costs, lambda_val, max_walking_distance)
                regret = isnothing(actual_cost) ? missing : actual_cost - ideal_cost
                if !isnothing(actual_cost)
                    feasible_ods += 1
                    push!(regrets, regret)
                    push!(points, (ideal_cost, actual_cost))
                end
                push!(detail_rows, (
                    k = nominal["k"],
                    in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
                    quantile = robust["comparison_quantile"],
                    nominal_job_id = nominal["job_id"],
                    robust_job_id = robust["job_id"],
                    model_label = model_label,
                    scenario_label = scenario_label,
                    origin_id = o,
                    destination_id = d,
                    ideal_cost = ideal_cost,
                    actual_cost = isnothing(actual_cost) ? missing : actual_cost,
                    regret = regret,
                    is_feasible_under_active_set = !isnothing(actual_cost),
                ))
            end
            scatter_data[model_label][scenario_label] = points
            push!(summary_rows, (
                k = nominal["k"],
                in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
                quantile = robust["comparison_quantile"],
                nominal_job_id = nominal["job_id"],
                robust_job_id = robust["job_id"],
                model_label = model_label,
                scenario_label = scenario_label,
                total_ods = total_ods,
                feasible_ods = feasible_ods,
                feasible_rate = total_ods == 0 ? missing : feasible_ods / total_ods,
                mean_regret = isempty(regrets) ? missing : mean(regrets),
                median_regret = isempty(regrets) ? missing : quantile(regrets, 0.5),
                p90_regret = isempty(regrets) ? missing : quantile(regrets, 0.9),
                p95_regret = isempty(regrets) ? missing : quantile(regrets, 0.95),
                max_regret = isempty(regrets) ? missing : maximum(regrets),
            ))
        end
    end

    return DataFrame(detail_rows), DataFrame(summary_rows), scatter_data
end

function plot_pair(nominal::Dict{String, Any}, robust::Dict{String, Any}, out_dir::String)
    detail_df, summary_df, scatter_data = compute_pair_results(nominal, robust)
    scenario_labels = sort!(collect(union(keys(scatter_data["Nominal"]), keys(scatter_data["Robust"]))))

    all_values = Float64[]
    for model_map in values(scatter_data), points in values(model_map), (x, y) in points
        push!(all_values, x)
        push!(all_values, y)
    end
    max_axis = isempty(all_values) ? 1.0 : maximum(all_values)

    plt = plot(layout=(2, 4), size=(1600, 800), legend=false)
    colors = Dict("Nominal" => :blue, "Robust" => :orange)

    for (col, scenario_label) in enumerate(scenario_labels)
        for (row_idx, model_label) in enumerate(["Nominal", "Robust"])
            points = get(scatter_data[model_label], scenario_label, Tuple{Float64, Float64}[])
            idx = (row_idx - 1) * 4 + col
            if !isempty(points)
                xs = first.(points)
                ys = last.(points)
                scatter!(plt[idx], xs, ys; ms=2.5, alpha=0.35, color=colors[model_label])
                regrets = ys .- xs
                annotate!(plt[idx], 0.03 * max_axis, 0.97 * max_axis,
                    text("n=$(length(points))\nmean=$(round(mean(regrets), digits=1))\np95=$(round(quantile(regrets, 0.95), digits=1))", 8, :left))
            end
            plot!(plt[idx], [0.0, max_axis], [0.0, max_axis]; color=:black, linestyle=:dash, linewidth=1)
            plot!(plt[idx]; xlims=(0, max_axis * 1.02), ylims=(0, max_axis * 1.02),
                  xlabel="Ideal OD cost", ylabel="Actual OD cost",
                  title="$(model_label) $(scenario_label)", gridalpha=0.2)
        end
    end

    plot!(plt; plot_title="OD ideal vs actual cost | k=$(nominal["k"]) | lambda=$(nominal["in_vehicle_time_weight"]) | nominal job $(nominal["job_id"]) vs robust job $(robust["job_id"]) | q=$(robust["comparison_quantile"])")

    out_path = joinpath(out_dir, figure_name(nominal, robust))
    savefig(plt, out_path)

    manifest_row = (
        k = nominal["k"],
        in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
        quantile = robust["comparison_quantile"],
        nominal_job_id = nominal["job_id"],
        robust_job_id = robust["job_id"],
        figure_path = out_path,
    )
    return manifest_row, detail_df, summary_df
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl scripts/plot_od_regret_vs_ideal.jl <experiment_dir>")
    exp_dir = normpath(ARGS[1])
    out_dir = joinpath(exp_dir, "figures", "od_regret_scatter")
    mkpath(out_dir)

    manifest_rows = NamedTuple[]
    detail_df = DataFrame()
    summary_df = DataFrame()

    for (nominal, robust) in pair_runs(exp_dir)
        manifest_row, pair_detail_df, pair_summary_df = plot_pair(nominal, robust, out_dir)
        push!(manifest_rows, manifest_row)
        append!(detail_df, pair_detail_df; cols=:union)
        append!(summary_df, pair_summary_df; cols=:union)
    end

    manifest_path = joinpath(out_dir, "manifest.csv")
    CSV.write(manifest_path, DataFrame(manifest_rows))
    detail_path = joinpath(exp_dir, "od_regret_detail.csv")
    summary_path = joinpath(exp_dir, "od_regret_summary.csv")
    CSV.write(detail_path, detail_df)
    CSV.write(summary_path, summary_df)

    println("Wrote $(length(manifest_rows)) OD regret scatter figures to $out_dir")
    println("Manifest: $manifest_path")
    println("Summary CSV: $summary_path")
    println("Detail CSV: $detail_path")
end

main()
