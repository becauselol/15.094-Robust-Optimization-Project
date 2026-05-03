#!/usr/bin/env julia

using TOML
using CSV
using DataFrames
using JSON
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "StationSelection.jl", "src"))
using StationSelection

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

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
        nominal_runs = sort(
            get(nominal_by_key, key, Dict{String, Any}[]);
            by = r -> (get(r, "job_id", -1), string(get(r, "timestamp", ""))),
        )
        robust_runs = sort(
            robust_by_key[key];
            by = r -> (get(r, "comparison_quantile", -1.0), get(r, "job_id", -1), string(get(r, "timestamp", ""))),
        )
        isempty(nominal_runs) && continue
        for nominal in nominal_runs, robust in robust_runs
            push!(pairs, (nominal, robust))
        end
    end
    return pairs
end

function load_active_schedule(run_dir::String)
    schedule_file = joinpath(run_dir, "backtest", "active_station_schedule.csv")
    df = CSV.read(schedule_file, DataFrame)
    return [(String(row.scenario_label), sort(parse.(Int, split(String(row.active_station_ids))))) for row in eachrow(df)]
end

function build_walking_inputs(run_dir::String)
    cfg = TOML.parsefile(joinpath(run_dir, "config.toml"))
    data_cfg = cfg["data"]
    params = cfg["parameters"]

    station_file = joinpath(PROJECT_ROOT, data_cfg["station_file"])
    stations = read_candidate_stations(station_file)
    walking_costs = compute_station_pairwise_costs(stations)
    station_ids = sort(Int.(stations.id))
    max_walking_distance = Float64(get(params, "max_walking_distance", Inf))

    return station_ids, walking_costs, max_walking_distance
end

function count_uncovered_stations(
    station_ids::Vector{Int},
    active_ids::Vector{Int},
    walking_costs::Dict{Tuple{Int, Int}, Float64},
    max_walking_distance::Float64,
)
    uncovered_station_ids = Int[]
    for station_id in station_ids
        has_access = any(get(walking_costs, (station_id, active_id), Inf) <= max_walking_distance for active_id in active_ids)
        has_access || push!(uncovered_station_ids, station_id)
    end
    return uncovered_station_ids
end

function compute_pair_results(nominal::Dict{String, Any}, robust::Dict{String, Any})
    station_ids, walking_costs, max_walking_distance = build_walking_inputs(String(nominal["run_dir"]))

    scenario_rows = NamedTuple[]
    aggregate_rows = NamedTuple[]

    for (model_label, record) in [("Nominal", nominal), ("Robust", robust)]
        uncovered_counts = Float64[]
        uncovered_rates = Float64[]
        for (scenario_label, active_ids) in load_active_schedule(String(record["run_dir"]))
            uncovered_station_ids = count_uncovered_stations(
                station_ids,
                active_ids,
                walking_costs,
                max_walking_distance,
            )
            uncovered_count = length(uncovered_station_ids)
            total_stations = length(station_ids)
            uncovered_rate = total_stations == 0 ? missing : uncovered_count / total_stations
            covered_count = total_stations - uncovered_count
            push!(scenario_rows, (
                k = nominal["k"],
                in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
                quantile = robust["comparison_quantile"],
                nominal_job_id = nominal["job_id"],
                robust_job_id = robust["job_id"],
                model_label = model_label,
                scenario_label = scenario_label,
                total_stations = total_stations,
                covered_stations = covered_count,
                uncovered_stations = uncovered_count,
                uncovered_rate = uncovered_rate,
            ))
            push!(uncovered_counts, uncovered_count)
            ismissing(uncovered_rate) || push!(uncovered_rates, uncovered_rate)
        end
        push!(aggregate_rows, (
            k = nominal["k"],
            in_vehicle_time_weight = nominal["in_vehicle_time_weight"],
            quantile = robust["comparison_quantile"],
            nominal_job_id = nominal["job_id"],
            robust_job_id = robust["job_id"],
            model_label = model_label,
            total_uncovered_stations_across_scenarios = isempty(uncovered_counts) ? missing : sum(uncovered_counts),
            avg_uncovered_stations = isempty(uncovered_counts) ? missing : mean(uncovered_counts),
            max_uncovered_stations = isempty(uncovered_counts) ? missing : maximum(uncovered_counts),
            avg_uncovered_rate = isempty(uncovered_rates) ? missing : mean(uncovered_rates),
        ))
    end

    return DataFrame(scenario_rows), DataFrame(aggregate_rows)
end

function default_output_prefix(exp_dir::String)
    return basename(normpath(exp_dir))
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=StationSelection.jl analyze_station_walk_coverage.jl <experiment_dir> [output_dir]")
    exp_dir = normpath(ARGS[1])
    output_dir = length(ARGS) >= 2 ? normpath(ARGS[2]) : exp_dir
    mkpath(output_dir)

    scenario_df = DataFrame()
    aggregate_df = DataFrame()
    for (nominal, robust) in pair_runs(exp_dir)
        pair_scenario_df, pair_aggregate_df = compute_pair_results(nominal, robust)
        append!(scenario_df, pair_scenario_df; cols = :union)
        append!(aggregate_df, pair_aggregate_df; cols = :union)
    end

    prefix = default_output_prefix(exp_dir)
    scenario_path = joinpath(output_dir, "station_walk_coverage_scenarios_" * prefix * ".csv")
    aggregate_path = joinpath(output_dir, "station_walk_coverage_summary_" * prefix * ".csv")
    CSV.write(scenario_path, scenario_df)
    CSV.write(aggregate_path, aggregate_df)

    println("Scenario CSV: $scenario_path")
    println("Aggregate CSV: $aggregate_path")
end

main()
