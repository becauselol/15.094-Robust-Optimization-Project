#!/usr/bin/env julia

using TOML
using CSV
using DataFrames
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "..", "StationSelection.jl", "src"))
using StationSelection

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

function load_active_schedule(run_dir::String)
    schedule_file = joinpath(run_dir, "backtest", "active_station_schedule.csv")
    isfile(schedule_file) || error("Missing active station schedule: $schedule_file")
    df = CSV.read(schedule_file, DataFrame)
    return Dict(
        Int(row.scenario_idx) => sort(parse.(Int, split(String(row.active_station_ids))))
        for row in eachrow(df)
    )
end

function load_assignment_exports(run_dir::String)
    assign_file = joinpath(run_dir, "variable_exports", "assignment_variables.csv")
    isfile(assign_file) || error("Missing assignment exports: $assign_file")
    return CSV.read(assign_file, DataFrame)
end

function feasible_under_active_set(
    origin_id::Int,
    dest_id::Int,
    active_ids::Vector{Int},
    walking_costs::Dict{Tuple{Int, Int}, Float64},
    routing_costs::Dict{Tuple{Int, Int}, Float64},
    max_walking_distance::Float64,
)::Bool
    for pickup_id in active_ids
        get(walking_costs, (origin_id, pickup_id), Inf) <= max_walking_distance || continue
        for dropoff_id in active_ids
            get(walking_costs, (dropoff_id, dest_id), Inf) <= max_walking_distance || continue
            route_cost = pickup_id == dropoff_id ? 0.0 : get(routing_costs, (pickup_id, dropoff_id), Inf)
            isfinite(route_cost) || continue
            return true
        end
    end
    return false
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

function analyze_run(run_dir::String)
    metrics = JSON.parsefile(joinpath(run_dir, "metrics.json"))
    assignment_df = load_assignment_exports(run_dir)
    schedule = load_active_schedule(run_dir)
    station_ids, walking_costs, routing_costs, max_walking_distance = build_cost_inputs(run_dir)

    all_ods = [(o, d) for o in station_ids for d in station_ids if o != d]
    total_ods = length(all_ods)

    rows = NamedTuple[]
    for scenario_idx in sort(collect(keys(schedule)))
        active_ids = schedule[scenario_idx]
        assign_s = assignment_df[Int.(assignment_df.scenario) .== scenario_idx, :]

        exported_ods = Set((Int(row.origin_id), Int(row.dest_id)) for row in eachrow(assign_s))
        violating_exports = 0
        nonactive_exports = 0
        for row in eachrow(assign_s)
            pickup_id = Int(row.pickup_id)
            dropoff_id = Int(row.dropoff_id)
            origin_id = Int(row.origin_id)
            dest_id = Int(row.dest_id)
            ((pickup_id in active_ids) && (dropoff_id in active_ids)) || (nonactive_exports += 1)
            wj = get(walking_costs, (origin_id, pickup_id), Inf)
            wk = get(walking_costs, (dropoff_id, dest_id), Inf)
            if !(wj <= max_walking_distance && wk <= max_walking_distance)
                violating_exports += 1
            end
        end

        feasible_ods = 0
        for (o, d) in all_ods
            feasible_under_active_set(o, d, active_ids, walking_costs, routing_costs, max_walking_distance) && (feasible_ods += 1)
        end

        push!(rows, (
            run_dir = run_dir,
            job_id = get(metrics, "job_id", missing),
            model_type = get(metrics, "model_type", missing),
            k = get(metrics, "k", missing),
            in_vehicle_time_weight = get(metrics, "in_vehicle_time_weight", missing),
            demand_quantile = string(get(metrics, "demand_quantile", "")),
            scenario_idx = scenario_idx,
            active_station_count = length(active_ids),
            exported_assignment_rows = nrow(assign_s),
            exported_od_count = length(exported_ods),
            total_theoretical_ods = total_ods,
            julia_feasible_od_count = feasible_ods,
            julia_feasible_od_rate = feasible_ods / total_ods,
            export_nonactive_count = nonactive_exports,
            export_walking_violation_count = violating_exports,
        ))
    end

    return DataFrame(rows)
end

function main()
    if length(ARGS) < 1
        error("Usage: julia --project=StationSelection.jl scripts/check_julia_ground_truth_feasibility.jl <experiment_dir>")
    end

    exp_dir = normpath(ARGS[1])
    runs_dir = joinpath(exp_dir, "runs")
    isdir(runs_dir) || error("Missing runs directory: $runs_dir")

    all_rows = DataFrame()
    for run_name in sort(readdir(runs_dir))
        run_dir = joinpath(runs_dir, run_name)
        isdir(run_dir) || continue
        try
            rows = analyze_run(run_dir)
            append!(all_rows, rows)
        catch err
            @warn "Failed to analyze run" run_dir err
        end
    end

    out_file = joinpath(exp_dir, "julia_ground_truth_feasibility.csv")
    CSV.write(out_file, all_rows)
    println("Wrote Julia ground-truth feasibility summary to $out_file")
end

main()
