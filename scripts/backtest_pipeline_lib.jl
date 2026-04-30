using CSV
using DataFrames
using Dates
using JSON
using Statistics

using StationSelection
using MicroTransitSimulator

function backtest_model_type(model_type::String)::String
    if model_type in ("NominalModel", "RobustTotalDemandCapModel")
        return "ClusteringTwoStageODModel"
    end
    error("No backtest transform mapping defined for model_type=$model_type")
end

# NominalTwoStageODModel appears in run metadata as the model_type field; map it to the
# same backtest transform as ClusteringTwoStageODModel.
function backtest_model_type_from_run(run_dir::String)::String
    metrics_file = joinpath(run_dir, "metrics.json")
    if isfile(metrics_file)
        m = JSON.parsefile(metrics_file)
        raw = get(get(m, "model", Dict()), "type", "NominalModel")
        return backtest_model_type(raw)
    end
    return "ClusteringTwoStageODModel"
end

function _json_number(value)
    return isfinite(value) ? value : nothing
end

function _scenario_profile_symbol(params::Dict)::Symbol
    return Symbol(get(params, "scenario_profile", "four_period"))
end

function _month_window(params::Dict)
    start_dt = DateTime("$(params["start_date"]) 00:00:00", "yyyy-mm-dd HH:MM:SS")
    end_dt = DateTime("$(params["end_date"]) 23:59:59", "yyyy-mm-dd HH:MM:SS")
    return start_dt, end_dt
end

function _backtest_month_window(params::Dict)
    start_str = get(params, "backtest_start_date", params["start_date"])
    end_str   = get(params, "backtest_end_date",   params["end_date"])
    start_dt = DateTime("$(start_str) 00:00:00", "yyyy-mm-dd HH:MM:SS")
    end_dt   = DateTime("$(end_str) 23:59:59",   "yyyy-mm-dd HH:MM:SS")
    return start_dt, end_dt
end

function _simulation_settings(cfg::Dict)::Dict{String, Any}
    sim_cfg = get(cfg, "simulation", Dict{String, Any}())
    return Dict{String, Any}(
        "base_date" => string(Date(get(sim_cfg, "base_date", Date(2025, 5, 1)))),
        "optimization_interval" => Float64(get(sim_cfg, "optimization_interval", 300.0)),
        "max_wait_time" => Float64(get(sim_cfg, "max_wait_time", 900.0)),
        "max_time" => Float64(get(sim_cfg, "max_time", 86400.0)),
        "assignment_mode" => string(get(sim_cfg, "assignment_mode", "batch")),
        "run_until_complete" => Bool(get(sim_cfg, "run_until_complete", true)),
        "verbose" => Bool(get(sim_cfg, "verbose", false)),
        "batch_write_logs" => Bool(get(sim_cfg, "batch_write_logs", false)),
        "batch_write_size" => Int(get(sim_cfg, "batch_write_size", 1000)),
    )
end

function build_active_station_schedule(selection_run_dir::String)::DataFrame
    scenario_file = joinpath(selection_run_dir, "variable_exports", "scenario_info.csv")
    activation_file = joinpath(selection_run_dir, "variable_exports", "scenario_activation.csv")
    scenario_df = CSV.read(scenario_file, DataFrame)
    activation_df = CSV.read(activation_file, DataFrame)

    active_by_scenario = Dict{Int, Vector{Int}}()
    for grp in groupby(filter(r -> r.value >= 0.5, activation_df), :scenario_idx)
        active_by_scenario[Int(grp[1, :scenario_idx])] = sort(Int.(grp.station_id))
    end

    rows = NamedTuple[]
    for row in eachrow(scenario_df)
        start_dt = DateTime(string(row.start_time))
        push!(rows, (
            scenario_idx = Int(row.scenario_idx),
            scenario_label = string(row.label),
            date = string(Date(start_dt)),
            period_start_hour = hour(start_dt),
            active_station_count = length(get(active_by_scenario, Int(row.scenario_idx), Int[])),
            active_station_ids = join(get(active_by_scenario, Int(row.scenario_idx), Int[]), " "),
        ))
    end

    return DataFrame(rows)
end

function _build_period_station_map(schedule_df::DataFrame)::Vector{Tuple{Int, Vector{Int}}}
    seen = Dict{Int, Vector{Int}}()
    for row in eachrow(schedule_df)
        h = Int(row.period_start_hour)
        haskey(seen, h) && continue
        seen[h] = parse_station_list(string(row.active_station_ids))
    end
    return sort([(h, ids) for (h, ids) in seen]; by=first)
end

function _period_start_hour(h::Int, period_map::Vector{Tuple{Int, Vector{Int}}})::Union{Int, Nothing}
    result = nothing
    for (ph, _) in period_map
        ph > h && break
        result = ph
    end
    return result
end

# Key: (origin_id, dest_id, period_start_hour)
# Value: (walk_j, walk_k, route) for the global min-cost (j,k) pair from that period's active stations.
# Only entries where a finite-cost pair exists are stored; missing key = no valid pair.
const _CostEntry = Tuple{Float64, Float64, Float64}

function _build_period_cost_lookup(
    period_map    :: Vector{Tuple{Int, Vector{Int}}},
    walking_costs :: Dict{Tuple{Int,Int}, Float64},
    routing_costs :: Dict{Tuple{Int,Int}, Float64},
    station_ids   :: Vector{Int},
    lambda_val    :: Float64,
)::Dict{Tuple{Int,Int,Int}, _CostEntry}
    lookup = Dict{Tuple{Int,Int,Int}, _CostEntry}()

    for (ph, active_ids) in period_map
        isempty(active_ids) && continue
        for o in station_ids
            for d in station_ids
                o == d && continue
                best_cost = Inf
                best_wj   = NaN
                best_wk   = NaN
                best_r    = NaN

                for j in active_ids
                    w_j = get(walking_costs, (o, j), Inf)
                    isfinite(w_j) || continue
                    for k in active_ids
                        w_k = get(walking_costs, (k, d), Inf)
                        isfinite(w_k) || continue
                        r = j == k ? 0.0 : get(routing_costs, (j, k), Inf)
                        isfinite(r) || continue
                        c = (w_j + w_k) + lambda_val * r
                        if c < best_cost
                            best_cost = c
                            best_wj   = w_j
                            best_wk   = w_k
                            best_r    = r
                        end
                    end
                end

                isfinite(best_cost) || continue
                lookup[(o, d, ph)] = (best_wj, best_wk, best_r)
            end
        end
    end

    return lookup
end

"""
    compute_period_aware_direct_cost(orders_df, active_schedule_file, stations_file, segment_file,
                                     lambda_val; max_walking_distance)

For each order, determine its time-of-day period, retrieve z[j,s] active stations for that period,
and find the minimum-cost (j,k) assignment: walk(o→j) + walk(k→d) + λ·route(j→k).

Costs are computed via a prebuilt lookup table keyed on (origin_id, dest_id, period_start_hour),
so the inner j×k search runs once per unique (o,d,period) triple rather than once per order.
Walking violations are flagged post-hoc: if the globally optimal (j,k) has either leg exceeding
max_walking_distance, the order is counted as a violation but its cost is still included.

`orders_df` must have columns: `origin_station_id`, `destination_station_id`, `order_time`.
"""
function compute_period_aware_direct_cost(
    orders_df::DataFrame,
    active_schedule_file::String,
    stations_file::String,
    segment_file::String,
    lambda_val::Float64;
    max_walking_distance::Float64 = Inf,
)
    schedule_df = CSV.read(active_schedule_file, DataFrame)
    period_map  = _build_period_station_map(schedule_df)

    stations      = read_candidate_stations(stations_file)
    walking_costs = compute_station_pairwise_costs(stations)
    routing_costs = read_routing_costs_from_segments(segment_file, stations)

    # Precompute best (j,k) cost for every (origin, dest, period) combination once.
    cost_lookup = _build_period_cost_lookup(
        period_map, walking_costs, routing_costs, stations.id, lambda_val
    )

    total_walk           = 0.0
    total_route          = 0.0
    fully_assigned       = 0
    n_outside_period     = 0
    n_missing_station_id = 0
    n_walking_violations = 0
    n_no_routing         = 0

    for row in eachrow(orders_df)
        origin_id = Int(row.origin_station_id)
        dest_id   = Int(row.destination_station_id)
        if origin_id == 0 || dest_id == 0
            n_missing_station_id += 1
            continue
        end

        ph = _period_start_hour(hour(DateTime(string(row.order_time), "yyyy-mm-dd HH:MM:SS")), period_map)
        if isnothing(ph)
            n_outside_period += 1
            continue
        end

        entry = get(cost_lookup, (origin_id, dest_id, ph), nothing)
        if isnothing(entry)
            n_no_routing += 1
            continue
        end

        w_j, w_k, r = entry
        if w_j > max_walking_distance || w_k > max_walking_distance
            n_walking_violations += 1
        end

        total_walk  += w_j + w_k
        total_route += r
        fully_assigned += 1
    end

    weighted_total = total_walk + lambda_val * total_route
    return Dict{String, Any}(
        "total_orders"                 => nrow(orders_df),
        "fully_assigned_orders"        => fully_assigned,
        "orders_outside_period"        => n_outside_period,
        "orders_missing_station_id"    => n_missing_station_id,
        "orders_walking_violation"     => n_walking_violations,
        "orders_no_routing"            => n_no_routing,
        "max_walking_distance_seconds" => max_walking_distance,
        "walking_time_seconds"         => total_walk,
        "routing_time_seconds"         => total_route,
        "in_vehicle_time_weight"       => lambda_val,
        "weighted_total_cost"          => weighted_total,
        "mean_weighted_cost_per_order" => fully_assigned > 0 ? weighted_total / fully_assigned : 0.0,
    )
end

function compute_direct_backtest_metrics(
    transformed_orders::DataFrame,
    stations_file::String,
    segment_file::String,
    lambda_val::Float64,
)
    stations = read_candidate_stations(stations_file)
    walking_costs = compute_station_pairwise_costs(stations)
    routing_costs = read_routing_costs_from_segments(segment_file, stations)

    total_walk = 0.0
    total_route = 0.0
    fully_assigned = 0
    missing_cost_pairs = 0

    for row in eachrow(transformed_orders)
        pickup_id = Int(row.assigned_pickup_id)
        dropoff_id = Int(row.assigned_dropoff_id)
        if pickup_id == 0 || dropoff_id == 0
            continue
        end

        origin_id = Int(row.origin_station_id)
        dest_id = Int(row.destination_station_id)
        walk_cost = get(walking_costs, (origin_id, pickup_id), NaN) +
                    get(walking_costs, (dropoff_id, dest_id), NaN)
        route_cost = get(routing_costs, (pickup_id, dropoff_id), NaN)

        if !isfinite(walk_cost) || !isfinite(route_cost)
            missing_cost_pairs += 1
            continue
        end

        total_walk += walk_cost
        total_route += route_cost
        fully_assigned += 1
    end

    weighted_total = total_walk + lambda_val * total_route
    return Dict{String, Any}(
        "total_orders" => nrow(transformed_orders),
        "fully_assigned_orders" => fully_assigned,
        "orders_missing_cost_pair" => missing_cost_pairs,
        "walking_time_seconds" => total_walk,
        "routing_time_seconds" => total_route,
        "in_vehicle_time_weight" => lambda_val,
        "weighted_total_cost" => weighted_total,
        "mean_weighted_cost_per_order" => fully_assigned > 0 ? weighted_total / fully_assigned : 0.0,
    )
end

function prepare_backtest_artifacts(project_root::String, cfg::Dict, run_dir::String)
    params = cfg["parameters"]
    data_cfg = cfg["data"]
    model_type = get(get(cfg, "model", Dict()), "type", "NominalModel")

    backtest_dir = joinpath(run_dir, "backtest")
    transform_dir = joinpath(backtest_dir, "transform")
    sim_input_dir = joinpath(backtest_dir, "simulation_inputs")
    sim_runs_dir = joinpath(backtest_dir, "simulation_runs")
    mkpath(transform_dir)
    mkpath(sim_input_dir)
    mkpath(sim_runs_dir)

    base_station_file = joinpath(project_root, data_cfg["station_file"])
    base_segment_file = joinpath(project_root, data_cfg["segment_file"])
    base_order_file = joinpath(project_root, data_cfg["order_file"])
    base_vehicle_file = joinpath(project_root, get(data_cfg, "vehicle_file", "Data/zhuzhou_data/vehicle.csv"))
    station_selection_file = joinpath(run_dir, "variable_exports", "station_selection.csv")

    backtest_method = backtest_model_type(model_type)
    start_dt, end_dt = _backtest_month_window(params)
    scenario_profile = _scenario_profile_symbol(params)

    # Build a cluster file that has geometry (:id, :lon, :lat) + :selected flag.
    # station_selection.csv only has array_idx/station_id/selected/value; transform_orders
    # needs :id, :lat, :lon (WGS-84) which read_candidate_stations provides.
    cluster_stations_file = joinpath(backtest_dir, "cluster_stations.csv")
    let base_stations = read_candidate_stations(base_station_file),
        sel_df        = CSV.read(station_selection_file, DataFrame)
        cluster_df = leftjoin(
            base_stations,
            select(sel_df, :station_id => :id, :selected),
            on = :id,
        )
        cluster_df.selected = coalesce.(cluster_df.selected, 0)
        CSV.write(cluster_stations_file, cluster_df)
    end

    transformed_df, transform_stats, daily_orders_manifest = transform_orders_for_month_backtest(
        base_order_file,
        run_dir,
        cluster_stations_file,
        backtest_method;
        output_dir=transform_dir,
        start_date=start_dt,
        end_date=end_dt,
        scenario_profile=scenario_profile,
    )

    # Build active station schedule first — needed for period-aware direct cost computation
    schedule_df = build_active_station_schedule(run_dir)
    active_schedule_file = joinpath(backtest_dir, "active_station_schedule.csv")
    CSV.write(active_schedule_file, schedule_df)

    station_df = prepare_station_data(base_station_file, station_selection_file)
    station_file = joinpath(sim_input_dir, "station.csv")
    CSV.write(station_file, station_df)

    segment_df = prepare_segment_data(base_segment_file, station_selection_file)
    segment_file = joinpath(sim_input_dir, "segment.csv")
    CSV.write(segment_file, segment_df)

    selected_station_ids = sort(Int.(station_df[station_df.is_station .== true, :station_id]))
    vehicle_df = prepare_vehicle_data(base_vehicle_file, selected_station_ids)
    vehicle_file = joinpath(sim_input_dir, "vehicle.csv")
    CSV.write(vehicle_file, vehicle_df)

    sim_settings = _simulation_settings(cfg)
    manifest_rows = NamedTuple[]
    for row in eachrow(daily_orders_manifest)
        output_dir = joinpath(sim_runs_dir, row.date)
        push!(manifest_rows, (
            day_index = Int(row.day_index),
            date = string(row.date),
            order_count = Int(row.order_count),
            orders_file = string(row.orders_file),
            station_file = station_file,
            segment_file = segment_file,
            vehicle_file = vehicle_file,
            output_dir = output_dir,
            base_date = sim_settings["base_date"],
            optimization_interval = sim_settings["optimization_interval"],
            max_wait_time = sim_settings["max_wait_time"],
            max_time = sim_settings["max_time"],
            assignment_mode = sim_settings["assignment_mode"],
            run_until_complete = sim_settings["run_until_complete"],
            verbose = sim_settings["verbose"],
            batch_write_logs = sim_settings["batch_write_logs"],
            batch_write_size = sim_settings["batch_write_size"],
        ))
    end

    manifest_df = DataFrame(manifest_rows)
    manifest_file = joinpath(backtest_dir, "simulation_manifest.csv")
    CSV.write(manifest_file, manifest_df)

    lambda_val            = Float64(get(params, "in_vehicle_time_weight", 1.0))
    max_walking_distance  = Float64(get(params, "max_walking_distance", Inf))

    # Out-of-sample direct backtest (May): use period-appropriate z[j,s] active stations
    direct_metrics = compute_period_aware_direct_cost(
        transformed_df,
        active_schedule_file,
        base_station_file,
        base_segment_file,
        lambda_val;
        max_walking_distance = max_walking_distance,
    )
    direct_metrics["transform"] = transform_stats
    direct_metrics["active_station_schedule_file"] = active_schedule_file
    direct_metrics_file = joinpath(backtest_dir, "direct_backtest_metrics.json")
    open(direct_metrics_file, "w") do io
        JSON.print(io, direct_metrics, 2)
    end

    # In-sample direct cost (April) — apply the same station layout to optimization-month orders
    in_sample_transform_dir = joinpath(backtest_dir, "transform_in_sample")
    mkpath(in_sample_transform_dir)
    in_sample_start_dt, in_sample_end_dt = _month_window(params)

    in_sample_transformed_df, in_sample_transform_stats, _ = transform_orders_for_month_backtest(
        base_order_file,
        run_dir,
        cluster_stations_file,
        backtest_method;
        output_dir=in_sample_transform_dir,
        start_date=in_sample_start_dt,
        end_date=in_sample_end_dt,
        scenario_profile=scenario_profile,
    )

    # In-sample direct cost (April): use period-appropriate z[j,s] active stations
    in_sample_direct_metrics = compute_period_aware_direct_cost(
        in_sample_transformed_df,
        active_schedule_file,
        base_station_file,
        base_segment_file,
        lambda_val;
        max_walking_distance = max_walking_distance,
    )
    in_sample_direct_metrics["transform"] = in_sample_transform_stats
    in_sample_direct_metrics_file = joinpath(backtest_dir, "in_sample_direct_metrics.json")
    open(in_sample_direct_metrics_file, "w") do io
        JSON.print(io, in_sample_direct_metrics, 2)
    end

    return Dict{String, Any}(
        "backtest_dir" => backtest_dir,
        "transform_dir" => transform_dir,
        "simulation_input_dir" => sim_input_dir,
        "simulation_runs_dir" => sim_runs_dir,
        "transformed_month_file" => transform_stats["month_order_file"],
        "daily_manifest_file" => transform_stats["daily_manifest_file"],
        "simulation_manifest_file" => manifest_file,
        "direct_metrics_file" => direct_metrics_file,
        "in_sample_direct_metrics_file" => in_sample_direct_metrics_file,
        "active_station_schedule_file" => active_schedule_file,
        "n_days" => nrow(manifest_df),
    )
end

function run_simulation_day_from_manifest(manifest_file::String, day_index::Int)
    manifest = CSV.read(manifest_file, DataFrame)
    rows = filter(r -> Int(r.day_index) == day_index, manifest)
    nrow(rows) == 1 || error("Expected exactly one manifest row for day_index=$day_index in $manifest_file")
    row = rows[1, :]

    output_dir = String(row.output_dir)
    mkpath(output_dir)

    state = run_simulation_from_files(
        String(row.station_file),
        String(row.segment_file),
        String(row.vehicle_file),
        String(row.orders_file),
        output_dir;
        base_date=DateTime("$(row.base_date) 00:00:00", "yyyy-mm-dd HH:MM:SS"),
        optimization_interval=Float64(row.optimization_interval),
        max_wait_time=Float64(row.max_wait_time),
        max_time=Float64(row.max_time),
        verbose=Bool(row.verbose),
        assignment_mode=Symbol(row.assignment_mode),
        batch_write_logs=Bool(row.batch_write_logs),
        batch_write_size=Int(row.batch_write_size),
        run_until_complete=Bool(row.run_until_complete),
    )

    return state
end

function _safe_summary(summary_file::String)
    return isfile(summary_file) ? JSON.parsefile(summary_file) : nothing
end

function aggregate_simulation_outputs(backtest_dir::String)
    manifest_file = joinpath(backtest_dir, "simulation_manifest.csv")
    direct_metrics_file = joinpath(backtest_dir, "direct_backtest_metrics.json")
    manifest = CSV.read(manifest_file, DataFrame)

    n_days_completed = 0
    missing_days = String[]
    total_requests = 0
    completed_requests = 0
    expired_requests = 0
    unassigned_requests = 0
    sum_waiting = 0.0
    sum_in_vehicle = 0.0
    sum_walking = 0.0
    total_vehicle_distance = 0.0
    total_vehicle_count = 0
    total_events = 0
    total_sim_duration = 0.0
    total_wall_time = 0.0

    daily_rows = NamedTuple[]
    for row in eachrow(manifest)
        summary_file = joinpath(row.output_dir, "summary_statistics.json")
        summary = _safe_summary(summary_file)
        if isnothing(summary)
            push!(missing_days, string(row.date))
            continue
        end

        n_days_completed += 1
        sim = summary["simulation"]
        passengers = summary["passengers"]
        vehicles = summary["vehicles"]

        total_requests += Int(passengers["total_requests"])
        completed_requests += Int(passengers["completed_requests"])
        expired_requests += Int(passengers["expired_requests"])
        unassigned_requests += Int(passengers["unassigned_requests"])
        sum_waiting += Float64(get(passengers, "mean_waiting_time", 0.0)) * Int(passengers["completed_requests"])
        sum_in_vehicle += Float64(get(passengers, "mean_in_vehicle_time", 0.0)) * Int(passengers["completed_requests"])
        sum_walking += Float64(get(passengers, "mean_walking_distance", 0.0)) * Int(passengers["completed_requests"])
        total_vehicle_distance += Float64(vehicles["total_distance_all_vehicles"])
        total_vehicle_count += Int(vehicles["num_vehicles"])
        total_events += Int(sim["events_processed"])
        total_sim_duration += Float64(sim["simulation_duration"])
        total_wall_time += Float64(sim["wall_time"])

        push!(daily_rows, (
            date = string(row.date),
            total_requests = Int(passengers["total_requests"]),
            completed_requests = Int(passengers["completed_requests"]),
            completion_rate = Float64(passengers["completion_rate"]),
            total_distance_all_vehicles = Float64(vehicles["total_distance_all_vehicles"]),
            wall_time = Float64(sim["wall_time"]),
        ))
    end

    aggregate = Dict{String, Any}(
        "manifest_days" => nrow(manifest),
        "completed_days" => n_days_completed,
        "missing_days" => missing_days,
        "passengers" => Dict(
            "total_requests" => total_requests,
            "completed_requests" => completed_requests,
            "expired_requests" => expired_requests,
            "unassigned_requests" => unassigned_requests,
            "completion_rate" => total_requests > 0 ? completed_requests / total_requests : 0.0,
            "mean_waiting_time" => completed_requests > 0 ? sum_waiting / completed_requests : 0.0,
            "mean_in_vehicle_time" => completed_requests > 0 ? sum_in_vehicle / completed_requests : 0.0,
            "mean_walking_distance" => completed_requests > 0 ? sum_walking / completed_requests : 0.0,
        ),
        "vehicles" => Dict(
            "total_distance_all_vehicles" => total_vehicle_distance,
            "mean_distance_per_vehicle_day" => total_vehicle_count > 0 ? total_vehicle_distance / total_vehicle_count : 0.0,
        ),
        "simulation" => Dict(
            "events_processed" => total_events,
            "simulation_duration" => total_sim_duration,
            "wall_time" => total_wall_time,
        ),
        "daily_rows_file" => joinpath(backtest_dir, "simulation_daily_summary.csv"),
        "direct_metrics_file" => direct_metrics_file,
    )

    daily_df = DataFrame(daily_rows)
    CSV.write(aggregate["daily_rows_file"], daily_df)

    if isfile(direct_metrics_file)
        aggregate["direct_backtest"] = JSON.parsefile(direct_metrics_file)
    end

    aggregate_file = joinpath(backtest_dir, "simulation_backtest_summary.json")
    open(aggregate_file, "w") do io
        JSON.print(io, aggregate, 2)
    end

    aggregate["aggregate_file"] = aggregate_file
    return aggregate
end
