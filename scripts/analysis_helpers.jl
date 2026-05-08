module AnalysisHelpers

using CSV
using DataFrames
using Dates
using JSON
using Statistics
using TOML

push!(LOAD_PATH, joinpath(@__DIR__, "..", "StationSelection.jl", "src"))
using StationSelection

export PROJECT_ROOT, iter_metric_files, load_metrics, collect_runs, pair_runs, safe_slug
export build_cost_inputs, load_active_schedule, best_feasible_pair_cost
export compute_order_costs, compute_order_costs_by_scenario, compute_order_costs_by_scenario_from_df
export get_realized_walking_violation_rate
export theoretical_costs_by_scenario, theoretical_walking_violation_rates
export get_order_cost_total, get_order_cost_mean, get_order_cost_std, get_daily_cost_std
export get_od_unweighted_metrics, get_theoretical_od_metrics
export parse_station_selection, parse_scenario_activation, jaccard

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

load_metrics(metrics_path::String) = JSON.parsefile(metrics_path)

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

function get_full_station_path(run_dir::String)
    cluster_path = joinpath(run_dir, "backtest", "cluster_stations.csv")
    return isfile(cluster_path) ? cluster_path : joinpath(run_dir, "backtest", "simulation_inputs", "station.csv")
end

function load_station_dataframe(station_path::String)
    df = CSV.read(station_path, DataFrame)
    cols = Set(propertynames(df))

    if :station_id in cols
        rename!(df, :station_id => :id)
    end

    if (:station_lon in cols) && (:station_lat in cols)
        wgs_coords = [bd09_to_wgs84(Float64(df.station_lon[i]), Float64(df.station_lat[i])) for i in 1:nrow(df)]
        df.station_lon = first.(wgs_coords)
        df.station_lat = last.(wgs_coords)
        rename!(df, :station_lon => :lon, :station_lat => :lat)
    end

    (:id in propertynames(df) && :lon in propertynames(df) && :lat in propertynames(df)) ||
        error("station file must contain either (:id,:lon,:lat) or (:station_id,:station_lon,:station_lat): $station_path")

    return df
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
            "metrics_path" => metrics_path,
            "run_dir" => dirname(metrics_path),
            "timestamp" => get(metrics, "timestamp", nothing),
            "k" => Int(k),
            "in_vehicle_time_weight" => Float64(lam),
            "demand_quantile" => get(metrics, "demand_quantile", nothing),
            "Q_cap_quantile" => get(metrics, "Q_cap_quantile", nothing),
            "q_high_quantile" => get(metrics, "q_high_quantile", nothing),
            "model_type" => String(model_type),
        )
        key = (Int(k), Float64(lam))
        if model_type in ("NominalModel", "NominalFeasibleModel")
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

function build_cost_inputs(run_dir::String)
    cfg = TOML.parsefile(joinpath(run_dir, "config.toml"))
    params = cfg["parameters"]
    station_file = get_full_station_path(run_dir)
    segment_file = joinpath(run_dir, "backtest", "simulation_inputs", "segment.csv")
    stations = load_station_dataframe(station_file)
    walking_costs = compute_station_pairwise_costs(stations)
    routing_costs = read_routing_costs_from_segments(segment_file, stations)
    station_ids = sort(Int.(stations.id))
    max_walking_distance = Float64(get(params, "max_walking_distance", Inf))
    lambda_val = Float64(get(params, "in_vehicle_time_weight", 0.0))
    return station_ids, walking_costs, routing_costs, max_walking_distance, lambda_val
end

function _load_period_station_map(run_dir::String)::Vector{Tuple{Int, Vector{Int}}}
    schedule_file = joinpath(run_dir, "backtest", "active_station_schedule.csv")
    isfile(schedule_file) || return Tuple{Int, Vector{Int}}[]
    schedule_df = CSV.read(schedule_file, DataFrame)
    seen = Dict{Int, Vector{Int}}()
    for row in eachrow(schedule_df)
        h = Int(row.period_start_hour)
        haskey(seen, h) && continue
        seen[h] = sort(parse.(Int, split(String(row.active_station_ids))))
    end
    return sort([(h, ids) for (h, ids) in seen]; by=first)
end

function _get_period_start_hour(h::Int, period_map::Vector{Tuple{Int, Vector{Int}}})::Union{Int, Nothing}
    result = nothing
    for (ph, _) in period_map
        ph > h && break
        result = ph
    end
    return result
end

# Precompute minimum feasible (j,k) cost for every (origin_id, dest_id, period_start_hour).
# Consistent with valid_jk_pairs in the optimization model: restricts to active stations and
# respects max_walking_distance. Missing key = no feasible assignment exists.
function _build_od_cost_lookup(
    period_map::Vector{Tuple{Int, Vector{Int}}},
    walking_costs::Dict{Tuple{Int,Int}, Float64},
    routing_costs::Dict{Tuple{Int,Int}, Float64},
    station_ids::Vector{Int},
    lambda_val::Float64,
    max_walking_distance::Float64,
)::Dict{Tuple{Int,Int,Int}, Float64}
    lookup = Dict{Tuple{Int,Int,Int}, Float64}()
    for (ph, active_ids) in period_map
        isempty(active_ids) && continue
        for o in station_ids
            for d in station_ids
                o == d && continue
                c = best_feasible_pair_cost(o, d, active_ids, walking_costs, routing_costs, lambda_val, max_walking_distance)
                isnothing(c) || (lookup[(o, d, ph)] = c)
            end
        end
    end
    return lookup
end

function load_active_schedule(run_dir::String)
    schedule_file = joinpath(run_dir, "backtest", "active_station_schedule.csv")
    df = CSV.read(schedule_file, DataFrame)
    return [(String(row.scenario_label), sort(parse.(Int, split(String(row.active_station_ids))))) for row in eachrow(df)]
end

function best_feasible_pair_cost(origin_id::Int, dest_id::Int, active_ids::Vector{Int},
    walking_costs::Dict{Tuple{Int, Int}, Float64}, routing_costs::Dict{Tuple{Int, Int}, Float64},
    lambda_val::Float64, max_walking_distance::Float64)
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

function transform_dir_for_month(run_dir::String, month::String)
    month == "April" && return joinpath(run_dir, "backtest", "transform_in_sample")
    month == "May" && return joinpath(run_dir, "backtest", "transform")
    error("unexpected month: $month")
end

function iter_transformed_orders(run_dir::String, month::String)
    tdir = transform_dir_for_month(run_dir, month)
    manifest_path = joinpath(tdir, "daily_manifest.csv")
    isfile(manifest_path) || return String[]
    manifest = CSV.read(manifest_path, DataFrame)
    files = String[]
    for row in eachrow(manifest)
        orders_file = String(row.orders_file)
        isfile(orders_file) ? push!(files, orders_file) : begin
            fallback = joinpath(tdir, "daily_orders", basename(orders_file))
            isfile(fallback) && push!(files, fallback)
        end
    end
    return files
end

function compute_order_costs(run_dir::String, month::String, lambda_val::Float64)
    station_ids, walking_costs, routing_costs, max_walking_distance, _ = build_cost_inputs(run_dir)
    period_map = _load_period_station_map(run_dir)
    cost_lookup = _build_od_cost_lookup(period_map, walking_costs, routing_costs, station_ids, lambda_val, max_walking_distance)
    costs = Float64[]
    for orders_path in iter_transformed_orders(run_dir, month)
        df = CSV.read(orders_path, DataFrame)
        for row in eachrow(df)
            origin_id = Int(row.origin_station_id)
            dest_id = Int(row.destination_station_id)
            (origin_id == 0 || dest_id == 0) && continue
            ph = _get_period_start_hour(Dates.hour(DateTime(String(row.order_time), "yyyy-mm-dd HH:MM:SS")), period_map)
            isnothing(ph) && continue
            c = get(cost_lookup, (origin_id, dest_id, ph), nothing)
            isnothing(c) || push!(costs, c)
        end
    end
    return costs
end

function compute_order_costs_by_scenario_from_df(run_dir::String, orders_df::DataFrame, lambda_val::Float64)
    station_ids, walking_costs, routing_costs, max_walking_distance, _ = build_cost_inputs(run_dir)
    period_map = _load_period_station_map(run_dir)
    cost_lookup = _build_od_cost_lookup(period_map, walking_costs, routing_costs, station_ids, lambda_val, max_walking_distance)
    result = Dict{String, Vector{Float64}}()
    for label in ("period_1", "period_2", "period_3", "period_4")
        result[label] = Float64[]
    end
    for row in eachrow(orders_df)
        origin_id = Int(row.origin_station_id)
        dest_id = Int(row.destination_station_id)
        (origin_id == 0 || dest_id == 0) && continue
        h = Dates.hour(row.order_time)
        label = period_label_from_hour(h)
        isnothing(label) && continue
        ph = _get_period_start_hour(h, period_map)
        isnothing(ph) && continue
        c = get(cost_lookup, (origin_id, dest_id, ph), nothing)
        isnothing(c) || push!(result[label], c)
    end
    return result
end

function period_label_from_hour(hour::Int)
    6 <= hour < 10 && return "period_1"
    10 <= hour < 15 && return "period_2"
    15 <= hour < 20 && return "period_3"
    20 <= hour < 24 && return "period_4"
    return nothing
end

function compute_order_costs_by_scenario(run_dir::String, month::String, lambda_val::Float64)
    station_ids, walking_costs, routing_costs, max_walking_distance, _ = build_cost_inputs(run_dir)
    period_map = _load_period_station_map(run_dir)
    cost_lookup = _build_od_cost_lookup(period_map, walking_costs, routing_costs, station_ids, lambda_val, max_walking_distance)
    result = Dict{String, Vector{Float64}}()
    for label in ("period_1", "period_2", "period_3", "period_4")
        result[label] = Float64[]
    end
    for orders_path in iter_transformed_orders(run_dir, month)
        df = CSV.read(orders_path, DataFrame)
        for row in eachrow(df)
            origin_id = Int(row.origin_station_id)
            dest_id = Int(row.destination_station_id)
            (origin_id == 0 || dest_id == 0) && continue
            h = Dates.hour(DateTime(String(row.order_time), "yyyy-mm-dd HH:MM:SS"))
            label = period_label_from_hour(h)
            isnothing(label) && continue
            ph = _get_period_start_hour(h, period_map)
            isnothing(ph) && continue
            c = get(cost_lookup, (origin_id, dest_id, ph), nothing)
            isnothing(c) || push!(result[label], c)
        end
    end
    return result
end

function get_realized_walking_violation_rate(run_dir::String, month::String)
    metrics = load_metrics(joinpath(run_dir, "metrics.json"))
    block = month == "April" ? get(metrics, "in_sample_direct", Dict{String, Any}()) : get(metrics, "direct_backtest", Dict{String, Any}())
    total_orders = get(block, "total_orders", nothing)
    walking_viol = get(block, "orders_walking_violation", nothing)
    (isnothing(total_orders) || isnothing(walking_viol) || total_orders == 0) && return nothing
    return Float64(walking_viol) / Float64(total_orders)
end

function theoretical_costs_by_scenario(run_dir::String)
    station_ids, walking_costs, routing_costs, max_walking_distance, lambda_val = build_cost_inputs(run_dir)
    result = Dict{String, Vector{Float64}}()
    for (label, active_ids) in load_active_schedule(run_dir)
        costs = Float64[]
        for o in station_ids, d in station_ids
            o == d && continue
            best = best_feasible_pair_cost(o, d, active_ids, walking_costs, routing_costs, lambda_val, max_walking_distance)
            !isnothing(best) && push!(costs, best)
        end
        result[label] = sort(costs)
    end
    return result
end

function theoretical_walking_violation_rates(run_dir::String)
    station_ids, walking_costs, routing_costs, max_walking_distance, lambda_val = build_cost_inputs(run_dir)
    rates = Dict{String, Union{Float64, Missing}}()
    vals = Float64[]
    for (label, active_ids) in load_active_schedule(run_dir)
        total = 0
        feasible = 0
        for o in station_ids, d in station_ids
            o == d && continue
            total += 1
            !isnothing(best_feasible_pair_cost(o, d, active_ids, walking_costs, routing_costs, lambda_val, max_walking_distance)) && (feasible += 1)
        end
        rate = total == 0 ? missing : 1.0 - feasible / total
        rates[label] = rate
        !ismissing(rate) && push!(vals, rate)
    end
    rates["scenario_avg"] = isempty(vals) ? missing : mean(vals)
    return rates
end

get_order_cost_total(run_dir::String, month::String, lambda_val::Float64) = (vals = compute_order_costs(run_dir, month, lambda_val); isempty(vals) ? nothing : sum(vals))
get_order_cost_mean(run_dir::String, month::String, lambda_val::Float64) = (vals = compute_order_costs(run_dir, month, lambda_val); isempty(vals) ? nothing : mean(vals))
get_order_cost_std(run_dir::String, month::String, lambda_val::Float64) = (vals = compute_order_costs(run_dir, month, lambda_val); length(vals) <= 1 ? nothing : std(vals; corrected=false))

function get_daily_cost_std(run_dir::String, month::String, lambda_val::Float64)
    station_ids, walking_costs, routing_costs, max_walking_distance, _ = build_cost_inputs(run_dir)
    period_map = _load_period_station_map(run_dir)
    cost_lookup = _build_od_cost_lookup(period_map, walking_costs, routing_costs, station_ids, lambda_val, max_walking_distance)
    totals = Float64[]
    for orders_path in iter_transformed_orders(run_dir, month)
        df = CSV.read(orders_path, DataFrame)
        total = 0.0
        for row in eachrow(df)
            origin_id = Int(row.origin_station_id)
            dest_id = Int(row.destination_station_id)
            (origin_id == 0 || dest_id == 0) && continue
            ph = _get_period_start_hour(Dates.hour(DateTime(String(row.order_time), "yyyy-mm-dd HH:MM:SS")), period_map)
            isnothing(ph) && continue
            c = get(cost_lookup, (origin_id, dest_id, ph), nothing)
            isnothing(c) || (total += c)
        end
        push!(totals, total)
    end
    return length(totals) <= 1 ? nothing : std(totals; corrected=false)
end

function get_od_unweighted_metrics(run_dir::String, month::String, lambda_val::Float64)
    od_costs = Dict{Tuple{Int, Int}, Vector{Float64}}()
    _, walking_costs, routing_costs, _, _ = build_cost_inputs(run_dir)
    for orders_path in iter_transformed_orders(run_dir, month)
        df = CSV.read(orders_path, DataFrame)
        for row in eachrow(df)
            pickup_id = Int(round(Float64(get(row, :assigned_pickup_id, 0))))
            dropoff_id = Int(round(Float64(get(row, :assigned_dropoff_id, 0))))
            (pickup_id == 0 || dropoff_id == 0) && continue
            origin_id = Int(row.origin_station_id)
            dest_id = Int(row.destination_station_id)
            route_cost = pickup_id == dropoff_id ? 0.0 : get(routing_costs, (pickup_id, dropoff_id), Inf)
            isfinite(route_cost) || continue
            walk_cost = get(walking_costs, (origin_id, pickup_id), Inf) + get(walking_costs, (dropoff_id, dest_id), Inf)
            isfinite(walk_cost) || continue
            push!(get!(od_costs, (origin_id, dest_id), Float64[]), walk_cost + lambda_val * route_cost)
        end
    end
    isempty(od_costs) && return (mean_od_mean_cost_unweighted=nothing, std_od_mean_cost_unweighted=nothing)
    od_means = [mean(v) for v in values(od_costs) if !isempty(v)]
    return (
        mean_od_mean_cost_unweighted = isempty(od_means) ? nothing : mean(od_means),
        std_od_mean_cost_unweighted = length(od_means) <= 1 ? nothing : std(od_means; corrected=false),
    )
end

function get_theoretical_od_metrics(run_dir::String)
    costs_by_scenario = theoretical_costs_by_scenario(run_dir)
    flat = [v for costs in values(costs_by_scenario) for v in costs]
    isempty(flat) && return (mean_od_mean_cost_theoretical=nothing, std_od_mean_cost_theoretical=nothing)
    return (
        mean_od_mean_cost_theoretical = mean(flat),
        std_od_mean_cost_theoretical = length(flat) <= 1 ? nothing : std(flat; corrected=false),
    )
end

function parse_station_selection(run_dir::String)
    path = joinpath(run_dir, "variable_exports", "station_selection.csv")
    df = CSV.read(path, DataFrame)
    return Set(Int.(df[df.selected .== 1, :station_id]))
end

function parse_scenario_activation(run_dir::String)
    path = joinpath(run_dir, "variable_exports", "scenario_activation.csv")
    df = CSV.read(path, DataFrame)
    s = Set{Tuple{Int, String}}()
    for row in eachrow(df)
        Float64(row.value) >= 0.5 || continue
        push!(s, (Int(row.station_id), String(row.scenario_label)))
    end
    return s
end

function jaccard(a::Set, b::Set)
    u = union(a, b)
    isempty(u) && return nothing
    return length(intersect(a, b)) / length(u)
end

end
