#!/usr/bin/env julia
# Plots empirical CDFs of per-order cost, one figure per (k, scenario),
# with selected models overlaid on the same axes.
#
# Usage:
#   julia --project=StationSelection.jl scripts/plot_cost_cdfs_by_scenario.jl \
#       <exp_dir> <full_od_orders_csv> [output_dir] [robust_quantile]
#
# Optional robust_quantile (e.g. "0.95") restricts robust lines to that quantile only.
# Omit or pass "all" to show all three robust quantiles.

using CSV, DataFrames, Statistics, Dates, Plots
include(joinpath(@__DIR__, "analysis_helpers.jl"))
using .AnalysisHelpers

const PERIOD_LABEL = Dict(
    "period_1" => "Morning (06–10h)",
    "period_2" => "Afternoon (10–15h)",
    "period_3" => "Evening (15–20h)",
    "period_4" => "Night (20–24h)",
)

struct ModelSpec
    key::String
    label::String
    color::Symbol
    style::Symbol
end

function model_specs(model_type::String, quantile)
    if model_type == "NominalModel"
        return ModelSpec("Nominal", "Nominal", :black, :solid)
    elseif model_type == "SmoothedNominalModel"
        return ModelSpec("Smoothed", "Smoothed nominal", :royalblue, :dash)
    elseif model_type == "RobustTotalDemandCapModel"
        q = ismissing(quantile) ? "?" : string(quantile)
        colors = Dict("0.9" => :darkgreen, "0.95" => :orange, "0.99" => :crimson)
        styles = Dict("0.9" => :solid, "0.95" => :dash, "0.99" => :dot)
        return ModelSpec("Robust_$q", "Robust q=$q",
            get(colors, q, :gray), get(styles, q, :solid))
    end
    return ModelSpec(model_type, model_type, :gray, :solid)
end

function model_sort_key(model_type::String, quantile)
    model_type == "NominalModel" && return (1, 0.0)
    model_type == "SmoothedNominalModel" && return (2, 0.0)
    return (3, ismissing(quantile) ? 0.0 : Float64(quantile))
end

function collect_costs(exp_dir::String, may_orders::DataFrame, k_filter::Int;
                       robust_quantile::Union{Float64, Nothing}=nothing)
    entries = []
    for metrics_path in iter_metric_files(exp_dir)
        metrics = load_metrics(metrics_path)
        model_type = get(metrics, "model_type", nothing)
        k = get(metrics, "k", nothing)
        lam = get(metrics, "in_vehicle_time_weight", nothing)
        (isnothing(model_type) || isnothing(k) || isnothing(lam)) && continue
        Int(k) == k_filter || continue

        mtype = String(model_type)
        mtype == "SmoothedNominalModel" && continue
        quantile = get(metrics, "demand_quantile", missing)

        if mtype == "RobustTotalDemandCapModel" && !isnothing(robust_quantile)
            (ismissing(quantile) || Float64(quantile) != robust_quantile) && continue
        end

        run_dir = dirname(metrics_path)
        lambda_val = Float64(lam)

        costs_by_period = compute_order_costs_by_scenario_from_df(run_dir, may_orders, lambda_val)
        spec = model_specs(mtype, quantile)
        push!(entries, (spec=spec, sort_key=model_sort_key(mtype, quantile), costs=costs_by_period))
    end
    sort!(entries; by=e -> e.sort_key)
    return entries
end

function plot_scenario_cdfs(entries, period::String, k_val::Int, out_path::String)
    title_str = "k=$(k_val)  $(get(PERIOD_LABEL, period, period))"
    p = plot(; xlabel="Weighted cost per order", ylabel="Cumulative fraction",
        title=title_str, legend=:bottomright, size=(680, 440),
        titlefontsize=10, guidefontsize=9, legendfontsize=8)

    for e in entries
        costs = get(e.costs, period, Float64[])
        isempty(costs) && continue
        sorted_c = sort(costs)
        cdf_y = (1:length(sorted_c)) ./ length(sorted_c)
        plot!(p, sorted_c, cdf_y;
            label=e.spec.label,
            color=e.spec.color,
            linestyle=e.spec.style,
            linewidth=1.8)
    end

    savefig(p, out_path)
end

function main()
    length(ARGS) >= 2 || error("""
    Usage: julia --project=StationSelection.jl scripts/plot_cost_cdfs_by_scenario.jl \\
               <exp_dir> <full_od_orders_csv> [output_dir] [robust_quantile|all]
    """)
    exp_dir     = normpath(ARGS[1])
    orders_path = normpath(ARGS[2])
    out_dir     = length(ARGS) >= 3 ? normpath(ARGS[3]) :
                  joinpath(exp_dir, "analysis", "cdfs")
    robust_quantile = if length(ARGS) >= 4 && ARGS[4] != "all"
        parse(Float64, ARGS[4])
    else
        nothing
    end
    mkpath(out_dir)

    println("Loading May orders from $orders_path ...")
    orders_df = CSV.read(orders_path, DataFrame)
    orders_df.order_time = DateTime.(orders_df.order_time, "yyyy-mm-dd HH:MM:SS")
    may_orders = filter(r -> month(r.order_time) == 5, orders_df)
    println("  $(nrow(may_orders)) May orders")
    isnothing(robust_quantile) || println("  Filtering robust to q=$(robust_quantile)")

    for k_val in [10, 15, 20]
        println("Computing costs for k=$(k_val) ...")
        entries = collect_costs(exp_dir, may_orders, k_val; robust_quantile)
        isempty(entries) && continue

        for period in ("period_1", "period_2", "period_3", "period_4")
            fname = joinpath(out_dir, "cdf_k$(k_val)_$(period).png")
            plot_scenario_cdfs(entries, period, k_val, fname)
            println("  Saved $fname")
        end
    end
end

main()
