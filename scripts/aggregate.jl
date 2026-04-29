#!/usr/bin/env julia
"""
Aggregate results from an experiment into summary tables.

Usage:
    julia scripts/aggregate.jl experiments/<experiment_name>

Reads all metrics.json files under <experiment_name>/runs/,
produces a summary CSV and prints key statistics.
"""

using JSON, DataFrames, CSV, Statistics

if length(ARGS) < 1
    error("Usage: julia scripts/aggregate.jl experiments/<experiment_name>")
end

EXP_DIR = abspath(ARGS[1])

println("="^80)
println("Aggregating results: $EXP_DIR")

# Find all metrics files
metrics_files = String[]
for (root, dirs, files) in walkdir(joinpath(EXP_DIR, "runs"))
    for f in files
        f == "metrics.json" && push!(metrics_files, joinpath(root, f))
    end
end

println("Found $(length(metrics_files)) runs")
println("="^80)

if isempty(metrics_files)
    println("No runs found. Run 03_run_job.jl first.")
    exit(1)
end

# Load all metrics
rows = []
for f in metrics_files
    row = JSON.parsefile(f)
    for field in ("out_of_sample", "direct_backtest", "simulation_backtest")
        if haskey(row, field) && row[field] isa Dict
            for (k, v) in row[field]
                if !(v isa Dict || v isa Vector)
                    row["$(field)_$(k)"] = v
                end
            end
        end
    end
    if haskey(row, "out_of_sample") && row["out_of_sample"] isa Dict
        for (k, v) in row["out_of_sample"]
            row["oos_$(k)"] = v
        end
        delete!(row, "out_of_sample")
    end
    push!(rows, row)
end

# Build DataFrame (only scalar fields)
scalar_rows = [Dict(k => v for (k, v) in r if !(v isa Dict || v isa Vector)) for r in rows]
df = DataFrame(scalar_rows)

# Summary tables directory
tables_dir = joinpath(EXP_DIR, "analysis")
mkpath(tables_dir)

csv_path = joinpath(tables_dir, "results.csv")
CSV.write(csv_path, df)
println("\n✓ Saved: $csv_path")

# Per-model summary
println("\n" * "="^80)
println("In-sample objective by model type")
println("="^80)
if hasproperty(df, :model_type) && hasproperty(df, :in_sample_objective)
    for grp in groupby(df, :model_type)
        mt = grp[1, :model_type]
        vals = filter(!ismissing, grp[!, :in_sample_objective])
        isempty(vals) && continue
        println("  $mt:")
        println("    n       = $(length(vals))")
        println("    mean    = $(round(mean(vals); digits=2))")
        println("    min     = $(round(minimum(vals); digits=2))")
        println("    max     = $(round(maximum(vals); digits=2))")
    end
end

# Side-by-side nominal vs robust comparison by shared config
println("\n" * "="^80)
println("Nominal vs Robust")
println("="^80)
if hasproperty(df, :model_type) && hasproperty(df, :in_sample_objective)
    nominal_rows = df[df[!, :model_type] .== "NominalModel", :]
    robust_rows  = df[df[!, :model_type] .== "RobustTotalDemandCapModel", :]
    candidate_join_cols = [
        :k, Symbol("in_vehicle_time_weight"), :max_walking_distance,
        :Q_cap_quantile, :q_high_quantile,
    ]
    join_cols = filter(c -> hasproperty(df, c), candidate_join_cols)
    if !isempty(nominal_rows) && !isempty(robust_rows) && length(join_cols) >= 3
        nominal_small = select(nominal_rows, vcat(join_cols, [:in_sample_objective]))
        robust_small = select(robust_rows, vcat(join_cols, [:in_sample_objective]))
        rename!(nominal_small, :in_sample_objective => :nominal_objective)
        rename!(robust_small, :in_sample_objective => :robust_objective)
        paired = innerjoin(nominal_small, robust_small, on=join_cols, makeunique=true)
        if !isempty(paired)
            paired.price_of_robustness_pct =
                100 .* (paired.robust_objective .- paired.nominal_objective) ./ paired.nominal_objective
            paired_path = joinpath(tables_dir, "nominal_vs_robust.csv")
            CSV.write(paired_path, paired)
            println("  Saved side-by-side table: $paired_path")
            println("  Mean price of robustness: $(round(mean(paired.price_of_robustness_pct); digits=2))%")
        else
            println("  No matched nominal/robust pairs found")
        end
    else
        println("  (Need both NominalModel and RobustTotalDemandCapModel runs to compare)")
    end
end

# Out-of-sample summary (when available)
oos_cols = [c for c in names(df) if startswith(string(c), "oos_") && c != "oos_note"]
if !isempty(oos_cols)
    println("\n" * "="^80)
    println("Out-of-sample performance by model type")
    println("="^80)
    if hasproperty(df, :model_type)
        for grp in groupby(df, :model_type)
            mt = grp[1, :model_type]
            println("  $mt:")
            for col in oos_cols
                vals = filter(!ismissing, grp[!, col])
                isempty(vals) && continue
                lbl = replace(string(col), "oos_" => "")
                println("    $(rpad(lbl, 12)) mean=$(round(mean(vals); digits=2))  " *
                        "p90=$(round(quantile(vals, 0.9); digits=2))  " *
                        "worst=$(round(maximum(vals); digits=2))")
            end
        end
    end
end

println("\n" * "="^80)
println("Aggregation complete → $tables_dir")
println("="^80)
