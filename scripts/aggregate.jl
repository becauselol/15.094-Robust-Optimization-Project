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
    # Flatten out-of-sample sub-dict
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

# Price of robustness: (robust_obj - nominal_obj) / nominal_obj
println("\n" * "="^80)
println("Price of Robustness")
println("="^80)
if hasproperty(df, :model_type) && hasproperty(df, :in_sample_objective)
    nominal_rows = df[df[!, :model_type] .== "NominalModel", :]
    robust_rows  = df[df[!, :model_type] .== "BudgetRobustModel", :]
    if !isempty(nominal_rows) && !isempty(robust_rows)
        nom_mean = mean(filter(!ismissing, nominal_rows[!, :in_sample_objective]))
        rob_mean = mean(filter(!ismissing, robust_rows[!, :in_sample_objective]))
        por = (rob_mean - nom_mean) / nom_mean * 100
        println("  Nominal mean objective:      $(round(nom_mean; digits=2))")
        println("  Budget-robust mean objective: $(round(rob_mean; digits=2))")
        println("  Price of robustness:         $(round(por; digits=2))%")
    else
        println("  (Need both NominalModel and BudgetRobustModel runs to compute PoR)")
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
