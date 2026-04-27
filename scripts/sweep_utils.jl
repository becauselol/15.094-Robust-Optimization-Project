#!/usr/bin/env julia
"""
Shared utilities for sweep experiments.
"""

module SweepUtils

using TOML, Dates, CSV, DataFrames, JSON

export generate_cartesian_product, save_job_list, get_git_info, save_metadata

"""
    generate_cartesian_product(sweep_params::Dict) -> Vector{Dict}

Generate all combinations (Cartesian product) of sweep parameters.
"""
function generate_cartesian_product(sweep_params::Dict)
    keys_vec = collect(keys(sweep_params))
    vals_vec = collect(values(sweep_params))
    combinations = vec(collect(Iterators.product(vals_vec...)))
    return [Dict(zip(keys_vec, combo)) for combo in combinations]
end

"""
    save_job_list(jobs::Vector{Dict}, config_dir::String)

Save job list to CSV and simple text format for SLURM array jobs.
"""
function save_job_list(jobs::Vector{Dict}, config_dir::String)
    mkpath(config_dir)

    jobs_df = DataFrame(jobs)

    csv_path = joinpath(config_dir, "job_list.csv")
    CSV.write(csv_path, jobs_df)
    println("✓ Saved job list: $csv_path")

    txt_path = joinpath(config_dir, "jobs.txt")
    open(txt_path, "w") do f
        for i in 1:nrow(jobs_df)
            println(f, i)
        end
    end
    println("✓ Saved SLURM job list: $txt_path")

    return jobs_df
end

"""
    get_git_info() -> Dict

Get current git repository information for provenance tracking.
"""
function get_git_info()
    info = Dict{String, String}()
    try
        info["commit"] = strip(read(`git rev-parse HEAD`, String))
        info["branch"] = strip(read(`git rev-parse --abbrev-ref HEAD`, String))
        info["status"] = strip(read(`git status --short`, String))
    catch e
        @warn "Could not retrieve git info: $e"
        info["error"] = string(e)
    end
    return info
end

"""
    save_metadata(study_dir::String, extra_info::Dict=Dict())

Save experiment metadata including environment, git info, and creation time.
"""
function save_metadata(study_dir::String, extra_info::Dict=Dict())
    metadata = Dict(
        "created" => string(now()),
        "julia_version" => string(VERSION),
        "hostname" => gethostname(),
        "user" => get(ENV, "USER", "unknown"),
        "git" => get_git_info(),
    )
    merge!(metadata, extra_info)

    metadata_path = joinpath(study_dir, "metadata.json")
    open(metadata_path, "w") do f
        JSON.print(f, metadata, 2)
    end
    println("✓ Saved metadata: $metadata_path")

    return metadata
end

end # module
