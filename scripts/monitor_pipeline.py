#!/usr/bin/env python3
"""
Pipeline monitor and auto-resubmitter for SLURM parameter sweeps.

Monitors runs/*/metrics.json and SLURM queue state, submitting new jobs
while staying within the server queue limit.  TIMEOUT jobs are always
permanently skipped (no_restart) — retrying with the same time limit
would always timeout again.  Other transient failures (FAILED, NODE_FAIL,
PREEMPTED, BOOT_FAIL) are retried up to --max-retries times.

Usage (run from project root, leave idle in a terminal or tmux):
    python scripts/monitor_pipeline.py <exp_dir> [options]

Examples:
    python scripts/monitor_pipeline.py experiments/2026-05-01_nominal_vs_robust
    python scripts/monitor_pipeline.py experiments/2026-05-01_nominal_vs_robust --interval 60
    python scripts/monitor_pipeline.py experiments/2026-05-01_nominal_vs_robust --max-queue 350 --dry-run
    python scripts/monitor_pipeline.py experiments/2026-05-01_nominal_vs_robust --once
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# SLURM final-state strings returned by sacct
SLURM_SUCCESS_STATES  = {"COMPLETED"}
SLURM_FAILURE_STATES  = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                          "PREEMPTED", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}
SLURM_TERMINAL_STATES = SLURM_SUCCESS_STATES | SLURM_FAILURE_STATES

# Permanently unrecoverable – never retry regardless of retry count.
# DEADLINE/CANCELLED: hit an external wall or was manually killed.
# OUT_OF_MEMORY is handled separately: resubmit with doubled memory up to 128G.
SLURM_NO_RESTART_STATES = {"DEADLINE", "CANCELLED"}

# Maximum memory (GB) for OOM escalation retries.
_OOM_MEM_CAP_GB = 128

# Maximum time (minutes) for TIMEOUT escalation retries.
_TIMEOUT_CAP_MINUTES = 60

# Transient hardware/scheduler issues – retry up to max_retries.
SLURM_RETRY_WITH_LIMIT_STATES = {"FAILED", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}

# Hours after submission before "no sacct record" is treated as failure
_GHOST_JOB_HOURS = 6.0


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

class SlurmInterface:
    """Thin wrappers around squeue / sacct / sbatch."""

    _queue_cache: dict | None = None

    def refresh_queue(self, user: str | None = None) -> dict:
        """
        Run squeue once and cache for this poll cycle.
        Returns {full_job_id -> state_code}.
        Array tasks appear as "12345_3"; non-array jobs as "12345".
        """
        cmd = ["squeue", "--format=%i %t", "--noheader"]
        if user:
            cmd += ["-u", user]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                print(f"  [warn] squeue: {r.stderr.strip()}")
                return {}
            jobs: dict = {}
            for line in r.stdout.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    raw_id, state = parts[0], parts[1]
                    # Expand compact array notation "12345_[1-3]" -> individual entries
                    if "_[" in raw_id:
                        base, rng = raw_id.split("_[", 1)
                        rng = rng.rstrip("]")
                        for tid in _expand_range(rng):
                            jobs[f"{base}_{tid}"] = state
                    else:
                        jobs[raw_id] = state
            self._queue_cache = jobs
            return jobs
        except subprocess.TimeoutExpired:
            print("  [warn] squeue timed out")
            return self._queue_cache or {}
        except FileNotFoundError:
            print("  [warn] squeue not found – are you on a SLURM node?")
            return {}

    def is_in_queue(self, array_job_id: str, task_id: int) -> bool:
        if self._queue_cache is None:
            return False
        return f"{array_job_id}_{task_id}" in self._queue_cache

    def sacct_state(self, array_job_id: str, task_id: int) -> str | None:
        """Return the sacct terminal state for array_job_id_task_id, or None.

        SLURM reports OOM kills as FAILED on the main task row; the actual
        OUT_OF_MEMORY state only appears on the srun substep (<id>.0).
        If the main task is FAILED but any substep is OUT_OF_MEMORY, we
        return OUT_OF_MEMORY so the caller can escalate memory correctly.
        """
        full = f"{array_job_id}_{task_id}"
        cmd = ["sacct", "-j", full, "--format=JobID,State",
               "--noheader", "--parsable2"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return None
            main_state = None
            substep_oom = False
            for line in r.stdout.strip().splitlines():
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                job_field = parts[0].strip()
                state     = parts[1].strip().split()[0]
                if job_field == full:
                    main_state = state
                elif job_field.startswith(full + ".") and state == "OUT_OF_MEMORY":
                    substep_oom = True
            if main_state == "FAILED" and substep_oom:
                return "OUT_OF_MEMORY"
            return main_state
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def submit(self, script: Path, task_ids: list, project_root: Path,
               dry_run: bool = False, mem_gb: int | None = None,
               time_minutes: int | None = None) -> tuple:
        """
        sbatch --parsable --array=<spec> <script>  (cwd = project_root).
        Returns (array_job_id, submitted_task_ids).
        mem_gb and time_minutes override the script's #SBATCH directives when set.
        """
        array_spec = _task_ids_to_spec(task_ids)
        cmd = ["sbatch", "--parsable", f"--array={array_spec}"]
        if mem_gb is not None:
            cmd.append(f"--mem={mem_gb}G")
        if time_minutes is not None:
            cmd.append(f"--time={_minutes_to_time_str(time_minutes)}")
        cmd.append(str(script))

        if dry_run:
            fake_id = f"DRY{int(time.time())}"
            print(f"  [dry-run] {' '.join(cmd)}")
            return fake_id, task_ids

        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=30, cwd=str(project_root))
            if r.returncode != 0:
                print(f"  [error] sbatch failed: {r.stderr.strip()}")
                return None, task_ids
            # --parsable output: "jobid" or "jobid;cluster"
            array_job_id = r.stdout.strip().split(";")[0].strip()
            return array_job_id, task_ids
        except subprocess.TimeoutExpired:
            print("  [error] sbatch timed out")
            return None, task_ids
        except FileNotFoundError:
            print("  [error] sbatch not found")
            return None, task_ids

    def clear_cache(self):
        self._queue_cache = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _expand_range(rng: str) -> list:
    """Parse "1-3,5,7-9" into [1,2,3,5,7,8,9]. Handles SLURM %throttle suffix."""
    result = []
    for part in rng.split(","):
        part = part.strip()
        if "%" in part:
            part = part[:part.index("%")]
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if not a or not b:
                continue
            try:
                result.extend(range(int(a), int(b) + 1))
            except ValueError:
                continue
        else:
            try:
                result.append(int(part))
            except ValueError:
                continue
    return result


def _task_ids_to_spec(ids: list) -> str:
    """Convert [1,2,3,5] -> '1-3,5' (compact SLURM array spec)."""
    if not ids:
        return ""
    sorted_ids = sorted(set(ids))
    parts = []
    start = prev = sorted_ids[0]
    for cur in sorted_ids[1:]:
        if cur == prev + 1:
            prev = cur
        else:
            parts.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = cur
    parts.append(f"{start}-{prev}" if prev != start else str(start))
    return ",".join(parts)


def _parse_time_minutes(s: str) -> int:
    """Parse a SLURM time string (HH:MM:SS or MM:SS or MM) into whole minutes."""
    parts = s.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 2:
            return int(parts[0])
        return int(parts[0])
    except ValueError:
        return 20  # fallback


def _minutes_to_time_str(minutes: int) -> str:
    """Format whole minutes as HH:MM:00 for sbatch --time."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


# ---------------------------------------------------------------------------
# Persistent monitor state
# ---------------------------------------------------------------------------

class MonitorState:
    """
    Stored in <exp_dir>/monitor_state.json.

    Structure::
        {
          "jobs": {
            "1": {"status": "pending", "array_job_id": null,
                  "task_id": null, "submitted_at": null, "retry_count": 0},
            ...
          }
        }

    Status values: pending | submitted | completed | failed | no_restart
    """

    def __init__(self, exp_dir: Path):
        self._path = exp_dir / "monitor_state.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"jobs": {}}

    def save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _entry(self, job_id: int) -> dict:
        key = str(job_id)
        jobs = self._data.setdefault("jobs", {})
        if key not in jobs:
            jobs[key] = {
                "status": "pending",
                "array_job_id": None,
                "task_id": None,
                "submitted_at": None,
                "retry_count": 0,
            }
        return jobs[key]

    def status(self, job_id: int) -> str:
        return self._entry(job_id)["status"]

    def retry_count(self, job_id: int) -> int:
        return self._entry(job_id).get("retry_count", 0)

    def update(self, job_id: int, **kwargs):
        self._entry(job_id).update(kwargs)

    def mark_submitted(self, job_id: int, array_job_id: str, task_id: int):
        self.update(job_id,
                    status="submitted",
                    array_job_id=array_job_id,
                    task_id=task_id,
                    submitted_at=time.time())

    def mark_completed(self, job_id: int):
        self.update(job_id, status="completed")

    def mark_failed(self, job_id: int) -> int:
        """Increment retry count, mark failed, return new count."""
        entry = self._entry(job_id)
        entry["retry_count"] = entry.get("retry_count", 0) + 1
        entry["status"] = "failed"
        return entry["retry_count"]

    def mark_no_restart(self, job_id: int, reason: str = "",
                        integrality_gap: str | None = None):
        kwargs: dict = {"status": "no_restart", "no_restart_reason": reason}
        if integrality_gap is not None:
            kwargs["integrality_gap"] = integrality_gap
        self.update(job_id, **kwargs)

    def mem_gb(self, job_id: int) -> int | None:
        """Return the escalated memory (GB) for this job, or None if using base."""
        return self._entry(job_id).get("mem_gb")

    def mark_oom_retry(self, job_id: int, new_mem_gb: int):
        """Record the new memory level and mark failed for resubmission."""
        entry = self._entry(job_id)
        entry["retry_count"] = entry.get("retry_count", 0) + 1
        entry["status"] = "failed"
        entry["mem_gb"] = new_mem_gb

    def time_limit_min(self, job_id: int) -> int | None:
        """Return the escalated time limit (minutes) for this job, or None if using base."""
        return self._entry(job_id).get("time_limit_min")

    def mark_timeout_retry(self, job_id: int, new_time_min: int):
        """Record the new time limit and mark failed for resubmission."""
        entry = self._entry(job_id)
        entry["retry_count"] = entry.get("retry_count", 0) + 1
        entry["status"] = "failed"
        entry["time_limit_min"] = new_time_min


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class PipelineMonitor:

    def __init__(self, exp_dir: Path, project_root: Path, args: argparse.Namespace):
        self.exp_dir      = exp_dir
        self.project_root = project_root
        self.interval     = args.interval
        self.max_queue    = args.max_queue
        self.max_retries  = args.max_retries
        self.dry_run      = args.dry_run
        self.user         = os.environ.get("USER")

        self.slurm = SlurmInterface()
        self.state = MonitorState(exp_dir)

        meta_path = exp_dir / "setup_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"setup_metadata.json not found at {meta_path}.\n"
                "Run 01_setup.jl first."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        self.total_jobs = meta["total_jobs"]

        self.submit_script = exp_dir / "02_submit.sh"
        if not self.submit_script.exists():
            raise FileNotFoundError(f"Submit script not found: {self.submit_script}")

        # Seed completed state from any existing metrics.json files
        self._seed_from_metrics()
        self.base_mem_gb   = self._parse_base_mem_gb()
        self.base_time_min = self._parse_base_time_minutes()

    # ------------------------------------------------------------------
    # Seeding from on-disk artifacts
    # ------------------------------------------------------------------

    def _seed_from_metrics(self):
        """Mark jobs completed in monitor state if they have a metrics.json."""
        runs_dir = self.exp_dir / "runs"
        if not runs_dir.exists():
            return
        for mf in runs_dir.glob("*/metrics.json"):
            try:
                m = json.loads(mf.read_text())
                job_id = m.get("job_id")
                if job_id is not None:
                    self.state.mark_completed(int(job_id))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SLURM log helpers
    # ------------------------------------------------------------------

    def _slurm_log_path(self, array_job_id: str, task_id: int) -> Path:
        # #SBATCH -o experiments/<exp>/slurm_logs/job-%A_%a.out, submitted from project_root
        return self.exp_dir / "slurm_logs" / f"job-{array_job_id}_{task_id}.out"

    def _gurobi_was_solving(self, log_path: Path) -> bool:
        """
        Return True if Gurobi started the MIP solve but did not finish
        (job was killed mid-solve, integrality gap still open).

        Strategy: find the LAST "Nodes    |" (main solve header) and check
        whether a post-solve summary appears after it.
        Timed-out mid-solve => started AND NOT finished.
        """
        if not log_path.exists():
            return False
        try:
            content = log_path.read_text(errors="replace")
        except OSError:
            return False

        main_solve_pos = content.rfind("Nodes    |")
        if main_solve_pos == -1:
            return False  # Gurobi never started the main MIP solve

        after_main = content[main_solve_pos:]
        finished = (
            "Explored " in after_main
            or "Model is infeasible" in after_main
            or "Infeasible model" in after_main
        )
        return not finished

    def _parse_integrality_gap(self, log_path: Path) -> str | None:
        """Return the last reported integrality gap from a mid-solve timeout, e.g. '30.3%'."""
        if not log_path.exists():
            return None
        try:
            content = log_path.read_text(errors="replace")
        except OSError:
            return None
        main_pos = content.rfind("Nodes    |")
        if main_pos == -1:
            return None
        gap = None
        for line in content[main_pos:].splitlines():
            m = re.search(r'(\d+\.?\d*)%\s+[\d.]+\s+\d+s\s*$', line)
            if m:
                gap = m.group(1) + "%"
        return gap

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _parse_base_mem_gb(self) -> int:
        """Parse --mem=XG from the submit script header; default 64G."""
        try:
            content = self.submit_script.read_text()
            m = re.search(r'#SBATCH\s+--mem=(\d+)G', content)
            if m:
                return int(m.group(1))
        except OSError:
            pass
        return 64

    def _parse_base_time_minutes(self) -> int:
        """Parse --time=HH:MM:SS from the submit script header; default 30min."""
        try:
            content = self.submit_script.read_text()
            m = re.search(r'#SBATCH\s+--time=([0-9:]+)', content)
            if m:
                return _parse_time_minutes(m.group(1))
        except OSError:
            pass
        return 30

    def _handle_oom(self, job_id: int):
        """Double memory and resubmit, up to _OOM_MEM_CAP_GB; then no_restart."""
        current_mem = self.state.mem_gb(job_id) or self.base_mem_gb
        new_mem = current_mem * 2
        if new_mem > _OOM_MEM_CAP_GB:
            print(f"  [skip] job {job_id} OOM at {current_mem}G – cap ({_OOM_MEM_CAP_GB}G) "
                  f"reached, marking no_restart")
            self.state.mark_no_restart(job_id, reason=f"OOM+cap({current_mem}G)")
        else:
            print(f"  [retry] job {job_id} OOM at {current_mem}G – retrying with {new_mem}G")
            self.state.mark_oom_retry(job_id, new_mem)

    def _mark_failed_or_exhaust(self, job_id: int, slurm_state: str):
        count = self.state.mark_failed(job_id)
        if count >= self.max_retries:
            print(f"  [skip] job {job_id} has failed {count}x "
                  f"(last: {slurm_state}) – marking no_restart")
            self.state.mark_no_restart(job_id, reason=f"max_retries({slurm_state})")

    def _handle_timeout(self, job_id: int, array_job_id: str, task_id: int):
        """
        Double the time limit and resubmit, up to _TIMEOUT_CAP_MINUTES; then no_restart.
        The integrality gap and solve phase are recorded for diagnostics.
        """
        log_path = self._slurm_log_path(array_job_id, task_id)
        gap = self._parse_integrality_gap(log_path)
        gap_str = f", gap={gap}" if gap else ""
        mid_solve = self._gurobi_was_solving(log_path)
        solve_str = "mid-solve" if mid_solve else "pre/post-solve"

        current_time = self.state.time_limit_min(job_id) or self.base_time_min
        new_time = current_time * 2
        if new_time > _TIMEOUT_CAP_MINUTES:
            print(f"  [skip] job {job_id} timed out ({solve_str}{gap_str}, {current_time}min) "
                  f"– cap ({_TIMEOUT_CAP_MINUTES}min) reached, marking no_restart"
                  f"  (log: {log_path.name})")
            self.state.mark_no_restart(job_id, reason=f"TIMEOUT+cap({current_time}min)",
                                       integrality_gap=gap)
        else:
            print(f"  [retry] job {job_id} timed out ({solve_str}{gap_str}, {current_time}min) "
                  f"– retrying with {new_time}min  (log: {log_path.name})")
            self.state.mark_timeout_retry(job_id, new_time)

    # ------------------------------------------------------------------
    # State update (single job)
    # ------------------------------------------------------------------

    def _refresh_job(self, job_id: int):
        """
        Reconcile monitor state with metrics.json + SLURM queue.
        Mutates self.state in place.
        """
        s = self.state.status(job_id)

        # Terminal states need no further work
        if s in ("completed", "no_restart"):
            return

        # For submitted jobs: check SLURM to detect silent failures
        if s == "submitted":
            entry   = self.state._entry(job_id)
            ajid    = entry.get("array_job_id") or ""
            task_id = entry.get("task_id")
            sub_at  = entry.get("submitted_at") or 0

            if task_id is None or ajid.startswith("DRY"):
                return

            if self.slurm.is_in_queue(ajid, task_id):
                return  # Still running/pending – nothing to do

            # Job left queue – metrics.json is the most authoritative success signal
            # (_seed_from_metrics was already called at the top of run_once)
            if self.state.status(job_id) == "completed":
                return

            final = self.slurm.sacct_state(ajid, task_id)
            if final in SLURM_SUCCESS_STATES:
                # SLURM says OK but metrics.json not written yet – wait for next poll
                pass
            elif final in SLURM_NO_RESTART_STATES:
                print(f"  [skip] job {job_id} ended with {final} – marking no_restart")
                self.state.mark_no_restart(job_id, reason=final)
            elif final == "OUT_OF_MEMORY":
                self._handle_oom(job_id)
            elif final == "TIMEOUT":
                self._handle_timeout(job_id, ajid, task_id)
            elif final in SLURM_RETRY_WITH_LIMIT_STATES:
                self._mark_failed_or_exhaust(job_id, final)
            elif final in SLURM_FAILURE_STATES:
                # Any remaining failure states not explicitly categorised
                print(f"  [skip] job {job_id} ended with {final} – marking no_restart")
                self.state.mark_no_restart(job_id, reason=final)
            elif final is None:
                # Not in sacct yet; if job is very old, assume failure
                age_h = (time.time() - sub_at) / 3600
                if age_h > _GHOST_JOB_HOURS:
                    print(f"  [warn] job {job_id} missing from sacct "
                          f"after {age_h:.1f}h – marking failed")
                    self._mark_failed_or_exhaust(job_id, "ghost")

    # ------------------------------------------------------------------
    # Single poll cycle
    # ------------------------------------------------------------------

    def run_once(self) -> bool:
        """
        Refresh state, print summary, submit pending jobs.
        Returns True when the entire sweep is complete (or stalled).
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*70}")
        print(f"  Poll  {ts}")
        print(f"{'='*70}")

        # Refresh SLURM queue cache once for this cycle
        self.slurm.clear_cache()
        active = self.slurm.refresh_queue(user=self.user)
        queue_count = len(active)
        print(f"  Queue: {queue_count} active SLURM tasks  (limit: {self.max_queue})")

        # Sync completed jobs from metrics.json before checking queue state
        self._seed_from_metrics()

        for job_id in range(1, self.total_jobs + 1):
            self._refresh_job(job_id)

        # Build summary counts
        c: dict = {"pending": 0, "submitted": 0, "completed": 0,
                   "failed": 0, "no_restart": 0}
        for job_id in range(1, self.total_jobs + 1):
            c[self.state.status(job_id)] = c.get(self.state.status(job_id), 0) + 1

        print()
        _print_status_line("Sweep jobs", self.total_jobs, c)

        # Check available slots
        slots = self.max_queue - queue_count
        if slots <= 0:
            print(f"\n  Queue full – skipping submissions this cycle")
            if not self.dry_run:
                self.state.save()
            return self._all_done(c)

        pending = [jid for jid in range(1, self.total_jobs + 1)
                   if self.state.status(jid) in ("pending", "failed")]

        if not pending:
            print("\n  Nothing to submit.")
        else:
            print(f"\n  To submit: {len(pending)} pending/failed  |  {slots} slots available")

            # SLURM opens the log file before the script runs, so the dir must exist
            (self.exp_dir / "slurm_logs").mkdir(parents=True, exist_ok=True)

            # Group by (mem_gb, time_min) – escalated jobs need resource overrides.
            by_config: dict = defaultdict(list)
            for jid in pending:
                mem  = self.state.mem_gb(jid)       or self.base_mem_gb
                time = self.state.time_limit_min(jid) or self.base_time_min
                by_config[(mem, time)].append(jid)

            remaining_slots = slots
            total_submitted = 0
            for (mem, time), job_ids in sorted(by_config.items()):
                if remaining_slots <= 0:
                    break
                to_submit = job_ids[:remaining_slots]
                mem_override  = mem  if mem  != self.base_mem_gb   else None
                time_override = time if time != self.base_time_min else None
                array_job_id, submitted = self.slurm.submit(
                    self.submit_script, to_submit, self.project_root,
                    dry_run=self.dry_run, mem_gb=mem_override, time_minutes=time_override,
                )
                if array_job_id:
                    if not self.dry_run:
                        for tid in submitted:
                            self.state.mark_submitted(tid, array_job_id, tid)
                    preview = (str(submitted[:4])[:-1] + ", ...]"
                               if len(submitted) > 4 else str(submitted))
                    overrides = "".join([
                        f"  --mem={mem}G"           if mem_override  else "",
                        f"  --time={_minutes_to_time_str(time)}" if time_override else "",
                    ])
                    print(f"  -> submitted tasks {preview}  (array job {array_job_id}{overrides})")
                    total_submitted += len(submitted)
                remaining_slots -= len(to_submit)

            remaining = len(pending) - total_submitted
            if remaining > 0:
                print(f"  -> {remaining} more job(s) will be submitted on the next poll cycle")

        if not self.dry_run:
            self.state.save()
        return self._all_done(c)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_done(self, c: dict) -> bool:
        if (c.get("pending", 0) == 0 and c.get("submitted", 0) == 0
                and c.get("failed", 0) == 0):
            if c.get("completed", 0) == self.total_jobs:
                print("\n  *** ALL SWEEP JOBS COMPLETE ***")
            else:
                skipped = c.get("no_restart", 0)
                print(f"\n  *** PIPELINE STALLED – {skipped} jobs skipped, "
                      f"{c.get('completed', 0)}/{self.total_jobs} completed ***")
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print("=" * 70)
        print("  Pipeline Monitor")
        print(f"  Experiment: {self.exp_dir.name}")
        print(f"  Total jobs: {self.total_jobs}")
        print(f"  Interval:   {self.interval}s  |  Max queue: {self.max_queue}"
              f"  |  Max retries: {self.max_retries}")
        if self.dry_run:
            print("  DRY-RUN MODE – no jobs will actually be submitted")
        print("=" * 70)
        print("  Ctrl+C to stop")

        try:
            while True:
                if self.run_once():
                    break
                print(f"\n  Next poll in {self.interval}s ...")
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
            if not self.dry_run:
                self.state.save()


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def _print_status_line(label: str, total: int, c: dict):
    bar_width = 30
    n_done   = c.get("completed", 0)
    n_active = c.get("submitted", 0)
    n_fail   = c.get("failed", 0)
    n_pend   = c.get("pending", 0)
    n_skip   = c.get("no_restart", 0)
    filled   = int(bar_width * n_done / max(total, 1))
    bar      = "#" * filled + "-" * (bar_width - filled)
    skip_str = f"  skip={n_skip:3d}" if n_skip else ""
    print(f"  {label} [{bar}] {n_done:4d}/{total}"
          f"   run/q={n_active:4d}  fail={n_fail:3d}  pend={n_pend:4d}{skip_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("exp_dir",
                        help="Experiment directory (relative to cwd or absolute)")
    parser.add_argument("--interval", type=int, default=300,
                        help="Poll interval in seconds (default: 300)")
    parser.add_argument("--max-queue", type=int, default=380,
                        help="Maximum total SLURM tasks in queue at once (default: 380)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries for transient failures before no_restart (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without executing them")
    parser.add_argument("--once", action="store_true",
                        help="Run a single poll cycle and exit")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        print(f"ERROR: directory not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    # project root is 2 levels up: <project_root>/experiments/<exp_name>
    project_root = exp_dir.parent.parent

    try:
        monitor = PipelineMonitor(exp_dir, project_root, args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.once:
        monitor.run_once()
        if not args.dry_run:
            monitor.state.save()
    else:
        monitor.run()


if __name__ == "__main__":
    main()
