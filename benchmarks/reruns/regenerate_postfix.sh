#!/bin/bash
# Regenerate every paper figure with the bug-fixed solver (fixes 1, 3, 4 on; band stress off).
# Pre-fix data and figures are archived in outputs/archive_prefix_2026-11/.
# One job at a time, low priority, 3 threads (the machine is shared). Resumable: each stage
# writes a marker in outputs/postfix_2026-11/done/ and is skipped when re-launched.
# Not regenerated: the wall-clock speedup / work-precision timings (need a quiet machine).
cd "$(dirname "$0")/../.." || exit 1
export NUMBA_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export RERUN_DIR=postfix_2026-11
LOG=outputs/postfix_2026-11/regen.log; DONE=outputs/postfix_2026-11/done
mkdir -p "$DONE"
stage () {  # stage <name> <command...>
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then echo "$(date +%H:%M) skip $name" >> "$LOG"; return; fi
  echo "$(date +%H:%M) START $name" >> "$LOG"
  if nice -n 10 "$@" >> "$LOG" 2>&1; then
    touch "$DONE/$name"; echo "$(date +%H:%M) DONE $name" >> "$LOG"
  else
    echo "$(date +%H:%M) FAILED $name (exit $?)" >> "$LOG"
  fi
}
# caches the generators would otherwise reuse (archived copies are kept)
for f in contact_conv contact_collision contact_collision_snaps; do
  [ -f "$DONE/contact_convergence" ] || rm -f outputs/overlay/$f.npz
done

stage energy_benchmark   python3 -u benchmarks/disc_tg_energy_figure.py
stage convergence_sweep  python3 -u benchmarks/reruns/convergence_after_fix.py final
stage convergence_figure python3 -u benchmarks/bulk_convergence.py --from-conv=outputs/rerun_2026-11/conv_final.npz
stage sls_energy         python3 -u benchmarks/sls_energy_figure.py
stage ve_snapshots       python3 -u benchmarks/ve_snapshots_figure.py
stage drift              python3 -u benchmarks/drift_diagnostics.py
stage temporal           python3 -u benchmarks/temporal_convergence.py
stage lid_tauinf         python3 -u benchmarks/lid_stab_and_tauinf.py
stage lid_overlay        python3 -u -c "import sys; sys.path.insert(0,'benchmarks'); sys.path.insert(0,'.'); from benchmarks.overlay_integrators import soft_disc_overlay; print(soft_disc_overlay())"
stage lid_overlay_copy   cp outputs/overlay/soft_disc_overlay.png paper/figures/soft_disc_overlay.png
stage lid_snapshots      bash -c "cd benchmarks && python3 -u soft_disc_lid_snapshots.py"
stage contact_convergence python3 -u benchmarks/contact_convergence.py
stage contact_figure     python3 -u benchmarks/contact_figure.py
stage shear_fourroll     python3 -u benchmarks/reruns/shear_and_fourroll.py
stage shear_long         python3 -u benchmarks/reruns/shear_long_highWi.py
stage fourroll_quarter   python3 -u benchmarks/reruns/fourroll_elastic_quarter_dt.py
stage shear_figure       python3 -u benchmarks/disc_in_shear_figure.py
stage fourroll_figure    bash -c "cd benchmarks && python3 -u ve_extension_figure.py"
echo "$(date +%H:%M) REGENERATION FINISHED" >> "$LOG"
