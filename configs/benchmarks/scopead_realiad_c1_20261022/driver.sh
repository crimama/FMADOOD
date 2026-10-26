#!/usr/bin/env bash
# Real-IAD C1 ScopeAD seed0 x shots{1,2,4,8,16}: wait for download -> verify -> prepare -> copy -> run.
set -uo pipefail
cd /home/hun/Volume/RESEARCH/FSAD-TTA
SSHW=/tmp/fsad_dsba3_ssh_cleanup.sh
RELEASE=/workspace/.local-sync/releases/9a588d926e8736fd6f3fc4282bd5ac81010660191549ee75ef7b573c9845e8d8
LOG=results/experiment_launch/scopead_realiad_c1_20261022
mkdir -p "$LOG"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG/driver.log"; }
rs() { bash "$SSHW" "$1"; }
say "driver start; waiting for download process"
while [[ "$(rs 'pgrep -f [d]ownload_realiad_1024 | wc -l' | tail -1)" != "0" ]]; do sleep 120; done
say "download process ended: $(rs 'du -sh /media/hunim/realiad | cut -f1')"
B64=$(base64 -w0 scripts/download_realiad_1024.sh)
rs "echo $B64 | base64 -d > /media/hunim/realiad/download_realiad_1024.sh"
say "verification pass (hash + resume)"
rs "cd /media/hunim/realiad && JOBS=4 bash ./download_realiad_1024.sh > _logs/download_verify.log 2>&1; tail -2 _logs/download_verify.log; grep -c '^OK' _logs/download_verify.log; grep -c MISMATCH _logs/download_verify.log" | tee -a "$LOG/driver.log"
rs "grep -q DOWNLOAD_COMPLETE /media/hunim/realiad/_logs/download_verify.log && ! grep -q MISMATCH /media/hunim/realiad/_logs/download_verify.log" || { say "ABORT: verification failed"; exit 1; }
say "extract + build C1 tree"
rs "cd /media/hunim/realiad && python3 prepare_realiad_c1.py --out /media/hunim/realiad/c1_tree --workers 6" | tee -a "$LOG/driver.log"
rs "cd /media/hunim/realiad && python3 -c \"import json;t=json.load(open('c1_tree/_prepare_summary.json'))['totals'];print(t);assert t=={'train_good':10408,'test_good':10387,'test_bad':9415,'masks':9415},t\"" || { say "ABORT: unexpected counts"; exit 1; }
say "copy tree into existing container /var/tmp/realiad_c1"
rs "docker exec hun_fsad_tta_012 test ! -e /var/tmp/realiad_c1 && docker cp /media/hunim/realiad/c1_tree hun_fsad_tta_012:/var/tmp/realiad_c1 && docker exec hun_fsad_tta_012 bash -lc 'ls /var/tmp/realiad_c1 | wc -l; du -sh /var/tmp/realiad_c1'" | tee -a "$LOG/driver.log"
say "launch experiments"
set -a; source configs/remote/dsba3.env; set +a
export REMOTE_WORKDIR=$RELEASE
export REMOTE_PYTHON_HOME=/workspace/.local-runtime/python311-cu124-036828932804
gpus=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
export REMOTE_GPU_DEVICES="$(IFS=,; echo "${gpus[*]}")"
cfg=configs/benchmarks/scopead_realiad_c1_20261022
labels=(realiad_c1_scopead_s01 realiad_c1_scopead_s02 realiad_c1_scopead_s04 realiad_c1_scopead_s08 realiad_c1_scopead_s16)
configs=(); for l in "${labels[@]}"; do configs+=("$cfg/$l.yaml"); done
objects=$(python3 - <<'PY'
import ast
src=open('src/fmad/datasets/realiad.py').read()
tree=ast.parse(src)
for n in tree.body:
    if isinstance(n,ast.Assign) and n.targets[0].id=='_OBJECTS': print(' '.join(ast.literal_eval(n.value)))
PY
)
bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012 \
  --configs "${configs[@]}" --labels "${labels[@]}" --objects $objects --devices "${gpus[@]}" \
  --data-root /var/tmp/realiad_c1 --output-root /workspace/results_local_compatible/scopead_realiad_c1_20261022 \
  --log-dir "$LOG" --split realiad_c1_test_seed0_final_scopead --timeout 43200 --workers 1 2>&1 | tee -a "$LOG/launch.out"
say "launcher exit: $?"
