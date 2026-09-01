#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"
dsh_select_python

DATA_ROOT="${DATA_ROOT:-${HOME}/data/dsh-evolution-v2}"
SCENARIO_FILE="${SCENARIO_FILE:-${DSH_REPO_ROOT}/examples/dsh/evolution_scenarios_v2.jsonl}"
FIXTURE_ROOT="${FIXTURE_ROOT:-${DSH_REPO_ROOT}}"
ENVIRONMENT_DIGEST="${ENVIRONMENT_DIGEST:-${DSH_ENVIRONMENT_DIGEST:-}}"
VERIFIER_CODE_DIGEST="${VERIFIER_CODE_DIGEST:-sha256:$(sha256sum "${DSH_REPO_ROOT}/examples/dsh/evolution_verifier.py" | cut -d' ' -f1)}"

if [[ -z "${ENVIRONMENT_DIGEST}" ]]; then
  echo "set ENVIRONMENT_DIGEST (the pinned DSH runtime digest) before building data" >&2
  exit 2
fi
dsh_require_file "${SCENARIO_FILE}"
dsh_require_file "${DSH_REPO_ROOT}/examples/dsh/evolution_verifier.py"
mkdir -p "${DATA_ROOT}"

"${PYTHON_BIN}" -m examples.dsh.prepare_evolution_dataset \
  --scenario-file "${SCENARIO_FILE}" \
  --fixture-root "${FIXTURE_ROOT}" \
  --output-dir "${DATA_ROOT}" \
  --environment-digest "${ENVIRONMENT_DIGEST}" \
  --verifier-code-digest "${VERIFIER_CODE_DIGEST}" \
  --patch examples/dsh/evolution.patch.yml

"${PYTHON_BIN}" "${SCRIPT_DIR}/write_dataset_manifest.py" \
  --repo-root "${DSH_REPO_ROOT}" \
  --scenario-file "${SCENARIO_FILE}" \
  --output-dir "${DATA_ROOT}" \
  --environment-digest "${ENVIRONMENT_DIGEST}" \
  --verifier-code-digest "${VERIFIER_CODE_DIGEST}"
