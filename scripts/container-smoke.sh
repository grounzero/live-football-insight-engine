#!/usr/bin/env bash
#
# Start the built image and drive the running service over HTTP.
#
# The point is to exercise the artifact that gets deployed, not the source tree.
# Everything below is done against a container started with no bind mounts, no
# dataset and a non-default port, because that is exactly the situation a
# hosting platform creates and the situation that used to fail.
#
# Usage:
#   scripts/container-smoke.sh
#   IMAGE=football-insights:ci SMOKE_PORT=8087 scripts/container-smoke.sh
#
# Shared verbatim between `make container-smoke` and CI, so the two cannot
# drift; CI is expected to print the logs this script dumps on failure.

set -euo pipefail

IMAGE="${IMAGE:-football-insights:deployment-test}"
# Deliberately not 8000: honouring PORT is one of the things under test, and a
# default port would pass whether or not it worked.
SMOKE_PORT="${SMOKE_PORT:-8087}"
CONTAINER="${CONTAINER:-fi-smoke-$$}"
BASE="http://127.0.0.1:${SMOKE_PORT}"

READY_TIMEOUT_S="${READY_TIMEOUT_S:-90}"
SSE_TIMEOUT_S="${SSE_TIMEOUT_S:-30}"

pass() { printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1" >&2; exit 1; }
step() { printf '\n== %s\n' "$1"; }

cleanup() {
  local status=$?
  if [ "${status}" -ne 0 ]; then
    printf '\n== container logs (exit %s)\n' "${status}" >&2
    docker logs "${CONTAINER}" 2>&1 | tail -n 200 >&2 || true
    printf '\n== container state\n' >&2
    docker inspect --format '{{json .State}}' "${CONTAINER}" 2>&1 >&2 || true
  fi
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  return "${status}"
}
trap cleanup EXIT

step "starting ${IMAGE} on port ${SMOKE_PORT}"
# No -v, no --network host, no dataset: if the image needs anything from the
# host to start, it fails here.
docker run -d \
  --name "${CONTAINER}" \
  -e PORT="${SMOKE_PORT}" \
  -p "127.0.0.1:${SMOKE_PORT}:${SMOKE_PORT}" \
  "${IMAGE}" >/dev/null
pass "container started"

step "waiting for /health"
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
until curl -fsS --max-time 3 "${BASE}/health" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    fail "the container exited before it answered"
  fi
  [ "$(date +%s)" -lt "${deadline}" ] || fail "/health did not answer within ${READY_TIMEOUT_S}s"
  sleep 1
done
pass "/health answered on the injected PORT"

step "readiness, before any stream client connects"
# Ordering matters: readiness must not depend on somebody watching, or a
# platform health check would gate on a browser that has not arrived yet.
ready_body="$(curl -fsS --max-time 5 "${BASE}/ready")" || fail "/ready did not return success"
printf '  %s\n' "${ready_body}"

python3 - "${ready_body}" <<'PY' || fail "/ready did not describe a usable public demo"
import json, sys

body = json.loads(sys.argv[1])
problems = []
if body.get("ready") is not True:
    problems.append(f"ready={body.get('ready')!r} reason={body.get('reason')!r}")
if body.get("mode") != "public_demo":
    problems.append(f"mode={body.get('mode')!r}")
if body.get("data_source") != "synthetic":
    problems.append(f"data_source={body.get('data_source')!r}")
if body.get("ui") is not True:
    problems.append("the built page is not being served")
if not isinstance(body.get("predictor"), dict):
    problems.append("no predictor reported")
# Whatever is loaded, it must say what it is. Both answers are acceptable;
# silence is not.
elif body["predictor"].get("is_ml") not in (True, False):
    problems.append("the predictor does not declare whether it is ML-backed")

for leaked in ("/opt/", "/app/", "fingerprint"):
    if leaked in sys.argv[1]:
        problems.append(f"the public payload leaks {leaked!r}")

if problems:
    print("; ".join(problems), file=sys.stderr)
    raise SystemExit(1)

predictor = body["predictor"]
kind = "ML-backed" if predictor["is_ml"] else "rule-based fallback"
print(f"  predictor: {predictor['name']} ({kind})")
PY
pass "/ready reports a usable public demo"

step "docker healthcheck on the injected port"
# The probe reads PORT in Python; an exec-form CMD would never expand a shell
# variable, so this is the check that the probe is looking where the service is.
deadline=$(( $(date +%s) + 60 ))
until [ "$(docker inspect --format '{{.State.Health.Status}}' "${CONTAINER}")" = "healthy" ]; do
  [ "$(date +%s)" -lt "${deadline}" ] || fail "the container never became healthy"
  sleep 2
done
pass "docker reports the container healthy"

step "one application process"
# Replay state, the subscriber set and the metric registry are all per-process,
# so a second worker would serve a second, divergent replay from one URL.
workers="$(docker top "${CONTAINER}" -o pid,args 2>/dev/null | grep -c 'football-insights' || true)"
[ "${workers}" -eq 1 ] || fail "expected exactly one application process, found ${workers}"
pass "exactly one worker"

step "the page and its assets"
index="$(curl -fsS --max-time 10 "${BASE}/")" || fail "/ did not serve the page"
printf '%s' "${index}" | grep -q '<div id="root"' || fail "/ served something that is not the demo"
pass "/ serves the built page"

asset="$(printf '%s' "${index}" | grep -oE '/assets/[A-Za-z0-9._-]+\.js' | head -n1 || true)"
[ -n "${asset}" ] || fail "the page references no built script"
curl -fsS --max-time 10 "${BASE}${asset}" >/dev/null || fail "the built asset ${asset} is not retrievable"
pass "built asset ${asset} is served"

step "the live insight stream"
# --no-buffer so the first event arrives as soon as it is written, --max-time so
# a stream that never produces one fails instead of hanging CI forever.
sse_out="$(mktemp)"
set +e
curl -fsS --no-buffer --max-time "${SSE_TIMEOUT_S}" "${BASE}/insights/stream" >"${sse_out}" 2>/dev/null
set -e

if [ ! -s "${sse_out}" ]; then
  rm -f "${sse_out}"
  fail "the stream produced nothing within ${SSE_TIMEOUT_S}s (timeout, not malformed output)"
fi

python3 - "${sse_out}" <<'PY' || { rm -f "${sse_out}"; fail "the stream produced output that is not valid SSE"; }
import json, sys

KNOWN = {"frame", "insight", "suppression", "restart", "match", "end"}
event = payload = None
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.rstrip("\n")
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            break

if event is None or payload is None:
    raise SystemExit("no complete `event:`/`data:` pair in the output")
if event != "update":
    raise SystemExit(f"unexpected SSE event name {event!r}")

message = json.loads(payload)          # malformed JSON fails loudly here
if message.get("type") not in KNOWN:
    raise SystemExit(f"unknown message type {message.get('type')!r}")
if "payload" not in message:
    raise SystemExit("message has no payload")

print(f"  first event: {message['type']}")
PY
rm -f "${sse_out}"
pass "the stream emits well-formed, parseable events"

step "the image itself"
docker run --rm "${IMAGE}" python -c "
from pathlib import Path
import football_insights
page = Path(football_insights.__file__).parent / 'serving' / 'static' / 'index.html'
assert page.is_file(), page
" >/dev/null || fail "the built page is not inside the installed package"
pass "the page ships inside the wheel, not beside it"

docker run --rm "${IMAGE}" sh -c '! command -v node && ! command -v npm' >/dev/null \
  || fail "the runtime image still contains Node"
pass "no Node in the runtime image"

docker run --rm "${IMAGE}" python -c "
import importlib.util as u
assert u.find_spec('torch') is None, 'torch is in the runtime image'
assert u.find_spec('onnxruntime') is not None, 'onnxruntime is missing'
" >/dev/null || fail "the runtime dependency set is wrong"
pass "no PyTorch; ONNX Runtime present"

docker run --rm "${IMAGE}" python -m pip check >/dev/null || fail "the installed dependency set is inconsistent"
pass "pip check clean"

printf '\n== all container smoke checks passed (%s)\n' "${IMAGE}"
