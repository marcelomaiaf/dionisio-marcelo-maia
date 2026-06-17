#!/usr/bin/env bash
set -euo pipefail

export DIONISIO_NEO4J_HTTP_PORT="${DIONISIO_NEO4J_HTTP_PORT:-7475}"
export DIONISIO_PROXY_PORT="${PORT:-7474}"
export NEO4J_server_http_listen__address="127.0.0.1:${DIONISIO_NEO4J_HTTP_PORT}"

neo4j_pid=""

shutdown() {
  if [[ -n "${neo4j_pid}" ]] && kill -0 "${neo4j_pid}" 2>/dev/null; then
    kill "${neo4j_pid}" 2>/dev/null || true
    wait "${neo4j_pid}" 2>/dev/null || true
  fi
}

trap shutdown TERM INT

/startup/docker-entrypoint.sh "$@" &
neo4j_pid="$!"

python3 /opt/dionisio/browser/proxy.py &
proxy_pid="$!"

wait -n "${neo4j_pid}" "${proxy_pid}"
exit_code="$?"
shutdown
exit "${exit_code}"
