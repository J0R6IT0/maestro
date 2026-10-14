#!/bin/sh

set -u

if [ "$(id -u)" = 0 ]; then
    chown maestro: /data
    gids=$(stat -c %g /dev/dri/card* /dev/dri/renderD* 2>/dev/null | sort -un | paste -sd, -)
    exec setpriv --reuid=maestro --regid=maestro --groups="${gids:-maestro}" "$0" "$@"
fi

port="${LISTEN_ADDR##*:}"
crontab=/tmp/crontab

echo "${ANALYZE_CRON:-0 2 * * *} analyzer" >"$crontab"

supercronic -no-reap -test "$crontab" || exit 1

server &
server_pid=$!

(
    trap 'kill -TERM "$child" 2>/dev/null; exit 0' TERM
    child=
    until curl -so /dev/null "http://127.0.0.1:$port/similar"; do
        sleep 1 & child=$!
        wait "$child"
    done
    analyzer & child=$!
    wait "$child"
    exec supercronic -no-reap -passthrough-logs "$crontab"
) &
loop_pid=$!

trap 'kill -TERM "$server_pid" 2>/dev/null' TERM INT
wait "$server_pid"
status=$?
if kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    status=$?
fi
kill -TERM "$loop_pid" 2>/dev/null
pkill -TERM -x analyzer
wait "$loop_pid"
exit "$status"
