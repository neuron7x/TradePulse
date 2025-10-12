#!/bin/sh
set -eu

APP_USER=${APP_USER:-nfpro}
APP_GROUP=${APP_GROUP:-nfpro}
APP_UID=${APP_UID:-1000}
APP_GID=${APP_GID:-1000}
READ_ONLY_ROOT=${READ_ONLY_ROOT:-1}

is_root_ro() {
    awk '$2=="/" {if ($4 ~ /(^|,)ro(,|$)/) exit 0} END {exit 1}' /proc/mounts
}

if [ "$(id -u)" -eq 0 ]; then
    if [ "${READ_ONLY_ROOT}" = "1" ]; then
        if ! is_root_ro; then
            echo "[entrypoint] root filesystem must be mounted read-only; run the container with --read-only" >&2
            exit 70
        fi
    fi

    if [ -d /var/lib/nfpro ]; then
        chown "${APP_UID}:${APP_GID}" /var/lib/nfpro 2>/dev/null || true
    fi

    exec setpriv \
        --reuid="${APP_UID}" \
        --regid="${APP_GID}" \
        --init-groups \
        --no-new-privs \
        --bounding-set=-all \
        "$@"
fi

exec "$@"
