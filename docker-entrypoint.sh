#!/bin/sh
set -eu

APP_USER=${APP_USER:-nfpro}
APP_GROUP=${APP_GROUP:-nfpro}
APP_UID=${APP_UID:-1000}
APP_GID=${APP_GID:-1000}
READ_ONLY_ROOT=${READ_ONLY_ROOT:-1}

is_root_fs_read_only() {
    awk '$2=="/" {if ($4 ~ /(^|,)ro(,|$)/) exit 0} END {exit 1}' /proc/mounts
}

ensure_read_only_root() {
    if [ "${READ_ONLY_ROOT}" != "1" ]; then
        return 0
    fi

    if ! is_root_fs_read_only; then
        printf '%s\n' "[entrypoint] root filesystem must be mounted read-only; start the container with --read-only" >&2
        exit 70
    fi
}

ensure_read_only_root

if [ "$(id -u)" -eq 0 ]; then
    if ! command -v setpriv >/dev/null 2>&1; then
        printf '%s\n' "[entrypoint] setpriv binary is required to drop capabilities but was not found" >&2
        exit 127
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
