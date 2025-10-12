#!/bin/sh
set -eu

APP_USER=${APP_USER:-nfpro}
APP_GROUP=${APP_GROUP:-nfpro}
APP_UID=${APP_UID:-1000}
APP_GID=${APP_GID:-1000}
READ_ONLY_ROOT=${READ_ONLY_ROOT:-1}

if [ "$(id -u)" -eq 0 ]; then
    if [ "${READ_ONLY_ROOT}" = "1" ]; then
        if ! mountpoint -q /tmp; then
            mkdir -p /tmp
            mount -t tmpfs tmpfs /tmp
        fi
        if [ -d /var/lib/nfpro ] && ! mountpoint -q /var/lib/nfpro; then
            mount -t tmpfs tmpfs /var/lib/nfpro
        fi
        if ! mount | grep -E ' on / ' | grep -q '(ro'; then
            mount -o remount,ro /
        fi
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
