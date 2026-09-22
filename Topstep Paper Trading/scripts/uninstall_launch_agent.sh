#!/bin/zsh
set -euo pipefail

LABEL="com.davidgasper.topstep.propchallenge.paper"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_VALUE="$(id -u)"

launchctl bootout "gui/${UID_VALUE}" "${PLIST}" 2>/dev/null || true
rm -f "${PLIST}"

echo "Stopped and removed ${LABEL}"
