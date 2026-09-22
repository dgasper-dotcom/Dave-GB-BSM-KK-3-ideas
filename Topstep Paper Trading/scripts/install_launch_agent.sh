#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:a:h}
REPO_DIR=${SCRIPT_DIR:h}
LABEL="com.davidgasper.topstep.propchallenge.paper"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
OUT_DIR="${REPO_DIR}/topstep_prop_challenge_paper"

mkdir -p "${HOME}/Library/LaunchAgents" "${OUT_DIR}"
chmod +x "${REPO_DIR}/scripts/run_live_paper.sh"

cat > "${PLIST}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>${REPO_DIR}/scripts/run_live_paper.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${REPO_DIR}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${OUT_DIR}/launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>${OUT_DIR}/launchd.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
</dict>
</plist>
PLIST

UID_VALUE="$(id -u)"
launchctl bootout "gui/${UID_VALUE}" "${PLIST}" 2>/dev/null || true
launchctl bootstrap "gui/${UID_VALUE}" "${PLIST}"
launchctl kickstart -k "gui/${UID_VALUE}/${LABEL}"

echo "Installed and started ${LABEL}"
echo "Dashboard: http://127.0.0.1:8787"
