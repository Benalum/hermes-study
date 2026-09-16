#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.hermes.study.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/data/logs"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.hermes.study</string>
<key>ProgramArguments</key><array><string>${ROOT}/scripts/run_mac.sh</string></array>
<key>WorkingDirectory</key><string>${ROOT}</string>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
<key>StandardOutPath</key><string>${ROOT}/data/logs/hermes-study.out.log</string>
<key>StandardErrorPath</key><string>${ROOT}/data/logs/hermes-study.err.log</string>
</dict></plist>
PLIST
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed LaunchAgent: $PLIST"
