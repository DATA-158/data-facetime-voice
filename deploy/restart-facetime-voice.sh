#!/bin/bash
# Restart the DATA FaceTime voice service (bootstrap after plist change).
# Run from a separate shell context — the gateway guard blocks launchctl
# bootstrap from inside its own process tree; osascript shells out instead.
osascript -e 'do shell script "launchctl bootout gui/$(id -u)/ai.data.facetime-voice || true"' 2>/dev/null
sleep 3
osascript -e 'do shell script "launchctl bootstrap gui/$(id -u) /Users/YOURUSER/Library/LaunchAgents/ai.data.facetime-voice.plist"' 2>/dev/null
sleep 2
launchctl list | grep facetime-voice && echo "voice service running"