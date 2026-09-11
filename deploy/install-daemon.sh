#!/bin/bash
# Build, sign and install the facetime-bridge daemon at its PERMANENT path.
#
# macOS ties an Accessibility grant to the binary's code signature. An ad-hoc
# signed binary changes hash on every rebuild, so every rebuild used to need
# a new grant (ax, ax2, ax3, ax4, ax5...). Signing with the self-signed
# "DATA FaceTime Bridge" identity (login keychain) makes the grant follow the
# identity + bundle identifier instead: grant ~/.local/bin/facetime-bridge-daemon
# once in System Settings > Privacy & Security > Accessibility, then this
# script can be re-run forever.
#
# The launchd job ai.data.facetime-bridge must point at $DEST.
set -euo pipefail
BRIDGE="${BRIDGE_REPO:-$HOME/Documents/Programming/Datas_Projects/facetime-bridge}"
DEST="$HOME/.local/bin/facetime-bridge-daemon"
IDENTITY="DATA FaceTime Bridge"
cd "$BRIDGE/native"
swift build -c release 2>&1 | tail -1
BIN=.build/release/facetime-bridge
codesign -s "$IDENTITY" -f --identifier com.data.facetime-bridge "$BIN"
FACETIME_BRIDGE_AUTHORIZED_CALLER_E164=+15550101001 FACETIME_BRIDGE_AUTHORIZED_CALLER_NAME="Captain Spencer" "$BIN" --self-check 2>/dev/null
cp "$BIN" "$DEST.tmp" && mv "$DEST.tmp" "$DEST"
codesign -dvv "$DEST" 2>&1 | grep -E "Authority|Identifier" | head -2
if launchctl print "gui/$(id -u)/ai.data.facetime-bridge" 2>/dev/null | grep -q "$DEST"; then
  launchctl kickstart -k "gui/$(id -u)/ai.data.facetime-bridge" && echo "daemon restarted"
else
  echo "NOTE: launchd job does not point at $DEST yet — edit the plist, then bootout + bootstrap (kickstart -k re-runs the OLD program)."
fi
