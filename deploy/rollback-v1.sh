#!/bin/bash
# Roll the live stack back to v1-audio-stable (FaceTime Audio only, verified
# 2026-09-13) without rebuilding anything. Restores the signed daemon binary,
# both launchd plists and the web-local plugin from the snapshot taken on
# 2026-09-14, checks out the matching commits in both repos, and restarts.
#
#     deploy/rollback-v1.sh            # roll back
#     deploy/rollback-v1.sh --check    # only show what would change
#
# The snapshot lives outside the repos on purpose:
#     ~/.local/share/dfv-backup/v1-audio-stable/
set -euo pipefail
B="$HOME/.local/share/dfv-backup/v1-audio-stable"
DFV="$HOME/Documents/Programming/Datas_Projects/data-facetime-voice"
BRIDGE="$HOME/Documents/Programming/Datas_Projects/facetime-bridge"
[ -d "$B" ] || { echo "no snapshot at $B"; exit 1; }

echo "snapshot: data-facetime-voice $(cat "$B/data-facetime-voice.commit" | cut -c1-8), facetime-bridge $(cat "$B/facetime-bridge.commit" | cut -c1-8)"
echo "live:     data-facetime-voice $(git -C "$DFV" rev-parse --short HEAD), facetime-bridge $(git -C "$BRIDGE" rev-parse --short HEAD)"
cmp -s "$B/facetime-bridge-daemon" "$HOME/.local/bin/facetime-bridge-daemon" && echo "daemon binary: same" || echo "daemon binary: DIFFERS"
for p in ai.data.facetime-voice ai.data.facetime-bridge; do
  cmp -s "$B/$p.plist" "$HOME/Library/LaunchAgents/$p.plist" && echo "$p.plist: same" || echo "$p.plist: DIFFERS"
done
[ "${1:-}" = "--check" ] && exit 0

for repo in "$DFV" "$BRIDGE"; do
  if [ -n "$(git -C "$repo" status --porcelain)" ]; then
    echo "$repo has uncommitted changes — commit or stash first"; exit 1
  fi
done
git -C "$DFV" checkout -q v1-audio-stable
git -C "$BRIDGE" checkout -q v1-audio-stable
UID_=$(id -u)
launchctl bootout "gui/$UID_/ai.data.facetime-voice" 2>/dev/null || true
launchctl bootout "gui/$UID_/ai.data.facetime-bridge" 2>/dev/null || true
cp "$B/facetime-bridge-daemon" "$HOME/.local/bin/facetime-bridge-daemon"
cp "$B/ai.data.facetime-voice.plist" "$B/ai.data.facetime-bridge.plist" "$HOME/Library/LaunchAgents/"
rm -rf "$HOME/.hermes/plugins/web-local" && cp -R "$B/web-local" "$HOME/.hermes/plugins/web-local"
launchctl bootstrap "gui/$UID_" "$HOME/Library/LaunchAgents/ai.data.facetime-bridge.plist"
sleep 2
launchctl bootstrap "gui/$UID_" "$HOME/Library/LaunchAgents/ai.data.facetime-voice.plist"
echo "rolled back to v1-audio-stable; both repos are on detached tags (git checkout main to return)"
