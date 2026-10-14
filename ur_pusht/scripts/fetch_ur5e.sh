#!/usr/bin/env bash
# Fetch the UR5e model from MuJoCo Menagerie (Apache-2.0) at a pinned commit.
set -euo pipefail

MENAGERIE_COMMIT=d7b481bac63d3d600ab404ee74f7474c5efe8a67
DEST="$(cd "$(dirname "$0")/.." && pwd)/src/ur_pusht/assets/universal_robots_ur5e"

if [ -f "$DEST/ur5e.xml" ]; then
  echo "UR5e already present at $DEST"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$tmp/menagerie"
git -C "$tmp/menagerie" checkout -q "$MENAGERIE_COMMIT"
git -C "$tmp/menagerie" sparse-checkout set universal_robots_ur5e
mkdir -p "$(dirname "$DEST")"
cp -r "$tmp/menagerie/universal_robots_ur5e" "$DEST"
echo "UR5e copied to $DEST"
