#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MACOS_DIR="$ROOT/release/macos"
PAYLOAD_DIR="$MACOS_DIR/pkg_pipeline/payload"
SCRIPTS_DIR="$MACOS_DIR/pkg_pipeline/scripts"
DIST_DIR="$MACOS_DIR/pkg_pipeline/dist"

APP_SUPPORT_DST="$PAYLOAD_DIR/Library/Application Support/Modeling-Tools"
APP_BUNDLE_DST="$PAYLOAD_DIR/Applications/Modeling Tools.app"

rm -rf "$PAYLOAD_DIR" "$DIST_DIR/ModelingTools.pkg"
mkdir -p "$APP_SUPPORT_DST" "$PAYLOAD_DIR/Applications" "$DIST_DIR"

# Copy project files
cp "$ROOT/pyproject.toml" "$APP_SUPPORT_DST/"
cp "$ROOT/README.md" "$APP_SUPPORT_DST/" || true
cp -R "$ROOT/src" "$APP_SUPPORT_DST/"

# Wrapper app should already exist in packaging/macos/app-template, or build it here:
mkdir -p "$APP_BUNDLE_DST/Contents/MacOS"
mkdir -p "$APP_BUNDLE_DST/Contents/Resources"

cp "$MACOS_DIR/pkg_pipeline/app-template/Info.plist" "$APP_BUNDLE_DST/Contents/Info.plist"
cp "$MACOS_DIR/pkg_pipeline/app-template/Modeling Tools" "$APP_BUNDLE_DST/Contents/MacOS/Modeling Tools"
chmod +x "$APP_BUNDLE_DST/Contents/MacOS/Modeling Tools"
cp "$ROOT/src/modeling_tools/assets/icon.icns" "$APP_BUNDLE_DST/Contents/Resources/AppIcon.icns"

pkgbuild \
  --root "$PAYLOAD_DIR" \
  --scripts "$SCRIPTS_DIR" \
  --identifier "edu.bc.modeling-tools.installer" \
  --version "1.0.0" \
  "$DIST_DIR/ModelingTools.pkg"

echo "Built: $DIST_DIR/ModelingTools.pkg"