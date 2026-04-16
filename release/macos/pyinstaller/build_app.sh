#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${1:-}"
BUNDLE_ID="com.alexcreiner.corssdualdynamicroprelease"
IDENTITY="Developer ID Application: Alex Creiner (YJG3692G9D)"

if [[ -z "$APP_NAME" ]]; then
  echo "Usage: ./build_app.sh APP_NAME"
  exit 1
fi

cd "$(dirname "$0")"

pyinstaller \
  -n "$APP_NAME" \
  --clean \
  --noconfirm \
  --windowed \
  --onedir \
  --additional-hooks-dir=. \
  --collect-data modeling_tools \
  --collect-data scienceplots \
  --hidden-import modeling_tools.tools.log_formatter \
  --paths ../../../src \
  --osx-bundle-identifier "$BUNDLE_ID" \
  --codesign-identity "$IDENTITY" \
  --osx-entitlements-file ./entitlements.plist \
  ./main.py

codesign --verify --deep --strict --verbose=2 "dist/${APP_NAME}.zip"
ditto -c -k --keepParent "dist/${APP_NAME}.app" "${APP_NAME}.zip"

xcrun notarytool submit "${APP_NAME}.zip" --keychain-profile "AC_PROFILE" --wait # this takes forever

xcrun stapler staple "dist/${APP_NAME}.app"
xcrun stapler validate "dist/${APP_NAME}.app"

echo "Done."