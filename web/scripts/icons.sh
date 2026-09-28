#!/bin/sh
# Render the PWA icons from public/icon.svg (needs rsvg-convert from librsvg).
set -eu
cd "$(dirname "$0")/../public"
rsvg-convert -w 180 icon.svg -o apple-touch-icon.png
rsvg-convert -w 192 icon.svg -o icon-192.png
rsvg-convert -w 512 icon.svg -o icon-512.png
