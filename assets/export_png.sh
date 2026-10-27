#!/usr/bin/env bash
# 把 assets/*.svg 渲染成各尺寸 PNG。
# 用 HTML wrapper 而不是直接截 SVG，这样能精确控制输出尺寸。
set -euo pipefail

ROOT="/Users/viralt/Project/RadTraceRAG"
OUT="$ROOT/assets/png"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
mkdir -p "$OUT"

render() {  # render <svg> <size> <outname>
  local svg="$1" size="$2" name="$3"
  local html="/tmp/_wrap_${name}_${size}.html"
  cat > "$html" <<HTML
<html><body style="margin:0;background:transparent">
<img src="file://$ROOT/assets/$svg" width="$size" height="$size" style="display:block">
</body></html>
HTML
  "$CHROME" --headless --disable-gpu --hide-scrollbars \
    --default-background-color=00000000 \
    --screenshot="$OUT/$name" --window-size="$size,$size" \
    "file://$html" >/dev/null 2>&1
  echo "  $name  ($size)"
}

render_wide() {  # render_wide <svg> <w> <h> <outname>
  local svg="$1" w="$2" h="$3" name="$4"
  local html="/tmp/_wrap_${name}.html"
  cat > "$html" <<HTML
<html><body style="margin:0;background:transparent">
<img src="file://$ROOT/assets/$svg" width="$w" height="$h" style="display:block">
</body></html>
HTML
  "$CHROME" --headless --disable-gpu --hide-scrollbars \
    --default-background-color=00000000 \
    --screenshot="$OUT/$name" --window-size="$w,$h" \
    "file://$html" >/dev/null 2>&1
  echo "  $name  (${w}x${h})"
}

echo "logo-mark:"
for s in 1024 512 256 128 64 32; do render logo-mark.svg "$s" "logo-mark-$s.png"; done

echo "logo-mark-light:"
for s in 512 256 128; do render logo-mark-light.svg "$s" "logo-mark-light-$s.png"; done

echo "favicon:"
for s in 180 32 16; do render favicon.svg "$s" "favicon-$s.png"; done

echo "lockup:"
render_wide logo-lockup.svg 900 200 "logo-lockup.png"
render_wide logo-lockup-light.svg 900 200 "logo-lockup-light.png"

echo
echo "输出目录: $OUT"
ls -1 "$OUT"
