#!/usr/bin/env bash
# 生成 GitHub Social Preview 图（1280x640）。
#
# GitHub 的 social preview 只能在仓库设置的网页上传，没有 API；
# 所以这个脚本负责「把图做好」，上传那一步由使用者手动完成。
#
# 用法：bash assets/make_social_preview.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/assets/social-preview.png"
CHROME="${CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
W=1280
H=640

if [ ! -x "$CHROME" ]; then
  echo "找不到 Chrome 可执行文件: $CHROME" >&2
  echo "可以用 CHROME_BIN=/path/to/chrome 覆盖。" >&2
  exit 1
fi

HTML="$(mktemp -t social_preview).html"
PROFILE="$(mktemp -d)"
cleanup() { rm -f "$HTML"; rm -rf "$PROFILE"; }
trap cleanup EXIT

cat > "$HTML" <<HTML
<html><head><meta charset="utf-8">
<style>
  @page { margin: 0; }
  * { box-sizing: border-box; }
  body {
    margin: 0; width: ${W}px; height: ${H}px; overflow: hidden;
    background:
      radial-gradient(900px 520px at 78% 18%, rgba(34,211,238,.16), transparent 62%),
      radial-gradient(760px 520px at 12% 92%, rgba(139,147,248,.15), transparent 62%),
      linear-gradient(135deg, #111C2E 0%, #0A101B 55%, #060A11 100%);
    font-family: 'PingFang SC', 'Hiragino Sans GB', -apple-system, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
    color: #E8F1FA;
  }
  .stage { display: flex; height: 100%; padding: 64px 68px; gap: 56px; }
  .left { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .brand { display: flex; align-items: center; gap: 20px; }
  .brand img { width: 76px; height: 76px; display: block; }
  .word { font-size: 40px; font-weight: 600; letter-spacing: -1.1px; color: #F2F7FC; }
  .word span {
    background: linear-gradient(90deg, #22D3EE, #38BDF8 52%, #8B93F8);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  h1 { margin: 46px 0 0; font-size: 46px; line-height: 1.22; font-weight: 700; letter-spacing: -0.6px; }
  h1 em { font-style: normal; color: #7DD3FC; }
  .sub { margin: 20px 0 0; font-size: 21px; line-height: 1.6; color: #90A9C4; max-width: 620px; }
  .chips { margin-top: auto; display: flex; gap: 12px; flex-wrap: wrap; }
  .chip {
    font-size: 15px; color: #A9C3DC; padding: 9px 16px; border-radius: 999px;
    border: 1px solid rgba(125,211,252,.24); background: rgba(56,189,248,.07);
  }
  .right { width: 396px; flex: none; }
  .panel {
    height: 100%; border-radius: 20px; padding: 26px 26px 22px;
    background: rgba(255,255,255,.045);
    border: 1px solid rgba(125,211,252,.22);
    box-shadow: 0 24px 60px rgba(0,0,0,.42);
    display: flex; flex-direction: column;
  }
  .panel h2 {
    margin: 0; font-size: 16px; font-weight: 600; color: #CFE2F3;
    display: flex; align-items: center; gap: 10px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #FBBF24; box-shadow: 0 0 12px #FBBF24; }
  .meta { margin: 8px 0 20px; font-size: 13px; color: #7B93AD; }
  .ev { display: flex; gap: 13px; padding: 13px 0; border-top: 1px solid rgba(125,211,252,.13); }
  .tag {
    flex: none; width: 34px; height: 26px; border-radius: 7px; font-size: 13px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    color: #06202C; background: linear-gradient(135deg, #22D3EE, #38BDF8);
  }
  .tag.amber { color: #2A1A03; background: linear-gradient(135deg, #FBBF24, #F59E0B); }
  .txt { font-size: 15px; line-height: 1.5; color: #C6D8E9; }
  .src { font-size: 12px; color: #6F87A2; margin-top: 4px; }
  .check {
    margin-top: auto; padding-top: 18px; border-top: 1px dashed rgba(125,211,252,.2);
    font-size: 13px; color: #86D9C0; display: flex; align-items: center; gap: 8px;
  }
</style></head>
<body><div class="stage">
  <div class="left">
    <div class="brand">
      <img src="file://$ROOT/assets/logo-mark.svg">
      <div class="word">RadTrace<span>RAG</span></div>
    </div>
    <h1>语料无关的<br><em>LightRAG 检索引擎</em></h1>
    <div class="sub">ingest → query 全链路本机跑通。每条回答都能回溯到 chunk 级证据，编造的引用编号会被自动剔除。</div>
    <div class="chips">
      <div class="chip">Python 3.11+</div>
      <div class="chip">LightRAG 1.5.7</div>
      <div class="chip">60 tests green</div>
      <div class="chip">MIT</div>
    </div>
  </div>
  <div class="right"><div class="panel">
    <h2><span class="dot"></span>检索轨迹 · mix 模式</h2>
    <div class="meta">5 条证据 · 412 ms · 全部引用校验通过</div>
    <div class="ev">
      <div class="tag">E1</div>
      <div><div class="txt">肝脏负责代谢与解毒，合成胆汁并调节血糖</div>
      <div class="src">samples/02_liver_function.md</div></div>
    </div>
    <div class="ev">
      <div class="tag">E2</div>
      <div><div class="txt">胆囊储存并浓缩肝细胞分泌的胆汁</div>
      <div class="src">samples/01_abdominal_anatomy.md</div></div>
    </div>
    <div class="ev">
      <div class="tag amber">E3</div>
      <div><div class="txt">CT 成像依赖组织密度差异形成灰度对比</div>
      <div class="src">samples/03_ct_imaging_basics.md</div></div>
    </div>
    <div class="ev">
      <div class="tag">E4</div>
      <div><div class="txt">胰腺兼具外分泌与内分泌功能，分泌胰酶与胰岛素</div>
      <div class="src">samples/04_spleen_pancreas.md</div></div>
    </div>
    <div class="ev">
      <div class="tag">E5</div>
      <div><div class="txt">脾脏参与血液过滤与免疫应答</div>
      <div class="src">samples/04_spleen_pancreas.md</div></div>
    </div>
    <div class="check">✓ 引用校验通过 —— 正文引用均指向实际存在的证据</div>
  </div></div>
</div></body></html>
HTML

"$CHROME" --headless --disable-gpu --hide-scrollbars \
  --user-data-dir="$PROFILE" \
  --default-background-color=00000000 \
  --screenshot="$OUT" --window-size="$W,$H" \
  "file://$HTML" >/dev/null 2>&1

echo "已生成: $OUT (${W}x${H})"
echo "上传路径: 仓库 Settings → General → Social preview → Edit"
