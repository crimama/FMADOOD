#!/usr/bin/env bash
# Resume-safe Real-IAD realiad_1024 downloader for dsba3 (/media/hunim/realiad).
# Needs a Hugging Face read token whose account has accepted Real-IAD access.
# The token file is passed to curl through a config file; its value is never printed.
set -euo pipefail
DEST="${REAL_IAD_ROOT:-/media/hunim/realiad}"
TOKEN_FILE="${HF_TOKEN_FILE:-/media/hunim/.hf_token}"
REPO="https://huggingface.co/datasets/Real-IAD/Real-IAD"
JOBS="${JOBS:-4}"
[[ -s "$TOKEN_FILE" ]] || { echo "Missing token file: $TOKEN_FILE" >&2; exit 2; }
chmod 600 "$TOKEN_FILE"
mkdir -p "$DEST/realiad_1024" "$DEST/_logs"
CURL_CFG="$(mktemp)"; trap 'rm -f "$CURL_CFG"' EXIT
printf 'header = "Authorization: Bearer %s"\n' "$(tr -d '\r\n' < "$TOKEN_FILE")" > "$CURL_CFG"; chmod 600 "$CURL_CFG"
python3 - "$DEST" "$TOKEN_FILE" <<'PY'
import json, sys, urllib.request
d=sys.argv[1]
u='https://huggingface.co/api/datasets/Real-IAD/Real-IAD/tree/main/realiad_1024'
tok=open(sys.argv[2]).read().strip()
items=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'Authorization':'Bearer '+tok}),timeout=60))
rows=[(x['path'],x.get('lfs',{}).get('size',x.get('size',0)),x.get('lfs',{}).get('oid','')) for x in items if x['path'].endswith('.zip')]
rows.append(('realiad_jsons.zip',3230739,''))
open(d+'/manifest.tsv','w').write(''.join('%s\t%d\t%s\n'%r for r in rows))
print(len(rows),'files',round(sum(r[1] for r in rows)/2**30,1),'GiB')
PY
fetch() {
  local path="$1" size="$2" sha="$3" out="$DEST/$1"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out.verified" && -f "$out" && "$(stat -c %s "$out")" == "$size" ]]; then return 0; fi
  if [[ ! -f "$out" || "$(stat -c %s "$out")" != "$size" ]]; then
    curl -fL --retry 8 --retry-delay 5 -C - -K "$CURL_CFG" -o "$out" "$REPO/resolve/main/$path" 2>>"$DEST/_logs/curl_$(basename "$path").log"
  fi
  [[ "$(stat -c %s "$out")" == "$size" ]] || { echo "SIZE_MISMATCH $path" >&2; return 1; }
  if [[ -n "$sha" ]]; then echo "$sha  $out" | sha256sum -c --quiet - || { echo "SHA_MISMATCH $path" >&2; return 1; }; fi
  touch "$out.verified"
  echo "OK $path"
}
export -f fetch; export DEST REPO CURL_CFG
cut -f1-3 "$DEST/manifest.tsv" | xargs -P "$JOBS" -L1 bash -c 'fetch "$0" "$1" "$2"'
echo DOWNLOAD_COMPLETE
