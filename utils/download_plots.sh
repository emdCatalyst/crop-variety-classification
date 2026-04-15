#!/usr/bin/env bash
#  MuST-C UAV3-MS plot-wise downloader
#  - Skips already-extracted plots
#  - Resumes interrupted downloads (wget -c)
#  - Retries failed downloads up to 3 times
#  - Validates zip before extracting
#  - Logs failures to failed_plots.txt for manual re-run

set -euo pipefail

OUTDIR="../mustc_plots"
LOGFILE="$OUTDIR/failed_plots.txt"
BASE="https://bonndata.uni-bonn.de/api/access/datafile/:persistentId?persistentId=doi:10.60507/FK2/OX9XTM"
MAX_RETRIES=3

mkdir -p "$OUTDIR"
> "$LOGFILE"   # clear log on fresh run

declare -A PLOTS=(
  [162]=GYB7NN [163]=5MPGQ0 [164]=0ID4KJ [165]=CNS2ES [166]=WVFSXD
  [167]=T8H9V7 [168]=N7O1US [169]=ANMNNH [170]=HGZAFS [171]=ICGSE5
  [172]=WDHOQY [173]=PITNXD [174]=9ODFEX [175]=VLCSRS [176]=AK2VIF
  [177]=UG1OPY [178]=WREYEI [179]=YWRXBW [180]=BFLR95 [181]=EIJ7JO
  [182]=LQJDYZ [183]=KJHASM [184]=YEE1QG [185]=JG5BQO [186]=RFNTNU
  [187]=LDRVHO [188]=CEBODL [189]=3UDCBL [190]=EYVGJ1 [191]=EYTUIY
  [192]=UKBGBL [193]=ZTDTJK [194]=CEJ44M [195]=AUA8AQ [196]=1WQFXU
  [197]=JUSBSI [198]=RRXG5A [199]=HA7TPZ [200]=DYWAQU [201]=FS9O39
  [202]=QKB7HF [203]=UWYYKF [204]=YI4BAT [205]=N8IYID [206]=EV5OHJ
  [207]=TYOEFS [208]=BDQQR8 [209]=CNKKLM [210]=8NHEUG [211]=FWWMGK
  [212]=KZVVNB [213]=FRODCZ [214]=U5EVCD [215]=65A9DA [216]=BL7B4H
  [217]=DHPH66 [218]=RLOKH9 [219]=N6QTDZ [220]=F8DNLA [221]=CBPHAM
  [222]=3JEE0L [223]=TOGCB6 [224]=89HWHJ [225]=YFZ0CB [226]=KXIXKP
  [227]=CRWL37 [228]=CYF60J [229]=X7I3WP [230]=8UTT1F [231]=TGADO3
  [232]=YI7W7X [233]=DNWHV0 [234]=FO8ARH [235]=T4X7B2 [236]=NZ1GRO
  [237]=TFLQFR [238]=UCERNP [239]=XTIXVN [240]=0GQECZ [241]=URKZLD
)

total=${#PLOTS[@]}
count=0

for plot_id in $(echo "${!PLOTS[@]}" | tr ' ' '\n' | sort -n); do
  token="${PLOTS[$plot_id]}"
  zip_file="$OUTDIR/plot_${plot_id}.zip"
  extract_dir="$OUTDIR/plot_${plot_id}"
  count=$((count + 1))

  echo "------------------------------------"
  echo "[$count/$total] Plot $plot_id"

  if [ -d "$extract_dir" ] && [ "$(ls -A "$extract_dir")" ]; then
    echo "  Already extracted, skipping"
    continue
  fi

  success=false
  for attempt in $(seq 1 $MAX_RETRIES); do
    echo "  Download attempt $attempt/$MAX_RETRIES..."
    if wget -c -q --show-progress \
            --timeout=60 \
            --tries=1 \
            -O "$zip_file" \
            "${BASE}/${token}"; then
      success=true
      break
    else
      echo "  Attempt $attempt failed"
      sleep $((attempt * 5))   # back-off: 5s, 10s, 15s
    fi
  done

  if [ "$success" = false ]; then
    echo "  FAILED after $MAX_RETRIES attempts — logged"
    echo "$plot_id" >> "$LOGFILE"
    rm -f "$zip_file"   
    continue
  fi

  echo "  Validating zip..."
  if ! unzip -t "$zip_file" > /dev/null 2>&1; then
    echo "  Zip is corrupt — removing and logging"
    echo "$plot_id (corrupt zip)" >> "$LOGFILE"
    rm -f "$zip_file"
    continue
  fi

  echo "  Extracting..."
  mkdir -p "$extract_dir"
  if ! unzip -q "$zip_file" -d "$extract_dir"; then
    echo "  Extraction failed — logging"
    echo "$plot_id (extraction failed)" >> "$LOGFILE"
    rm -rf "$extract_dir"
    continue
  fi

  rm -f "$zip_file"   
  echo "  Done"
done

echo ""
echo ""------------------------------------"
echo "  Download complete"
echo "  Total plots : $total"
echo "  Extracted   : $(ls -d "$OUTDIR"/plot_*/ 2>/dev/null | wc -l)"
if [ -s "$LOGFILE" ]; then
  echo "  Failed     : $(wc -l < "$LOGFILE") plot(s) — see $LOGFILE"
  echo "  Re-run this script to retry failed plots automatically"
else
  echo "  No failures"
fi
echo ""------------------------------------"
shutdown