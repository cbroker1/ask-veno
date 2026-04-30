#!/usr/bin/env bash

input="$(cat)"

MODEL="$(echo "$input" | jq -r '.model.display_name // .model.id // "Claude"')"
CTX="$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)"

# RAM usage
RAM_LINE="$(free -h | awk '/^Mem:/ {print $3 "/" $2}')"

# CPU usage sampled briefly from /proc/stat
read -r cpu user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
prev_idle=$((idle + iowait))
prev_total=$((user + nice + system + idle + iowait + irq + softirq + steal))
sleep 0.15
read -r cpu user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
idle_now=$((idle + iowait))
total_now=$((user + nice + system + idle + iowait + irq + softirq + steal))
diff_idle=$((idle_now - prev_idle))
diff_total=$((total_now - prev_total))

if [ "$diff_total" -gt 0 ]; then
  CPU=$(( (100 * (diff_total - diff_idle)) / diff_total ))
else
  CPU=0
fi

# GPU / VRAM via nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  if [ -n "$GPU_LINE" ]; then
    GPU_UTIL="$(echo "$GPU_LINE" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
    VRAM_USED_MB="$(echo "$GPU_LINE" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
    VRAM_TOTAL_MB="$(echo "$GPU_LINE" | awk -F',' '{gsub(/ /,"",$3); print $3}')"

    VRAM_USED_GB="$(awk "BEGIN {printf \"%.1f\", $VRAM_USED_MB/1024}")"
    VRAM_TOTAL_GB="$(awk "BEGIN {printf \"%.0f\", $VRAM_TOTAL_MB/1024}")"

    GPU_STATUS="gpu:${GPU_UTIL}% vram:${VRAM_USED_GB}G/${VRAM_TOTAL_GB}G"
  else
    GPU_STATUS="gpu:n/a"
  fi
else
  GPU_STATUS="gpu:no-nvidia-smi"
fi

# Battery / AC status if available
if command -v upower >/dev/null 2>&1; then
  BAT="$(upower -i "$(upower -e | grep BAT | head -n 1)" 2>/dev/null | awk '/percentage:/ {print $2}' | head -n 1)"
  [ -z "$BAT" ] && BAT="AC"
else
  BAT="AC"
fi

printf "%s | ctx:%s%% | cpu:%s%% | ram:%s | %s | bat:%s\n" \
  "$MODEL" "$CTX" "$CPU" "$RAM_LINE" "$GPU_STATUS" "$BAT"
