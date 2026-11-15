#!/usr/bin/env bash
# Append one line every 5 s: time, GPU power / temperature / load / memory, CPU package temperature, CPU load,
# free memory, and which users have processes on the GPU. Written line by line, so the last lines before a crash survive it.  Usage: log_hardware.sh LOGFILE
LOG="${1:?log file}"
echo "time,gpu_power_w,gpu_temp_c,gpu_util_pct,gpu_mem_mib,cpu_pkg_temp_c,load_1min,mem_avail_mib,gpu_users" >> "${LOG}"
while true; do
  gpu=$(nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  cpu=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone6/temp 2>/dev/null)
  load=$(cut -d' ' -f1 /proc/loadavg)
  mem=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  users=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r -n1 ps -o user= -p | sort | uniq -c | awk '{printf "%s:%s ", $2, $1}')
  echo "$(date +%T),${gpu},${cpu},${load},${mem},${users}" >> "${LOG}"
  sync -f "${LOG}" 2>/dev/null
  sleep 5
done
