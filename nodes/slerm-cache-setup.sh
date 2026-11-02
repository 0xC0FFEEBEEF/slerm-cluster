#!/bin/bash
# One-time: local model cache for slerm (/var/cache/slerm-llm, owned by slerm).
# "disk" also formats + mounts the dedicated 100G scsi1 disk (GPU nodes only).
set -euo pipefail
DIR=/var/cache/slerm-llm
if [ "${1:-}" = disk ]; then
  size=$(lsblk -bdno SIZE /dev/sdb)
  [ "$size" = 107374182400 ] || { echo "sdb is $size bytes, not 100G - refusing"; exit 1; }
  if [ -z "$(lsblk -no FSTYPE /dev/sdb | tr -d ' \n')" ] && [ "$(lsblk -no NAME /dev/sdb | wc -l)" = 1 ]; then
    mkfs.xfs -q -L slerm-cache /dev/sdb
  fi
  mkdir -p "$DIR"
  grep -q 'LABEL=slerm-cache' /etc/fstab || \
    echo 'LABEL=slerm-cache /var/cache/slerm-llm xfs defaults,noatime,nofail,x-systemd.device-timeout=10s 0 0' >> /etc/fstab
  systemctl daemon-reload
  mountpoint -q "$DIR" || mount "$DIR"
fi
mkdir -p "$DIR"
chown slerm:slerm "$DIR"; chmod 755 "$DIR"
restorecon -R "$DIR" 2>/dev/null || true
echo "$(hostname -s): $(df -h --output=target,size,avail "$DIR" | tail -1)"
