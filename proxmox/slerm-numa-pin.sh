#!/bin/bash
# slerm-numa-pin - Proxmox hookscript for slerm-node04 (VM 304), 2026-10-10.
#
# Since the RAM move byteme has two unequal memory pools: CPU1 (host NUMA node 0,
# even host CPUs) holds one 32 GB DIMM = one memory channel, CPU2 (node 1, odd
# host CPUs) holds 2x 16 GB = two channels. VM 304 mirrors that with two virtual
# sockets / NUMA nodes; its numa0/numa1 config lines bind the guest RAM to the
# matching host node. Proxmox binds the memory but lets the vCPU threads float
# over both sockets, which would make the guest's NUMA map wrong. So:
#
#   pre-start : drop the host page cache and compact memory, so the bound and
#               pinned (PCI passthrough) guest RAM finds free pages on each node.
#   post-start: pin every "CPU n/KVM" thread to the CPUs of the host node its
#               numaN entry names (they still float within that socket).
#
# It never blocks a start: failures are logged and the script exits 0.
# Log: /var/log/slerm-numa-pin.log
set -u
vmid="$1"; phase="$2"
LOG=/var/log/slerm-numa-pin.log
log() { echo "$(date -Is) [$vmid] $*" >> "$LOG"; }
placed() {      # guest RAM per host node, from the QEMU process
    awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^N[0-9]+=/) { split($i, a, "="); n[a[1]] += a[2] } }
         END { for (k in n) printf "%s %d MB  ", k, n[k] * 4 / 1024 }' "/proc/$1/numa_maps" 2>/dev/null
}
free_per_node() {
    for n in /sys/devices/system/node/node[0-9]*; do
        awk -v n="${n##*node}" '/MemFree/ { printf "node%s %d MB  ", n, $4 / 1024 }' "$n/meminfo"
    done
}
# "0-3;8" (Proxmox list) -> one number per line
expand() {
    local part a b
    for part in ${1//;/ }; do
        a=${part%-*}; b=${part#*-}
        seq "$a" "$b"
    done
}

case "$phase" in
pre-start)
    sync
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null
    echo 1 > /proc/sys/vm/compact_memory 2>/dev/null
    log "pre-start: page cache dropped, free: $(free_per_node)"
    ;;
post-start)
    pid=$(cat "/run/qemu-server/$vmid.pid" 2>/dev/null)
    [ -n "$pid" ] && [ -d "/proc/$pid/task" ] || { log "post-start: no QEMU process found, nothing pinned"; exit 0; }
    declare -A cpus_of=()         # vCPU index -> host cpu list
    mapped=0
    while IFS= read -r line; do
        vc=$(sed -n 's/.*cpus=\([0-9;-]*\).*/\1/p' <<<"$line")
        hn=$(sed -n 's/.*hostnodes=\([0-9;-]*\).*/\1/p' <<<"$line")
        [ -n "$vc" ] && [ -n "$hn" ] || continue
        list=""
        for n in $(expand "$hn"); do
            l=$(cat "/sys/devices/system/node/node$n/cpulist" 2>/dev/null) || continue
            list="${list:+$list,}$l"
        done
        [ -n "$list" ] || continue
        for v in $(expand "$vc"); do cpus_of[$v]=$list; mapped=$((mapped + 1)); done
    done < <(qm config "$vmid" 2>/dev/null | grep '^numa[0-9]\+: ')
    [ "$mapped" -gt 0 ] || { log "post-start: no numaN lines with hostnodes, nothing pinned; guest RAM on: $(placed "$pid")"; exit 0; }
    done_n=0; fail_n=0
    for t in /proc/"$pid"/task/*; do
        name=$(cat "$t/comm" 2>/dev/null) || continue
        case "$name" in "CPU "*"/KVM") ;; *) continue ;; esac
        v=${name#CPU }; v=${v%/KVM}
        [ -n "${cpus_of[$v]:-}" ] || continue
        if taskset -pc "${cpus_of[$v]}" "${t##*/}" >/dev/null 2>&1; then done_n=$((done_n + 1)); else fail_n=$((fail_n + 1)); fi
    done
    log "post-start: pinned $done_n vCPU threads to their host node ($fail_n failed); guest RAM on: $(placed "$pid") free: $(free_per_node)"
    ;;
esac
exit 0
