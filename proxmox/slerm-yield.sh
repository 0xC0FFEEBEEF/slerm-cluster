#!/bin/bash
# slerm-yield - Proxmox hookscript: the user's VMs outrank slerm.
#
# Attached to every NON-slerm VM on this host (slerm-yield-ensure keeps it
# that way). In pre-start it shuts down whichever slerm node VMs are in the
# way, so the user just presses Start as usual:
#   * a slerm VM that passes through the same PCI device (GamingBox and
#     slerm-node03 share the GTX 1660 SUPER at 04:00) always goes;
#   * then slerm VMs, least valuable first, until there is RAM for this VM.
# slerm-head (300) is never touched: it is slurmctld and the NFS server for
# every node. The slerm orchestrator notices the nodes are gone and re-plans;
# it brings them back on demand once the RAM is free again.
#
# A PCI-passthrough VM pins ALL of its RAM, and byteme has no swap: starting
# one into too little memory OOM-kills VMs that are already running
# (2026-08-31); any other VM does the same later, as it touches its RAM. So if
# even stopping every slerm node is not enough, the start is REFUSED.
# Override (at your own risk): touch /etc/slerm-yield/allow-overcommit
set -u
vmid="$1"; phase="$2"
LOG=/var/log/slerm-yield.log
MARGIN_MB=1536
log() { echo "$(date -Is) [$vmid] $*" | tee -a "$LOG" >&2; }

[ "$phase" = pre-start ] || exit 0

case "$(hostname -s)" in
    byteme)  SLERM_ORDER="304 303" ;;      # 980 Ti node first: slower card
    byteme2) SLERM_ORDER="302 301" ;;
    *)       exit 0 ;;
esac
case " 300 $SLERM_ORDER " in *" $vmid "*) exit 0 ;; esac

cfg() { qm config "$1" 2>/dev/null; }
pcis() { cfg "$1" | awk -F': ' '/^hostpci[0-9]+:/ { split($2, a, ","); sub(/^0000:/, "", a[1]); sub(/\.[0-9]$/, "", a[1]); print a[1] }'; }
mem_mb() { cfg "$1" | awk -F': ' '$1 == "memory" { print $2 }'; }
running() { qm status "$1" 2>/dev/null | grep -q "status: running"; }
avail_mb() { awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo; }

want_pci="$(pcis "$vmid")"
need=$(( $(mem_mb "$vmid") + MARGIN_MB ))
victims=""
freed=0
for s in $SLERM_ORDER; do
    running "$s" || continue
    for p in $(pcis "$s"); do
        if grep -qx "$p" <<<"$want_pci"; then
            victims="$victims $s"; freed=$((freed + $(mem_mb "$s")))
            log "slerm VM $s holds PCI $p that this VM needs"
            break
        fi
    done
done
avail=$(avail_mb)
for s in $SLERM_ORDER; do
    (( avail + freed >= need )) && break
    running "$s" || continue
    case " $victims " in *" $s "*) continue ;; esac
    victims="$victims $s"; freed=$((freed + $(mem_mb "$s")))
done

if (( avail + freed < need )); then
    # Refused for every VM, not only passthrough: a guest touches its RAM
    # sooner or later (lazily, unless it pins it), and byteme has no swap, so
    # the OOM killer would take slerm-head or another VM when it does.
    if [ ! -e /etc/slerm-yield/allow-overcommit ]; then
        log "REFUSED: needs ${need} MB (incl. ${MARGIN_MB} margin); ${avail} free + ${freed} from slerm nodes is not enough." \
            "Lower its memory, or stop another VM first. Override: touch /etc/slerm-yield/allow-overcommit"
        echo "slerm-yield: not enough RAM for VM $vmid even after stopping the slerm nodes (see $LOG)" >&2
        exit 1
    fi
    log "WARNING: ${need} MB wanted, only $((avail + freed)) reachable; starting anyway (override set)"
fi

[ -z "$victims" ] && { log "no slerm VM in the way (${avail} MB free, ${need} needed)"; exit 0; }
log "stopping slerm VM(s)${victims} to start this VM (${avail} MB free, ${need} needed)"
if [ "${SLERM_YIELD_DRY:-0}" = 1 ]; then
    log "DRY RUN: would shut down${victims}"; exit 0
fi
for s in $victims; do
    qm shutdown "$s" --timeout 120 --forceStop 1 >>"$LOG" 2>&1 &
done
wait
for s in $victims; do running "$s" && log "slerm VM $s is STILL running"; done
log "done; $(avail_mb) MB free"
exit 0
