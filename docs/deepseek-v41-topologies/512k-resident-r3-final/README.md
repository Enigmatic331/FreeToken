# Versioned production deployment

This directory preserves the exact production deployment used by the
[512K resident-r3 topology](../512k-resident-r3-final.md). It is deliberately
machine-specific: the service contains qualified GPU UUIDs, local model paths,
the release worktree path, and the stable Docker ingress address. Review those
values before using it on another host.

Vision is enabled by the production unit with:

```text
FREETOKEN_LOAD_VISION=1
--vision-device 0
```

Logical device 0 is the 48 GiB RTX 4090 that owns the text backbone and
attention/KV authority. The startup prewarm must return exact OCR `7429` from
the sealed fixture before the stable ingress socket is opened.

## Contents

- `systemd/freetoken-dsv41-production.service`: final model geometry and native
  vision placement.
- `systemd/freetoken-dsv41-production-prewarm.service`: per-instance readiness
  gate.
- `systemd/freetoken-dsv41-production-ingress.*`: stable ingress exposed only
  after the readiness gate passes.
- `systemd/freetoken-dsv41-production-watchdog.*`: bounded two-strike recovery.
- `scripts/freetoken-dsv41-production-prewarm.py`: stochastic, exact text,
  omitted-output-token, native-vision, and cleanup gates.
- `scripts/freetoken-dsv41-production-watchdog.py`: local service watchdog.
- `fixtures/ocr-7429.png`: deterministic native-vision prewarm fixture.

## Host installation

After reviewing all machine-specific paths and UUIDs:

```bash
sudo install -m 0644 systemd/* /etc/systemd/system/
sudo install -m 0755 scripts/freetoken-dsv41-production-prewarm.py \
  /usr/local/libexec/freetoken-dsv41-production-prewarm.py
sudo install -m 0755 scripts/freetoken-dsv41-production-watchdog.py \
  /usr/local/libexec/freetoken-dsv41-production-watchdog.py
sudo install -D -m 0644 fixtures/ocr-7429.png \
  /usr/local/share/freetoken-dsv41/ocr-7429.png
sudo systemctl daemon-reload
sudo systemctl enable freetoken-dsv41-production.service \
  freetoken-dsv41-production-prewarm.service \
  freetoken-dsv41-production-ingress.socket \
  freetoken-dsv41-production-watchdog.timer
```

The ingress socket depends on a successful prewarm receipt for the current
model instance. Do not bypass that ordering: it prevents Sembang or Tokios from
reaching an unqualified cold process.

The model unit intentionally pulls in the prewarm unit but does not directly
pull in the ingress socket. The separately enabled socket waits for prewarm.
Adding the socket back to the model unit's `Wants=` creates the ordering cycle
model -> socket -> prewarm -> model, causing systemd to discard the initial
prewarm job. The watchdog remains a recovery path, not the normal boot path.

The prewarm receipt uses a per-process temporary filename before atomic rename.
This avoids a stale `prewarm.json.tmp` from an interrupted or privileged manual
run blocking the unprivileged service on the next boot.

## Qualified Ada boot state

This repository records FreeToken's serving configuration, not the external
`gpu-unlock` DKMS source or firmware. On the qualified host, the normal GRUB
entry passes exact PCI-address targets for the RTX 4090 and experimental AD103
RTX 4080 SUPER:

```text
gpu_unlock.targets=0000:c1:00.0,0000:45:00.0 gpu_unlock.ad103_experimental=1
```

Do not use `targets=all`. Production acceptance requires UUID-bound RM
readbacks of `fmla32=0`, `imla2=0`, and `imla4=0` for both cards, plus the
sealed 40K and decode gates. The combined qualification measured 31.8289
seconds median 40K TTFT / 1,256.72 prefill token/s and 21.5084 decode token/s
with exact outputs and clean safety telemetry.
