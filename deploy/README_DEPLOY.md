# Deploy: scoring + Telegram bots on a VPS, scraper + dashboard stay local

As-built notes for the Hetzner deploy. The VPS (Hetzner **CX23**, x86, 2 vCPU / 4 GB,
Ubuntu 24.04) runs the **scoring loop + both Telegram bots**. The **scraper and the
Streamlit dashboard stay on the laptop**.

## What runs where — and why

| Component | Where | Why |
| --- | --- | --- |
| Collector (scraper) | **Laptop** (home IP) | Datacenter IPs get rate-limited harder; keep on the residential IP |
| Scoring loop (`basic_5_giant_model`, `--send-telegram`) | **VPS** | CPU-only, IP-agnostic, no images needed |
| `bot.py`, `image_bot.py` (Telegram pollers) | **VPS** | Share accountability/event state with scoring |
| Streamlit `app.py` (dashboard) | **Laptop** | Reads the full multi-GB `data/` tree + needs torch/pyiqa/libGL — impractical to ship to a 40 GB box. It's read-only viz of laptop data. |

Data flow is **one-way**: collector writes CSVs on the laptop → rsync to VPS → scoring/bots read them.

> ⚠️ The repo layout the VPS needs (`experiments/current/...`) only existed once the
> `scripts/experiments → experiments` restructure was **committed** (commit `1c5b2c3`).
> A fresh clone must be at/after that commit, or paths won't resolve.

---

## 1. Provision (Hetzner)

- Type **CX23** (2 vCPU / 4 GB / 40 GB), Image **Ubuntu 24.04**, add your SSH public key.
- No inbound port needed (Telegram is outbound polling; the dashboard is not on the VPS).

## 2. Base setup on the VPS

```bash
ssh root@VPS_IP
# non-root service user
adduser --disabled-password --gecos "" vinted
install -d -m700 -o vinted -g vinted /home/vinted/.ssh
cp /root/.ssh/authorized_keys /home/vinted/.ssh/authorized_keys
chown vinted:vinted /home/vinted/.ssh/authorized_keys && chmod 600 /home/vinted/.ssh/authorized_keys
# 2 GB swap (CX23 has 4 GB RAM)
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
apt-get update -qq && apt-get install -y git curl
su - vinted
```

```bash
# Miniconda x86
curl -fsSL -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && exec bash
# Recent Miniconda gates the default channels behind a ToS prompt — accept it:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n vinted_scraper python=3.11

git clone <REPO_URL> $HOME/Vinted_New_Version
cd $HOME/Vinted_New_Version
# Repo has no requirements.txt; use the committed VPS set (no torch/pyiqa/CUDA):
$HOME/miniconda3/envs/vinted_scraper/bin/pip install -r deploy/requirements_vps.txt
```

There is **no committed `requirements.txt`**. `deploy/requirements_vps.txt` is the lean,
torch-free dependency set for scoring + bots. (The dashboard's heavier deps —
torch/pyiqa/opencv/libGL — are deliberately excluded since it runs on the laptop.)

## 3. Copy secrets + models (once, from laptop)

```bash
scp .env                                   vinted@VPS_IP:~/Vinted_New_Version/.env
# the main bot reads BOT_TOKEN from a SEPARATE, gitignored file:
scp scripts/telegram_scripts/bot_env.env   vinted@VPS_IP:~/Vinted_New_Version/scripts/telegram_scripts/bot_env.env

# model pickles (new layout path) + the two threshold tables scoring actually reads:
B=experiments/current/basic_5_giant_model/data
rsync -az $B/models/ vinted@VPS_IP:~/Vinted_New_Version/$B/models/
for r in basic_5_giant_20260525_185552 basic_5_giant_20260614_221642; do
  ssh vinted@VPS_IP "mkdir -p ~/Vinted_New_Version/$B/offline_runs/$r"
  scp $B/offline_runs/$r/per_search_threshold_metrics.csv vinted@VPS_IP:~/Vinted_New_Version/$B/offline_runs/$r/
done
```

`.env` holds `IMAGE_BOT_TOKEN`, `RECOMMENDED_DEALS_CHAT_ID`, etc. `BOT_TOKEN` lives only in
`scripts/telegram_scripts/bot_env.env` — copy both or `vinted-bot` fails with
`BOT_TOKEN is not configured`.

## 4. First data sync + recurring sync (from laptop)

`deploy/sync_to_vps.sh` pushes the collector run-dir CSVs (excludes `visual_features/`,
`raw_snapshots/`, `image_cache/`, `rechecks/` — see `deploy/rsync-exclude.txt`). Edit the
vars at its top (`VPS_HOST`, `VPS_REPO`, `SSH_KEY`), then:

```bash
ssh vinted@VPS_IP 'mkdir -p ~/Vinted_New_Version/data/experiments/time_to_sell/live_runs/bin_collector_20260602_214104'
deploy/sync_to_vps.sh   # first push ≈ 200 MB
```

Recurring sync runs as a **laptop systemd --user timer** (cron was blocked by
`/var/spool/cron` perms): `~/.config/systemd/user/vinted-sync.{service,timer}`, every 10 min.

## 5. Services on the VPS (systemd --user)

```bash
loginctl enable-linger vinted
mkdir -p ~/.config/systemd/user
cp ~/Vinted_New_Version/deploy/systemd/vinted-scoring.service \
   ~/Vinted_New_Version/deploy/systemd/vinted-bot.service \
   ~/Vinted_New_Version/deploy/systemd/vinted-image-bot.service \
   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now vinted-scoring vinted-bot vinted-image-bot
journalctl --user -u vinted-scoring -f
```

(`deploy/systemd/vinted-streamlit.service` exists but is **not** used — the dashboard is local.)

## 6. ⚠️ Cutover — move, don't duplicate

- **One poller per bot token** — two `bot.py` on the same token → `Conflict: terminated by other getUpdates`.
- **Two scoring loops with `--send-telegram`** → every deal sent **twice**.

So on the laptop, stop the local scoring loop before the VPS one sends. Leave the collector running:

```bash
pkill -f 'apply_to_live_collector.py .*--run-loop'
```

## 7. Laptop persistence (systemd --user)

Collector, dashboard, and the sync timer run as laptop user units so they survive reboot
(`loginctl enable-linger ale`):

| unit | what |
| --- | --- |
| `vinted-collector.service` | the scraper `run-loop` |
| `vinted-dashboard.service` | Streamlit on `127.0.0.1:8501` (localhost only) |
| `vinted-sync.timer` | pushes CSVs to the VPS every 10 min |

Dashboard is bound to `127.0.0.1`. To reach it from another LAN device, change
`--server.address` to `0.0.0.0` in `vinted-dashboard.service`.

## 8. Verify

```bash
# VPS
ssh vinted@VPS_IP 'for s in vinted-scoring vinted-bot vinted-image-bot; do echo "$s: $(systemctl --user is-active $s)"; done; free -h | sed -n 2p; df -h /'
# laptop
systemctl --user is-active vinted-collector vinted-dashboard vinted-sync.timer
```

## Rollback

```bash
# VPS
systemctl --user disable --now vinted-scoring vinted-bot vinted-image-bot
# laptop: restart the local scoring loop (e.g. re-enable a local scoring unit or run --run-loop)
```
