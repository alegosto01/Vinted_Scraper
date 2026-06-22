# Deploy: scoring + Telegram bots + dashboard on a VPS, scraper stays local

Target VPS: **Hetzner Cloud CX22** (x86, 2 vCPU / 4 GB), Ubuntu 22.04/24.04.
Sync: **cron rsync over SSH**, laptop → VPS, one-way.

## What runs where

| Component | Where | Why |
| --- | --- | --- |
| Collector (scraper) | **Laptop** (home IP) | Datacenter IPs get rate-limited harder; keep on residential IP |
| Scoring loop (`basic_5_giant_model`, `--send-telegram`) | VPS | CPU-bound, IP-agnostic, no images needed |
| `bot.py`, `image_bot.py` (Telegram pollers) | VPS | Share accountability/event state with scoring |
| Streamlit `app.py` | VPS | Reads synced CSVs + local scoring/telegram state |

Data flow is **one-way**: collector writes CSVs on the laptop → rsync to VPS → scoring/bot/streamlit read them. Nothing writes back to the laptop.

---

## 1. Provision the Hetzner instance

- Hetzner Cloud Console → new project → add server.
- Type: **CX22** (2 vCPU, 4 GB). Bump to **CX32** (8 GB) only if RAM gets tight later.
- Image: Ubuntu 22.04/24.04. Add your SSH public key.
- Location: a region near you / near Vinted IT (Nuremberg/Falkenstein fine).
- No inbound port for Telegram (outbound polling only). Do **not** open 8501 publicly — reach Streamlit via SSH tunnel (step 8). Optionally enable Hetzner Cloud Firewall allowing inbound 22 only.

You log in as `root` by default.

## 2. Base setup on the VPS

```bash
ssh root@YOUR_VPS_IP

# --- non-root service user (don't run bots/scoring as root) ---
adduser --disabled-password --gecos "" vinted
install -d -m700 -o vinted -g vinted /home/vinted/.ssh
cp ~/.ssh/authorized_keys /home/vinted/.ssh/authorized_keys
chown vinted:vinted /home/vinted/.ssh/authorized_keys && chmod 600 /home/vinted/.ssh/authorized_keys

# --- 2 GB swap (CX22 has 4 GB RAM; gives headroom) ---
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# from here on, work as the vinted user
su - vinted
```

```bash
# Miniconda (x86_64)
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && exec bash

conda create -y -n vinted_scraper python=3.11
conda activate vinted_scraper

# Clone the repo to the path the units expect
git clone <YOUR_REPO_URL> $HOME/Vinted_New_Version
cd $HOME/Vinted_New_Version
pip install -r requirements.txt
```

x86 = no ARM wheel friction; `numpy`/`pandas`/`scikit-learn`/`xgboost`/`lightgbm` install clean. `basic_5_giant_model` scoring needs **no `torch`** (no images) — skip it on the VPS if it's only pulled by the scraper/visual path.

## 3. Copy secrets + models (once, from laptop)

```bash
# from repo root on the laptop
scp .env vinted@YOUR_VPS_IP:~/Vinted_New_Version/.env

# model pickles the scoring loop loads
rsync -az experiments/current/basic_5_giant_model/data/models/ \
  vinted@YOUR_VPS_IP:~/Vinted_New_Version/experiments/current/basic_5_giant_model/data/models/
```

The `.env` already holds `BOT_TOKEN`, `IMAGE_BOT_TOKEN`, `RECOMMENDED_DEALS_CHAT_ID`, `BOUGHT_ITEMS_CHAT_ID`, `ACCOUNTABILITY_ENABLED`, API keys. The BrightData proxy vars are scraper-only — harmless on the VPS, unused there.

## 4. First data sync (from laptop)

Edit the top of `deploy/sync_to_vps.sh` (`VPS_HOST`, `VPS_REPO`, `SSH_KEY`), then:

```bash
chmod +x deploy/sync_to_vps.sh
# create the target dir on the VPS once if your rsync lacks --mkpath
ssh vinted@YOUR_VPS_IP 'mkdir -p ~/Vinted_New_Version/data/experiments/time_to_sell/live_runs/bin_collector_20260602_214104'
deploy/sync_to_vps.sh
```

First push ≈ 154 MB (CSV state only; `visual_features/` and `raw_snapshots/` are excluded). Later pushes are incremental deltas.

## 5. Cron the sync (on laptop)

```bash
crontab -e
# every 10 min; logs to /tmp
*/10 * * * * /home/ale/Desktop/vinted/Vinted_New_Version/deploy/sync_to_vps.sh >> /tmp/vinted_sync.log 2>&1
```

Collector collects every 2h, so 10-min sync is plenty fresh.

## 6. Install the systemd --user services (on VPS, as `vinted`)

```bash
# allow user services to run without an active login session
loginctl enable-linger vinted

mkdir -p ~/.config/systemd/user
cp ~/Vinted_New_Version/deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload

systemctl --user enable --now vinted-scoring vinted-bot vinted-image-bot vinted-streamlit
systemctl --user status vinted-scoring --no-pager
journalctl --user -u vinted-scoring -f
```

The units reference `%h/Vinted_New_Version` and `%h/miniconda3/envs/vinted_scraper/bin/python`. If your home, repo path, or env name differ, edit the `.service` files before copying.

## 7. ⚠️ Cutover — move, do not duplicate

Two hard conflicts if laptop and VPS run the same thing at once:

- **One poller per bot token.** Two `bot.py` on the same token → `Conflict: terminated by other getUpdates`. The bot silently breaks.
- **Two scoring loops with `--send-telegram`** → every deal sent **twice** to your chat.

So stop the laptop copies. On the laptop, kill the local scoring loop + bots + streamlit (the collector daemon stays running):

```bash
pkill -f 'apply_to_live_collector.py .*--run-loop'      # local scoring loop
pkill -f 'telegram_implementation/bot.py'               # local bot
pkill -f 'telegram_implementation/image_bot.py'         # local image bot
pkill -f 'streamlit run app.py'                         # local dashboard
# leave the collector (run-loop / live_bin_collector) ALONE
```

Order: stop the **local bot first**, then `systemctl --user start vinted-bot` on the VPS, so the token is never polled by two processes.

## 8. Reach the dashboard (SSH tunnel, no public port)

```bash
ssh -N -L 8501:localhost:8501 vinted@YOUR_VPS_IP
# then open http://localhost:8501 on the laptop
```

## 9. Verify

```bash
# VPS: all four active
systemctl --user is-active vinted-scoring vinted-bot vinted-image-bot vinted-streamlit
# VPS: scoring actually progressing
journalctl --user -u vinted-scoring -n 30 --no-pager
# VPS: memory headroom under load
free -h
# laptop: collector still local, bots/scoring/streamlit gone
ps -ef | grep -E 'live_bin_collector|apply_to_live_collector|telegram_implementation|streamlit' | grep -v grep
# laptop: sync ran
tail /tmp/vinted_sync.log
```

## Rollback

Stop VPS services, restart the local ones:

```bash
# VPS
systemctl --user disable --now vinted-scoring vinted-bot vinted-image-bot vinted-streamlit
# laptop: restart local scoring/bots/streamlit as before
```
