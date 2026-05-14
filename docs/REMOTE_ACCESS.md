# Remote Access

You can work on this project from another computer by connecting to this machine remotely.

## Important Concept

If you SSH into this computer and run the scraper here, the process runs on this computer.

That means:

- CPU/RAM/disk used are from this computer.
- Files are read/written on this computer.
- Internet traffic uses this computer's connection.
- The scraper keeps using this computer's network environment.

## Same Network

If both computers are on the same local network, you can usually connect with:

```bash
ssh ale@<local-ip>
```

Find the local IP on this computer with:

```bash
hostname -I
```

## Different Network

If the computers are not on the same network, using the public IP directly is usually annoying because:

- The public IP may change.
- The router may need port forwarding.
- Some providers use CGNAT, which can block inbound connections.
- Exposing SSH to the internet is risky if not configured carefully.

## Recommended Option: Tailscale

Tailscale creates a private network between your devices. It is usually much easier and safer than opening SSH to the public internet.

On this computer, after Tailscale is installed:

```bash
sudo tailscale up --ssh
```

Then open the login URL, approve the machine, and check:

```bash
tailscale status
tailscale ip -4
```

On the other computer:

1. Install Tailscale.
2. Log in with the same Tailscale account.
3. Connect with one of these:

```bash
tailscale ssh ale@ale-HKD-WXX
```

or:

```bash
tailscale ssh ale@<tailscale-ip>
```

## Keeping Long Jobs Alive

For long scraper runs, use `tmux`, `screen`, or `nohup` so the process does not stop if your SSH session disconnects.

Example:

```bash
tmux new -s vinted
```

Then run:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/main.py
```

Detach from tmux with:

```text
Ctrl-b then d
```

Reconnect later:

```bash
tmux attach -t vinted
```

