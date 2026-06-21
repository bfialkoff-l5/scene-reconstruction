#!/usr/bin/env bash
# Resume the SAME stopped box (files intact). Not "launch" -- that makes a new instance.
# The public IP changes on every stop/start, so this refreshes the ~/.ssh/config alias.
set -euo pipefail

PROFILE="${AWS_PROFILE:-bfialkoff}"
REGION="${AWS_REGION:-eu-north-1}"
KEY_NAME="${KEY_NAME:-scene-recon-eitan}"
ALIAS="${SSH_ALIAS:-scene-recon}"
IID_FILE="$(cd "$(dirname "$0")/.." && pwd)/.ec2-instance-id"
IID="${1:-$(cat "$IID_FILE" 2>/dev/null || true)}"
[ -n "$IID" ] || { echo "no instance id (pass as arg, or launch first)"; exit 1; }

aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }

echo "Starting $IID ..."
aws ec2 start-instances --instance-ids "$IID" >/dev/null
aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

CFG="$HOME/.ssh/config"; touch "$CFG"; chmod 600 "$CFG"
python3 - "$CFG" "$ALIAS" "$IP" "$KEY_NAME" <<'PY'
import re, sys
cfg, alias, ip, key = sys.argv[1:5]
block = (f"# >>> launch-ec2.sh {alias} >>>\n"
         f"Host {alias}\n  HostName {ip}\n  User ubuntu\n"
         f"  IdentityFile ~/.ssh/{key}.pem\n  ForwardAgent yes\n"
         f"  StrictHostKeyChecking accept-new\n  UserKnownHostsFile /dev/null\n"
         f"# <<< launch-ec2.sh {alias} <<<\n")
txt = open(cfg).read()
pat = re.compile(rf"# >>> launch-ec2\.sh {re.escape(alias)} >>>.*?# <<< launch-ec2\.sh {re.escape(alias)} <<<\n", re.S)
open(cfg, "w").write(pat.sub(block, txt) if pat.search(txt) else (txt.rstrip()+"\n\n"+block if txt.strip() else block))
PY

echo "running at $IP (ssh alias: $ALIAS)."
echo "if SSH/VS Code hangs and you're on a new network, run ./scripts/allow-my-ip.sh first."
