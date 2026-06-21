#!/usr/bin/env bash
# Allow SSH (port 22) from THIS network's current public IP. Run once per network
# (office, home, phone hotspot). Hotspot/home IPs change, so re-run when SSH hangs.
# ponytail: leaves stale /32 rules behind on IP change; harmless, prune with
#   aws ec2 describe-security-group-rules --filters Name=group-id,Values=$SG_ID
set -euo pipefail

PROFILE="${AWS_PROFILE:-bfialkoff}"
REGION="${AWS_REGION:-eu-north-1}"
SG_ID="${SG_ID:-sg-0ff68353813837568}"

IP="$(curl -fsS https://checkip.amazonaws.com)"
echo "this network's IP: $IP  ->  $SG_ID port 22"

if err="$(command aws --profile "$PROFILE" --region "$REGION" \
    ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$IP/32" 2>&1)"; then
  echo "added."
elif grep -q 'InvalidPermission.Duplicate' <<<"$err"; then
  echo "already allowed."
else
  echo "$err" >&2; exit 1
fi
