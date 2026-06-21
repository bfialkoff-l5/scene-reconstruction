#!/usr/bin/env bash
# STOP (not terminate) the box: powers it off but KEEPS the EBS disk and everything on it
# (repo, /data, ODM outputs). GPU compute billing stops; you still pay ~$0.80/day for the
# idle 300GB EBS. Resume later with start-ec2.sh. Use terminate-ec2.sh only to destroy.
set -euo pipefail

PROFILE="${AWS_PROFILE:-bfialkoff}"
REGION="${AWS_REGION:-eu-north-1}"
IID_FILE="$(cd "$(dirname "$0")/.." && pwd)/.ec2-instance-id"
IID="${1:-$(cat "$IID_FILE" 2>/dev/null || true)}"
[ -n "$IID" ] || { echo "no instance id (pass as arg, or launch first)"; exit 1; }

echo "Stopping $IID (files persist on EBS) ..."
aws --profile "$PROFILE" --region "$REGION" ec2 stop-instances --instance-ids "$IID" >/dev/null
aws --profile "$PROFILE" --region "$REGION" ec2 wait instance-stopped --instance-ids "$IID"
echo "stopped. GPU billing off; EBS still ~\$0.80/day. Resume: ./scripts/start-ec2.sh"
