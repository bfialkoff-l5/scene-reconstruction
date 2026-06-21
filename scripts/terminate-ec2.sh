#!/usr/bin/env bash
# Terminate the box launched by launch-ec2.sh (reads the saved instance id, or pass one).
# Terminate = gone for good + billing stops. There is no "oops" -- the EBS root goes too.
set -euo pipefail

PROFILE="${AWS_PROFILE:-bfialkoff}"
REGION="${AWS_REGION:-eu-north-1}"
IID_FILE="$(cd "$(dirname "$0")/.." && pwd)/.ec2-instance-id"
IID="${1:-$(cat "$IID_FILE" 2>/dev/null || true)}"
[ -n "$IID" ] || { echo "no instance id (pass as arg, or run launch-ec2.sh first)"; exit 1; }

echo "Terminating $IID ..."
aws --profile "$PROFILE" --region "$REGION" ec2 terminate-instances --instance-ids "$IID" >/dev/null
aws --profile "$PROFILE" --region "$REGION" ec2 wait instance-terminated --instance-ids "$IID"
rm -f "$IID_FILE"
echo "terminated. billing stopped."
