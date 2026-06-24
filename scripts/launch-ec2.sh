#!/usr/bin/env bash
# Launch the disposable scene-reconstruction GPU box on EC2. KISS: one instance,
# pulls code + data itself, terminate when done. The key pair + security group were
# created once already (vars below); this only does run-instances + wait + print.
#
# Refresh AMI id when it goes stale (region-specific, versioned):
#   aws --profile bfialkoff --region eu-north-1 ssm get-parameters \
#     --names /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
#     --query 'Parameters[0].Value' --output text
set -euo pipefail

PROFILE="${AWS_PROFILE:-bfialkoff}"
REGION="${AWS_REGION:-eu-north-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.4xlarge}"
AMI="${AMI:-ami-095ffa7a2ada3d9ca}"      # Deep Learning AMI (Ubuntu 22.04, NVIDIA driver + Docker preinstalled)
KEY_NAME="${KEY_NAME:-scene-recon-eitan}"
SG_ID="${SG_ID:-sg-0ff68353813837568}"
DISK_GB="${DISK_GB:-300}"                 # root gp3; dense + true-ortho is storage-heavy. Bump if a run runs out.
NAME="${NAME:-scene-recon-eitan}"
# REQUIRED by the account's ec2-vcpu-limiter Lambda: an instance with no `owner` tag (or
# an owner over their per-owner vcpu/gpu budget) is auto-stopped seconds after launch.
# betzalel's budget is 16 vCPU / 1 GPU -> g6.4xlarge fits exactly, with zero headroom, so
# don't run any other owner=betzalel instance at the same time.
OWNER="${OWNER:-betzalel}"
# Per-bucket S3 instance role (scoped read-write to the eval-data bucket only). PassRole is
# granted to power users, so we attach it ourselves at launch -> the box reads/writes S3 via
# IMDS with no `aws sso login`. Set IAM_PROFILE="" to launch without it (fall back to SSO).
IAM_PROFILE="${IAM_PROFILE:-l5-localization-evaluation-data-rw}"
IID_FILE="$(cd "$(dirname "$0")/.." && pwd)/.ec2-instance-id"

aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }

# Default VPC: omitting --subnet-id lets AWS pick a default subnet that auto-assigns a
# public IP. SG must live in that default VPC (it does).
echo "Launching $INSTANCE_TYPE ($AMI, ${DISK_GB}GB) in $REGION ..."
iam_arg=(); [ -n "$IAM_PROFILE" ] && iam_arg=(--iam-instance-profile "Name=$IAM_PROFILE")
IID=$(aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=owner,Value=$OWNER}]" \
  "${iam_arg[@]}" \
  --query 'Instances[0].InstanceId' --output text)
echo "$IID" > "$IID_FILE"
echo "instance: $IID  (id saved to $IID_FILE)"

echo "Waiting for running ..."
aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# Rewrite a managed ~/.ssh/config block so `ssh scene-recon` (and VS Code Remote-SSH ->
# "scene-recon") always hit the latest IP. ForwardAgent lends our github key for cloning;
# the disposable box's host key changes each launch, so don't pin it (accept-new + no
# known_hosts churn) to avoid VS Code's "HOST KEY CHANGED" hard stop.
ALIAS="${SSH_ALIAS:-scene-recon}"
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

cat <<EOF

Up. Public IP: $IP   (ssh alias: $ALIAS)

0) VS Code: Remote-SSH -> "Connect to Host" -> $ALIAS  (familiar editor on the box)

1) Or plain SSH (agent-forwarded github key for cloning the PRIVATE repo):
     ssh $ALIAS

2) On the box, get the code (already current on origin/main):
     git clone git@github.com:bfialkoff-l5/scene-reconstruction.git
     cd scene-reconstruction && ./scripts/bootstrap-ec2.sh   # near no-op on the DLAMI

3) S3 access: the box has the instance role ${IAM_PROFILE:-(none — IAM_PROFILE was empty)}
   attached, so \`aws s3 ...\` on the eval-data bucket just works (creds via IMDS, no login).
   Pull the dataset:  ./scripts/fetch-data.sh 0088_20260122_eitan_1 Eitan.gpkg
   (If you launched with IAM_PROFILE="", fall back to: aws sso login --profile bfialkoff --no-browser)

4) Terminate when done (STOPS billing -- the GPU box is ~\$1-2/hr):
     ./scripts/terminate-ec2.sh
EOF
