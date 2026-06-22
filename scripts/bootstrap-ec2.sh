#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu EC2 GPU instance.
# Host installs Docker + NVIDIA Container Toolkit only.
#
# Usage: ./scripts/bootstrap-ec2.sh

set -euo pipefail

echo "==> Docker"
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg git
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "Log out and back in for docker group membership."
fi

echo "==> NVIDIA Container Toolkit (GPU instances)"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
  nvidia-smi || true
else
  echo "No nvidia-smi — CPU-only ODM will be very slow."
fi

echo "==> /data (the EC2 DATA_ROOT) owned by you"
# The 300GB EBS root mounts at /, so /data is just a dir that starts out root-owned.
# Hand it to the login user so non-sudo aws/fetch-data.sh and the builder container (which
# runs as your uid) can write. Never run the data scripts with sudo -- root has no SSO profile.
sudo mkdir -p /data
sudo chown "$(id -u):$(id -g)" /data

echo "==> AWS SSO config (~/.aws/config)"
# The repo owns this box's AWS config so setup is reproducible -- no interactive
# `aws configure sso`. The one subtlety that bit us: sso_region is where IAM Identity
# Center lives (us-east-1), NOT where our data/S3 lives (the profile's region, eu-north-1).
# Setting sso_region to eu-north-1 makes `aws sso login` fail at RegisterClient with
# InvalidRequestException. We write the file authoritatively (backing up any existing one),
# since this is a single-purpose disposable box.
AWS_CFG="$HOME/.aws/config"
mkdir -p "$HOME/.aws"
[ -f "$AWS_CFG" ] && cp "$AWS_CFG" "$AWS_CFG.bak"
cat > "$AWS_CFG" <<'EOF'
[sso-session okta-sso]
sso_start_url = https://d-9066027ed5.awsapps.com/start
sso_region = us-east-1
sso_registration_scopes = sso:account:access

[profile bfialkoff]
sso_session = okta-sso
sso_account_id = 939103584914
sso_role_name = PowerUserAccess
region = eu-north-1
output = json
EOF
echo "wrote $AWS_CFG (backup at $AWS_CFG.bak if one existed)"

echo ""
echo "Done. Next:"
echo "  aws sso login --profile bfialkoff --no-browser   # only human step: approve in browser"
echo "  export DATA_ROOT=/data"
echo "  docker compose build"
echo "  ./scripts/build.sh <slug>"
