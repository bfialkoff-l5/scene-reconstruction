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

echo ""
echo "Done. Next:"
echo "  export DATA_ROOT=/data"
echo "  docker compose build"
echo "  ./scripts/build.sh <slug>"
