#!/bin/bash
set -e

echo "[+] Installing system utilities..."
sudo yum install -y yum-utils || echo "[!] Failed to install yum-utils (try manually)"

echo "[+] Adding Docker repository..."
sudo yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo || echo "[!] Failed to add Docker repo (check network)"

echo "[+] Installing Docker components..."
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin || echo "[!] Failed to install Docker (check dependencies)"

echo "[+] Starting Docker service..."
sudo systemctl start docker || echo "[!] Failed to start Docker (check logs with 'systemctl status docker')"

echo "[+] Configuring NVIDIA container repository..."
curl -sSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo || echo "[!] Failed to configure NVIDIA repo"

echo "[+] Installing NVIDIA Container Toolkit..."
sudo yum install -y nvidia-container-toolkit || echo "[!] Failed to install NVIDIA Container Toolkit (check repo)"

echo "[+] Restarting Docker service..."
sudo systemctl restart docker || echo "[!] Failed to restart Docker (check configuration)"

# These steps are not necessary 
echo
echo "==> Verifying installation (optional)..."

echo "[i] Checking Docker service status..."
systemctl is-active docker | grep -q "active" && echo "[✓] Docker service is running" || echo "[!] Docker service may not be active"

echo "[i] Checking NVIDIA runtime configuration..."
grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json && echo "[✓] NVIDIA runtime is configured" || echo "[!] NVIDIA runtime not found in daemon.json"

echo "[i] Testing GPU support (run manually if needed):"
echo "   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"