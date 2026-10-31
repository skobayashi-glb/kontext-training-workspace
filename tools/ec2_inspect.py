"""Inspect the existing training EC2 with a bounded lifetime; never start training.

Run explicitly with --start. AWS login and an existing trusted SSH host key are
required. Temporary reports live in .local/ec2-inspection/ (no credentials).
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


INSTANCE = "i-02c9d1450ba6014f4"
REGION = "ap-northeast-1"
ROOT = Path(__file__).resolve().parents[1]
AWS = shutil.which("aws")
OPENSSH = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/OpenSSH"


def aws(*args, timeout=35):
    command = [AWS, "ec2", *args, "--region", REGION, "--profile", "default",
               "--no-cli-pager", "--output", "json", "--cli-connect-timeout", "5",
               "--cli-read-timeout", "15"]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def instance():
    return aws("describe-instances", "--instance-ids", INSTANCE)["Reservations"][0]["Instances"][0]


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def stop_and_wait(report):
    for attempt in range(75):
        try:
            state = instance()["State"]["Name"]
            if attempt % 5 == 0:
                print("EC2 state: " + state, flush=True)
            if state == "stopped":
                (report / "stopped.txt").write_text("stopped\n", encoding="utf-8")
                print("EC2 state: stopped", flush=True)
                return True
            if state not in {"stopping", "shutting-down", "terminated"}:
                aws("stop-instances", "--instance-ids", INSTANCE)
                print("EC2 stop requested", flush=True)
            elif state in {"shutting-down", "terminated"}:
                raise RuntimeError("Unexpected instance lifecycle state: " + state)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"Stop check {attempt + 1}: {exc}", flush=True)
        time.sleep(5)
    return False


def watchdog(report):
    """Independent local fallback if the inspecting process is interrupted."""
    state_path = report / "session.json"
    while not (report / "stopped.txt").exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        remaining = state["stop_at"] - time.time()
        if remaining <= 0:
            if state["start_attempted"]:
                stop_and_wait(report)
            return
        time.sleep(min(remaining, 3))


def trusted_alias():
    result = subprocess.run([str(OPENSSH / "ssh.exe"), "-G", "samurai-AWS"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, timeout=10)
    settings = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
    # Reuse the already trusted alias or previous address; never accept an
    # unverified replacement key, even when the public IP has changed.
    candidates = [settings.get("hostkeyalias"), "samurai-AWS", settings.get("hostname")]
    for alias in dict.fromkeys(x for x in candidates if x):
        lookup = f"[{alias}]:{settings['port']}" if settings.get("port", "22") != "22" else alias
        found = subprocess.run([str(OPENSSH / "ssh-keygen.exe"), "-F", lookup],
                               capture_output=True, timeout=10)
        if found.returncode == 0:
            return alias
    raise RuntimeError("No existing trusted SSH host key. Verify the instance host key before starting.")


DISCOVERY = r"""
set -u
printf '\n=== OS ===\n'
cat /etc/os-release
printf '\n=== USER AND HOME ===\n'
id
printf '%s\n' "$HOME"
printf '\n=== STORAGE ===\n'
df -h -T
printf '\n=== HOME ENTRIES ===\n'
find "$HOME" -mindepth 1 -maxdepth 2 -type d ! -path '*/.*' 2>/dev/null | head -n 100
printf '\n=== TRAINING PROJECT CANDIDATES (paths only) ===\n'
for base in "$HOME" /opt /workspace /workspaces /data /mnt /home; do
  [ -d "$base" ] || continue
  timeout 25 find "$base" -maxdepth 5 \( -name .git -o -name .cache -o -name node_modules -o -name site-packages -o -name pkgs \) -prune -o -type f \( -iname '*kontext*' -o -name 'train.py' -o -name 'train*.sh' -o -name 'pyproject.toml' -o -name 'environment.yml' -o -name 'requirements.txt' \) -print 2>/dev/null | head -n 160
done
printf '\n=== GPU ===\n'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv 2>/dev/null || true
printf '\n=== GPU PROCESSES ===\n'
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null || true
printf '\n=== EXECUTABLES ===\n'
command -v python python3 conda uv accelerate systemctl systemd-run || true
printf '\n=== PYTHON ENVIRONMENT PATHS ===\n'
for base in "$HOME" /opt; do
  timeout 15 find "$base" -maxdepth 5 \( -name .cache -o -name pkgs -o -name site-packages \) -prune -o -type f \( -name pyvenv.cfg -o -path '*/conda-meta/history' \) -print 2>/dev/null | head -n 80
done
"""


def inspect(stop_at=None, copy_source=False, ready_hook=None):
    alias = trusted_alias()
    current = instance()
    if current["State"]["Name"] != "stopped":
        raise RuntimeError("Inspection requires a stopped instance; current state: " + current["State"]["Name"])
    behavior = aws("describe-instance-attribute", "--instance-id", INSTANCE,
                   "--attribute", "instanceInitiatedShutdownBehavior")
    if current.get("RootDeviceType") != "ebs" or behavior["InstanceInitiatedShutdownBehavior"]["Value"] != "stop":
        raise RuntimeError("EBS root and shutdown behavior=stop are required. No settings were changed.")

    token = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    report = ROOT / ".local/ec2-inspection" / token
    report.mkdir(parents=True)
    # Reserve nine minutes for the slow shutdown observed during inspection.
    deadline = min(time.time() + 6 * 60, stop_at or float("inf"))
    if deadline - time.time() < 240:
        raise RuntimeError("Not enough time remains for a bounded inspection; instance was not started.")
    state = {"instance_id": INSTANCE, "stop_at": deadline,
             "start_attempted": False, "host_key_alias": alias}
    save(report / "session.json", state)
    with (report / "watchdog.log").open("w", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watchdog", str(report)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    print("Report directory: " + str(report), flush=True)
    try:
        # Set this BEFORE the API: a timed-out StartInstances may still succeed.
        state["start_attempted"] = True
        save(report / "session.json", state)
        aws("start-instances", "--instance-ids", INSTANCE)
        print("Start requested; 6-minute stop deadline armed", flush=True)
        boot_deadline = time.monotonic() + 120
        while time.monotonic() < boot_deadline:
            current = instance()
            if current["State"]["Name"] == "running" and current.get("PublicIpAddress"):
                break
            time.sleep(5)
        else:
            raise RuntimeError("Boot timeout")
        state["public_ip"] = current["PublicIpAddress"]
        save(report / "session.json", state)
        ssh = [str(OPENSSH / "ssh.exe"), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=5", "-o", "ConnectionAttempts=1",
               "-o", "HostName=" + current["PublicIpAddress"], "-o", "HostKeyAlias=" + alias,
               "samurai-AWS"]
        while time.monotonic() < boot_deadline:
            ready = subprocess.run(ssh + ["true"], capture_output=True, timeout=15)
            if ready.returncode == 0:
                break
            if b"HOST IDENTIFICATION HAS CHANGED" in ready.stderr or b"Host key verification failed" in ready.stderr:
                raise RuntimeError("SSH host key verification failed; stopping without accepting a new key.")
            time.sleep(5)
        else:
            raise RuntimeError("SSH timeout: " + ready.stderr.decode(errors="replace"))

        remaining = max(1, int(state["stop_at"] - time.time()))
        timer = f"test -x /usr/bin/systemctl && sudo -n systemd-run --unit=kontext-inspect-stop-{token} --on-active={remaining}s /usr/bin/systemctl poweroff"
        result = subprocess.run(ssh + [timer], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        if result.returncode:
            raise RuntimeError("Remote shutdown timer could not be armed: " + result.stderr)
        print("Remote shutdown timer armed; inspecting files and environment only", flush=True)
        if copy_source:
            subprocess.run([sys.executable, str(ROOT / 'tools/ec2_pull_source.py'), '--pull'],
                           check=True, timeout=60)
        if ready_hook:
            ready_hook(ssh, report)
        # Send bytes: Windows text-mode pipes otherwise convert LF to CRLF,
        # which bash interprets as part of commands and file names.
        result = subprocess.run(ssh + ["bash -s"], input=DISCOVERY.encode("utf-8"),
                                capture_output=True, timeout=min(150, remaining - 30))
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        (report / "discovery.txt").write_text(stdout + stderr, encoding="utf-8")
        print(stdout, flush=True)
        if result.returncode:
            print("Discovery returned an error; see discovery.txt: " + stderr, flush=True)
        print("Inspection ready. Create finish.txt in the report directory to stop now.", flush=True)
        # Leave a short window for follow-up reads, within both stop deadlines.
        followup_deadline = min(state["stop_at"] - 60, time.time() + 90)
        while time.time() < followup_deadline and not (report / "finish.txt").exists():
            time.sleep(2)
    finally:
        print("Finishing inspection; stopping EC2", flush=True)
        if not stop_and_wait(report):
            raise RuntimeError("Stop not yet confirmed. Check AWS immediately; independent fallback remains armed.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="Start the existing EC2 for a short inspection")
    group.add_argument("--watchdog", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stop-at", type=float, help="An earlier absolute stop deadline, in Unix seconds")
    parser.add_argument("--copy-source", action="store_true", help="Copy the already approved source immediately after SSH is ready")
    args = parser.parse_args()
    try:
        if not AWS:
            raise RuntimeError("AWS CLI is required")
        if args.watchdog:
            watchdog(args.watchdog)
        else:
            inspect(args.stop_at, args.copy_source)
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        print(str(error), file=sys.stderr, flush=True)
        sys.exit(1)
