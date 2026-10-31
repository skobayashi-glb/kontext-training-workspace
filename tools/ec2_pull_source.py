"""Copy approved source files from the existing RUNNING EC2. Never starts EC2."""

import argparse
import hashlib
import io
import json
from pathlib import PurePosixPath
import subprocess
import sys
import tarfile
import time

from ec2_inspect import INSTANCE, OPENSSH, ROOT, instance, trusted_alias


REMOTE_ROOT = "/home/ec2-user/workspace/flux-kontext-training"
DESTINATION = ROOT / "flux-kontext-training"
REMOTE_SCRIPT = r'''
import io, pathlib, sys, tarfile
root = pathlib.Path('/home/ec2-user/workspace/flux-kontext-training')
files = []
allowed = {'.py', '.md', '.toml', '.json', '.ts', '.tsx', '.js', '.jsx', '.css', '.html', '.yaml', '.yml', '.txt', '.sh'}
for name in ['AGENTS.md', 'README.md', 'pyproject.toml', 'main.py', '.python-version', '.gitignore']:
    path = root / name
    if path.is_file() and not path.is_symlink():
        files.append(path)
for directory in ['src/flux_kontext_training', 'ui/src', 'ui/public']:
    for path in (root / directory).rglob('*'):
        if (path.is_file() and not path.is_symlink() and path.suffix in allowed
                and '__pycache__' not in path.parts
                and not any(p.startswith('.env') for p in path.parts)):
            files.append(path)
for name in ['ui/package.json', 'ui/package-lock.json', 'ui/tsconfig.json', 'ui/vite.config.ts', 'ui/index.html', 'datasets/default_prompts.json']:
    path = root / name
    if path.is_file() and not path.is_symlink():
        files.append(path)
if sum(p.stat().st_size for p in files) > 15000000:
    raise RuntimeError('Unexpected code size; transfer refused')
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for path in sorted(set(files)):
        archive.add(str(path), arcname=str(path.relative_to(root)), recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''


def pull():
    if DESTINATION.exists():
        raise RuntimeError("Destination exists; refusing to overwrite: " + str(DESTINATION))
    alias = trusted_alias()
    current = instance()
    if current["State"]["Name"] != "running" or not current.get("PublicIpAddress"):
        raise RuntimeError("EC2 is not running with a public IP. It was NOT started.")
    ssh = [str(OPENSSH / "ssh.exe"), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "ConnectTimeout=5", "-o", "ConnectionAttempts=1",
           "-o", "HostName=" + current["PublicIpAddress"], "-o", "HostKeyAlias=" + alias,
           "samurai-AWS", "python3 -"]
    result = subprocess.run(ssh, input=REMOTE_SCRIPT.encode("utf-8"), capture_output=True, timeout=35)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    report = ROOT / ".local/source-copy" / time.strftime("%Y%m%d-%H%M%S")
    report.mkdir(parents=True, exist_ok=False)
    (report / "source.tar.gz").write_bytes(result.stdout)
    manifest = []
    contents = []
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or "\\" in member.name or ":" in member.name):
                raise RuntimeError("Unexpected archive member: " + member.name)
            if member.size > 15000000 or sum(item[2] for item in contents) + member.size > 15000000:
                raise RuntimeError("Unexpected archive size")
            content = archive.extractfile(member).read()
            contents.append((name, content, member.size))
            manifest.append({"path": member.name, "sha256": hashlib.sha256(content).hexdigest(),
                             "bytes": len(content)})
    # Validate all archive entries before creating anything in the project.
    staged = report / "staged"
    staged.mkdir()
    for name, content, _ in contents:
        target = staged.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (report / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if DESTINATION.exists():
        raise RuntimeError("Destination appeared during transfer; staged copy retained in " + str(report))
    staged.rename(DESTINATION)
    print(f"Copied {len(manifest)} source/config files to {DESTINATION}")
    print("Manifest: " + str(report / "manifest.json"))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Show the plan without AWS/SSH access")
    group.add_argument("--pull", action="store_true", help="Copy source from the already running instance")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Instance: {INSTANCE}\nRemote: {REMOTE_ROOT}\nLocal: {DESTINATION}")
        print("Include: Python/UI text sources, project metadata, default prompt settings")
        print("Exclude: images, weights, datasets, runs, virtualenvs, secrets, Git history")
        print("EC2 start/stop: never. Existing local project files: never overwritten.")
    else:
        pull()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.SubprocessError, OSError, tarfile.TarError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
