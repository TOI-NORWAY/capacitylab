# Dev Container Profiles

This repository includes two profiles:

- `devcontainer.cpu.json`: starts without GPU runtime requirements.
- `devcontainer.gpu.json`: starts with `--gpus all` and NVIDIA container env vars.

## Switch profiles

From the repo root in PowerShell:

```powershell
.\scripts\switch-devcontainer.ps1 cpu
# or
.\scripts\switch-devcontainer.ps1 gpu
```

Then in VS Code, run:

- Dev Containers: Rebuild and Reopen in Container

## Notes

Use the GPU profile only after host-side GPU passthrough is working for Docker Desktop + WSL.
