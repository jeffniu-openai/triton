#!/usr/bin/env python3
"""Run the signed-i8 MMAv5 artifact on a CaaS GB200 container.

Prerequisites:
  - `cd ~/code/openai && oaipkg install caas`
  - `CAAS_API_KEY` exported in the environment

The token is intentionally read only from the environment and is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def add_openai_paths(openai_root: Path) -> None:
    sys.path.insert(0, str(openai_root / "lib/caas"))
    sys.path.insert(0, str(openai_root / "lib/caas/examples"))


async def run(args: argparse.Namespace) -> None:
    if not os.environ.get("CAAS_API_KEY"):
        raise SystemExit("CAAS_API_KEY must be set in the environment")

    openai_root = Path(args.openai_root).expanduser().resolve()
    add_openai_paths(openai_root)

    from examples_endpoint import resolve_example_caas_endpoint
    from caas.api import caas_api
    from caas.commands import RawExec
    from caas.protocol import (
        ComposeSessionRequest,
        ComposeSessionSpec,
        ContainerSpec,
        ContainerSpecAttachedGpuSpec,
        GpuAttachmentSpec,
        GpuType,
        LifecycleSpec,
        NetworkMode,
        ResourceLimits,
        ResourceReservations,
        SandboxRuntime,
    )

    async def raw(session, cmd: str, *, timeout: int = 120, workdir: str | None = None) -> tuple[int, str]:
        status, output = await session.run(
            RawExec(
                ["bash", "-lc", cmd],
                timeout=timeout,
                workdir=workdir,
                enable_public_logging=False,
            )
        )
        text = output.decode("utf-8", errors="replace")
        print(f"$ {cmd}")
        print(f"status={status}")
        print(text)
        return status, text

    async def create_session(api):
        gpu_names = [f"gpu{i}" for i in range(args.gpus)]
        spec = ContainerSpec(
            image=args.image,
            cmd=["/bin/sh", "-lc", "tail -f /dev/null"],
            default_network=NetworkMode.CAAS_DEFAULT,
            reservations=ResourceReservations(cpu=args.cpu_reservation, memory=args.memory_reservation),
            limits=ResourceLimits(cpu=args.cpu_limit, memory=args.memory_limit, disk=args.disk_limit),
            attached_gpus=[ContainerSpecAttachedGpuSpec(gpu=name) for name in gpu_names],
            sandbox_runtime=SandboxRuntime.UNSAFE,
            lifecycle=LifecycleSpec(ttl=args.ttl),
            env={},
        )
        request = ComposeSessionRequest(
            spec=ComposeSessionSpec(
                network_specs={},
                gpu_specs={name: GpuAttachmentSpec(gpu_type=GpuType.GB200) for name in gpu_names},
                container_specs={"main": spec},
                root_container="main",
                ttl=args.ttl,
            )
        )
        return await api.compose_session(request=request, timeout=args.create_timeout)

    endpoint = resolve_example_caas_endpoint(cluster=args.cluster)
    print(f"endpoint={endpoint}")
    print(f"image={args.image}")
    print(f"gpus={args.gpus}")
    api = caas_api(endpoint=endpoint)

    print("Creating CaaS session...")
    session = await create_session(api)
    print(f"session_id={session.id}")
    try:
        await raw(
            session,
            "nvidia-smi --query-gpu=name,compute_cap,pci.bus_id --format=csv,noheader",
            timeout=120,
        )
        await raw(
            session,
            """python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY""",
            timeout=120,
        )

        if args.skip_artifact_run:
            return

        await raw(session, f"rm -rf {args.remote_dir} && mkdir -p {args.remote_dir}", timeout=60)
        for path in sorted(HERE.iterdir()):
            if path.is_file():
                await session.container.write(f"{args.remote_dir}/{path.name}", path.read_bytes(), timeout=60)
                print(f"uploaded {path.name} ({path.stat().st_size} bytes)")

        await raw(session, "python3 run_i8_ptx_torch.py --dry-run", timeout=120, workdir=args.remote_dir)
        command = f"python3 run_i8_ptx_torch.py --launcher {args.launcher} --module {args.module}"
        await raw(session, command, timeout=args.run_timeout, workdir=args.remote_dir)
    finally:
        if args.keep_session:
            print("Keeping CaaS session alive by request. Remember to close it manually.")
        else:
            print("Closing CaaS session...")
            await session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", default="caas-gpu10")
    parser.add_argument("--image", default="cudaberry-arm")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--openai-root", default="~/code/openai")
    parser.add_argument("--remote-dir", default="/tmp/mmav5_i8_remote")
    parser.add_argument("--ttl", type=int, default=1200)
    parser.add_argument("--create-timeout", type=int, default=900)
    parser.add_argument("--run-timeout", type=int, default=360)
    parser.add_argument("--cpu-reservation", default="4")
    parser.add_argument("--memory-reservation", default="8g")
    parser.add_argument("--cpu-limit", default="8")
    parser.add_argument("--memory-limit", default="32g")
    parser.add_argument("--disk-limit", default="20g")
    parser.add_argument("--launcher", choices=["cpp", "ctypes"], default="cpp")
    parser.add_argument("--module", choices=["auto", "cubin", "ptx"], default="auto")
    parser.add_argument("--skip-artifact-run", action="store_true")
    parser.add_argument("--keep-session", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
