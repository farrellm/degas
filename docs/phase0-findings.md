# Phase 0 — Colab CLI spike findings

Date: 2026-09-27 · `google-colab-cli` 0.7.4 · Scripts: `spike/`

Everything was measured from the home server. Most tests used a CPU runtime, and GPU-specific checks used a T4.

## Answers to the §3.3 questions

| Question | Answer |
|---|---|
| Do consecutive `exec` calls share one kernel? | **Yes.** Globals survive between calls; this is documented and confirmed. |
| Does `exec` stream stdout? | **Yes**, line by line in real time (1 s prints arrived 1.00 s apart through a pipe). |
| Fixed overhead per call | Warm `exec` takes 1.3–1.5 s; the first few calls took up to 14 s. `ls`/`status`/`download` take about 0.9 s. `upload` takes about 1.2 s. Throughput is about 15 MB/s. |
| Can upload/download run during an `exec`? | **Yes**, at normal speed (about 0.95 s). A job cancelled itself within about 0.3 s of the cancel-flag file being uploaded. |
| VM reclaimed or stopped | `exec` on a stopped session fails cleanly: `Session '…' not found`, exit 1, 0.6 s. Behaviour when the kernel sits idle is in the liveness test below. |
| Copy speed from Drive to local disk | Using rclone (below), a cold 1 GB file copied in 15.8 s, **about 65 MB/s**. That is about 100 s for a 6.5 GB SDXL checkpoint and about 7 min for Wan A14B (about 28 GB). The Drive FUSE mount was not benchmarked, because it can't be automated (below). |

## Surprises

1. **`drivemount` requires interactive consent on every new VM.** The consent URL is tied to the VM (it carries `prompt=consent` and the VM endpoint in `state`). On a fresh VM the mount failed with HTTP 400 when nobody approved it. The CLI then waits on `/dev/tty`. Automatic session start cannot use the Drive FUSE mount.
2. **Killing the `exec` client does not stop the kernel's code.** The code keeps running, and the next `exec` waits until it finishes. The only ways to stop running code are cooperative cancellation or `colab restart-kernel`, which also drops the loaded pipelines.
3. **`exec --timeout` has no effect.** Both a job that printed output and a silent `sleep` ran past it.
4. **`exec` exits with 0 even when the code raises.** Tracebacks are printed with ANSI colour codes. Success has to be judged from structured events, not from the exit code.
5. **`colab ssh` exists** and works as an OpenSSH `ProxyCommand`. Connecting takes about 1.4 s. Over a multiplexed (ControlMaster) connection, a command takes about 0.17 s, and an HTTP request through `ssh -L` takes **about 0.12 s** round trip. SSE through the tunnel streams in real time. `scp` runs at about 12 MB/s. Background processes started over SSH survive the SSH connection closing.
6. **SSH shells lack the kernel's environment.** CUDA is invisible over SSH until `LD_LIBRARY_PATH=/usr/lib64-nvidia` is set. A process started from the kernel (`subprocess.Popen` inside an `exec`) inherits the correct environment.
7. **The runtime image already has nearly everything:**
   - torch 2.11 (cu128 on GPU), diffusers 0.40, transformers 5.16, accelerate, peft, safetensors, opencv, pillow;
   - fastapi, uvicorn and ffmpeg 6.1.

   `uv pip install` of these packages was a no-op (0.5 s). `colab new` took 3.5 s for CPU.
8. **Small standard shapes.** A CPU VM and a T4 standard VM each have 2 vCPUs and 12 GB of RAM, with about 190–206 GB of free disk. CPU offload of large models (Wan A14B) needs `--high-mem`.
9. Your rclone remote uses rclone's **shared client_id, which is being retired during 2026**. Degas should use its own OAuth client.

## Liveness test (idle kernel, SSH-launched worker only)

This test checks whether Colab keeps a VM alive when the kernel has been idle since 15:43 and the only activity is HTTP traffic over SSH. See `spike/liveness.log` for the raw log. Result: **inconclusive.** The VM was still alive and the worker still answered 15 minutes after the kernel's last activity. The test was then stopped early. Colab's idle reclaim window is probably longer than that, so this question is carried into Phase 1: the session manager should include the optional `exec` heartbeat (design §3.1), and we should watch for reclaims in real use.

## Recommendations (applied to design.md)

1. **Transport: an HTTP worker behind an SSH tunnel, not per-job `exec`.**
   - A single `exec` starts the worker (`uvicorn`, bound to `127.0.0.1` on the VM) from the kernel, so the worker inherits the CUDA environment.
   - The server keeps one ControlMaster SSH connection with `-L` forwarding to the worker.
   - Jobs, progress (SSE), cancel, file transfer, preprocessing and SAM all run over HTTP, at about 0.12 s per call instead of 1.3 s.
   - Cancel becomes a plain HTTP call.
   - Re-attaching after a server restart becomes a matter of reconnecting the tunnel and calling `GET /state`.
2. **Drive: no FUSE mount.** The server holds a Drive OAuth refresh token from Degas's own Google Cloud OAuth client, and pushes short-lived access tokens to the worker. The worker copies assets with rclone (the binary is uploaded at bootstrap) or the Drive API. Asset indexing uses the Drive API from the server itself, so no session is needed to rescan.
3. **Use `exec` only for bootstrap and as a heartbeat**, if the liveness test shows one is needed. Never rely on its exit code or `--timeout`.
4. **Cooperative cancellation only.** The step callback checks a cancel flag. `restart-kernel` is a last resort, shown in the UI as "Force reset worker".
5. Request `--high-mem` by default for Wan A14B sessions.
