# Scene Reconstruction

Manual two-step pipeline: **build** (score all frames → select keyframes → ODM layout) then **odm** (stock OpenDroneMap).

See [docs/PLAN.md](docs/PLAN.md) and [docs/BUILDER.md](docs/BUILDER.md).

## Quick start

```bash
cp .env.example .env   # set DATA_ROOT
# datum at $DATA_ROOT/raw/{slug}/

docker compose build
./scripts/build.sh /home/you/s3/raw/0088_20260122_eitan_1
# or slug shorthand:
./scripts/build.sh 0088_20260122_eitan_1

./scripts/run_odm.sh 0088_20260122_eitan_1 -- --fast-orthophoto
```

Builder scores **every** pose frame once per slug (cached at `odm-results/{slug}/candidates_scored.csv`), then writes timestamped runs under `odm-results/{slug}/runs/{ts}/`. Selection uses path-walk connectivity and fails the build if health checks do not pass. Re-select without re-scoring:

```bash
./scripts/build.sh 0088_20260122_eitan_1 --select-only
./scripts/run_odm.sh 0088_20260122_eitan_1 -- --fast-orthophoto
```

Output images are `{FrameNumber:06d}.png`.

## File ownership

Containers run as your host user (`HOST_UID`/`HOST_GID`). Always use `./scripts/build.sh` and `./scripts/run_odm.sh`.

```bash
./scripts/clean-root-artifacts.sh   # if old root-owned .venv in repo
sudo rm -rf "$DATA_ROOT/odm-results" && mkdir -p "$DATA_ROOT/odm-results"  # if needed once
```
