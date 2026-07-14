# NeMo/Megatron Image

`Dockerfile` builds the local candidate NeMo/Megatron runtime from the inspected
immutable NGC NeMo `linux/amd64` digest. It preserves the NVIDIA-pinned
framework stack and sets the explicit Python path required for `nemo`,
`megatron.core`, and `megatron.bridge` imports.

Build locally with:

```bash
make build-nemo-image
```

The local tag is `multi-gpu-training-nemo:local`. It is not a recorded
experiment image until provider-side GPU qualification passes and the same
content is pushed to ECR and GHCR with immutable digest records.
