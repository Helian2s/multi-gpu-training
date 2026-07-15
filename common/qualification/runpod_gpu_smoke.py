#!/usr/bin/env python3
"""Container-side Runpod GPU qualification smoke check.

This intentionally reuses the provider-neutral behavior of the existing GPU
smoke implementation while giving Runpod queues a provider-appropriate entry
point and artifact name.
"""

from __future__ import annotations

from common.qualification.aws_gpu_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
