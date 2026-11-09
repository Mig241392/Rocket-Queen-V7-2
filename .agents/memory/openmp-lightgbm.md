---
name: LightGBM OpenMP on Replit
description: Replit Nix package-catalog and runtime-loader behavior for LightGBM's OpenMP dependency.
---

In this environment, the system-package catalog rejected `libgomp` and `stdenv.cc.cc.lib`, while `gcc` installed successfully. GCC can report its OpenMP library with `gcc -print-file-name=libgomp.so.1`, but that library may still be absent from `LD_LIBRARY_PATH`. Preload the reported absolute path with `ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)` before importing LightGBM.

**Why:** LightGBM's Python wheel depends on the system OpenMP library, and installing the compiler alone may not expose that runtime library to Python's dynamic loader.

**How to apply:** When LightGBM import fails with `libgomp.so.1` in this Replit environment, install GCC through the system-dependency flow and preload the library before importing LightGBM. Keep the import optional so the rest of the bot can run when the backend is unavailable.
