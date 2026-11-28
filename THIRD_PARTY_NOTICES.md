# Third-party notices

## T3 Code

The Rust connection layer and pixel stream in `studio/crates/studio-core/src/` incorporate
code from T3 Code commit `d2c440312e5f39d59827ed466838a623516454d6`.

| T3 Code source | TomoJAX source |
| --- | --- |
| `packages/ssh/src/command.ts` | `connection/command.rs` |
| `packages/ssh/src/auth.ts` | `connection/auth.rs` |
| `packages/ssh/src/errors.ts` | `connection/mod.rs` |
| `packages/ssh/src/tunnel.ts` | `connection/command.rs`, `connection/process.rs` |
| `packages/client-runtime/src/connection/supervisor.ts`, `packages/client-runtime/src/connection/presentation.ts` | `connection/supervisor.rs` |
| `packages/client-runtime/src/connection/registry.ts` | `connection/registry.rs` |
| `packages/client-runtime/src/preview/serverBrowserStream.ts`, `apps/server/src/preview/ServerBrowser.ts` | `pixels/stream.rs` |

MIT License

Copyright (c) 2026 T3 Tools Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
