# Third-party notices

## T3 Code

The Rust connection layer and pixel stream in `studio/crates/studio-core/src/` and worker in `studio/crates/tomojax-worker/src/` incorporate
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
| `apps/server/src/attachmentStore.ts` | `studio-core/src/transfer.rs` |
| `apps/server/src/preview/ServerBrowser.ts` | `tomojax-worker/src/wire.rs` |

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

The MCP panel labels use a 10-pixel bitmap of Sora Sagano's Aileron Regular.
The embedded font in Pillow credits the typeface as “No Rights Reserved.”
[Original typeface](https://dotcolon.net/fonts/aileron/)

## Pillow

The MCP annotation rasterizer ports the line, polygon and ellipse rules from
Pillow 12.2.0, `src/libImaging/Draw.c` at commit `3c41c095`. The thumbnail
filters port the same release’s `src/PIL/Image.py`, `src/libImaging/Resample.c`
and `src/libImaging/Reduce.c`.

The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
