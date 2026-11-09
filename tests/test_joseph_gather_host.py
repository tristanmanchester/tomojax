"""Execute the production gather header on a CPU against exhaustive ray sums.

This checks float32 support enumeration and detector layouts without a GPU. It
does not qualify CUDA compilation, occupancy, speed, or its device execution.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

_HOST = r"""
template<int R, typename Image>
void exhaustive(const float* c, Image image, int ix,int iy,int iz,
                int nu,int nv,float (&total)[R]) {
    int a=(int)c[0];
    for(int u=0;u<nu;++u) for(int v=0;v<nv;++v) {
        float pb=__fadd_rn(__fmul_rn(c[1],(float)u),__fmul_rn(c[2],(float)v));
        float pc=__fadd_rn(__fmul_rn(c[5],(float)u),__fmul_rn(c[6],(float)v));
        float value=joseph_pixel(image,nu,u,v);
        for(int j=0;j<R;++j) {
            float k=(float)(a==0?ix:(a==1?iy:iz+j));
            float b=(float)(a==0?iy:(a==1?iz+j:ix));
            float t=(float)(a==0?iz+j:(a==1?ix:iy));
            float qb=__fadd_rn(pb,__fadd_rn(__fmul_rn(c[3],k),c[4]));
            float qc=__fadd_rn(pc,__fadd_rn(__fmul_rn(c[7],k),c[8]));
            total[j]+=value*(plane_weight(qb-b)*plane_weight(qc-t)*c[9]);
        }
    }
}
template<int R, typename Image>
void apply(bool dense,const float* c,Image image,const int* xyz,
           int nu,int nv,float* out) {
    float total[R]={};
    bool unit=(c[2]==0.f && c[5]==0.f && fabsf(c[6])==1.f)
        ||(c[1]==0.f && c[6]==0.f && fabsf(c[2])==1.f);
    if(dense) exhaustive<R>(c,image,xyz[0],xyz[1],xyz[2],nu,nv,total);
    else if(unit) gather_unit_rows<R>(c,image,xyz[0],xyz[1],xyz[2],nu,nv,total);
    else gather_view<R>(c,image,xyz[0],xyz[1],xyz[2],nu,nv,total);
    for(int j=0;j<R;++j) out[j]=total[j];
}
template<typename Image>
void run(int r,bool dense,const float* c,Image image,const int* xyz,
         int nu,int nv,float* out) {
    if(r==1) apply<1>(dense,c,image,xyz,nu,nv,out);
    if(r==2) apply<2>(dense,c,image,xyz,nu,nv,out);
    if(r==3) apply<3>(dense,c,image,xyz,nu,nv,out);
    if(r==4) apply<4>(dense,c,image,xyz,nu,nv,out);
    if(r==16) {
        float total[16]={};
        gather_tile<4,4>(c,image,xyz[0],xyz[1],xyz[2],nu,nv,total);
        for(int j=0;j<16;++j) out[j]=total[j];
    }
}
extern "C" void evaluate(int r,int dense,int layout,int n,int nu,int nv,
        int wu,int wv,int u0,int v0,const float* coeff,const float* images,
        const int* samples,float* out) {
    for(int i=0;i<n;++i) {
        const int* sample=samples+4*i;
        const float* c=coeff+14*sample[0];
        const float* image=images+(long)sample[0]*(layout==2?wu*wv:nu*nv);
        if(layout==0) run(r,dense,c,image,sample+1,nu,nv,out+r*i);
        if(layout==1) run(r,dense,c,DetectorColumns{image,nv},sample+1,nu,nv,out+r*i);
        if(layout==2) run(r,dense,c,DetectorRegion{image,wu,wv,u0,v0},
                         sample+1,nu,nv,out+r*i);
    }
}
extern "C" void intervals(int n,const float* cases,const int* bounds,int* result) {
    for(int i=0;i<n;++i) {
        const float* c=cases+6*i;
        int first=bounds[2*i],stop=bounds[2*i+1];
        float inverse=c[1]==0.f?0.f:1.f/c[1];
        clip_coordinate(c[0],c[1],inverse,c[2],c[3],c[4],c[5],first,stop);
        result[2*i]=first;result[2*i+1]=stop;
    }
}
extern "C" void volume(int tiled,int nviews,int nx,int ny,int nz,int nu,int nv,
        const float* coeff,const float* images,const float* initial,float* out) {
    for(int i=0;i<nx*ny*nz;++i) out[i]=initial[i];
    if(!tiled) {
        for(int ix=0;ix<nx;++ix) for(int iy=0;iy<ny;++iy) for(int iz=0;iz<nz;iz+=4) {
            float total[4]={};
            for(int view=0;view<nviews;++view)
                gather_view<4>(coeff+14*view,images+(long)view*nu*nv,ix,iy,iz,nu,nv,total);
            for(int j=0;j<4 && iz+j<nz;++j) out[VOXEL(long,ix,iy,iz+j)]+=total[j];
        }
    } else for(int a=0;a<3;++a) {
        int na=a==0?nx:(a==1?ny:nz),nb=a==0?ny:(a==1?nz:nx),nc=a==0?nz:(a==1?nx:ny);
        int count=na*((nb+3)/4)*((nc+3)/4);
        blockDim.x=1;threadIdx.x=0;
        // One-thread blocks execute the actual CUDA indexing and accumulation
        // on the host; shared-memory barriers become no-ops in this setting.
        for(int tile=0;tile<count+3;++tile) {
            blockIdx.x=tile;
            const float* start=a==0?initial:out;
            if(a==0) joseph_gather_tile_0(coeff,images,out,start,nviews,nx,ny,nz,nu,nv);
            if(a==1) joseph_gather_tile_1(coeff,images,out,start,nviews,nx,ny,nz,nu,nv);
            if(a==2) joseph_gather_tile_2(coeff,images,out,start,nviews,nx,ny,nz,nu,nv);
        }
    }
}
"""


def host_source(core, z_fast):
    """Compile production headers and kernel bodies with one-thread CUDA stubs."""
    prefix = """
#include <algorithm>
#include <cmath>
using std::min; using std::max;
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__
#define Z_FAST 0
inline float __fmul_rn(float a,float b) { return a*b; }
inline float __fadd_rn(float a,float b) { return a+b; }
struct CudaIndex { long x; };
CudaIndex blockIdx,blockDim,threadIdx;
inline void __syncthreads() {}
"""
    prefix = prefix.replace("#define Z_FAST 0", f"#define Z_FAST {z_fast}")
    for kernel in ("adjoint_unit", "region"):
        code = (core / "cuda_projector" / (kernel + ".cu")).read_text()
        prefix += code.split('#include "joseph_sampling.cuh"')[0]
    kernel = (core / "cuda_projector/adjoint_tile.cu").read_text()
    kernel = "\n".join(line for line in kernel.splitlines() if not line.startswith("#include"))
    kernels = ""
    for axis in range(3):
        kernels += (
            f"\n#define AXIS {axis}\n#define joseph_gather_tile joseph_gather_tile_{axis}\n"
            + kernel
            + "\n#undef AXIS\n#undef joseph_gather_tile\n"
        )
    return (
        prefix + '\n#include "joseph_sampling.cuh"\n#include "joseph_gather.cuh"\n'
        '#include "joseph_tile.cuh"\n' + kernels + _HOST
    )


@pytest.fixture(scope="module", params=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1)])
def gather_host(request, tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host execution of CUDA gather headers requires g++")
    core = Path(__file__).resolve().parents[1] / "src/tomojax/core"
    directory = tmp_path_factory.mktemp("gather_host")
    cubic, absolute, z_fast = request.param
    source = directory / "gather.cpp"
    source.write_text(host_source(core, z_fast))
    library = directory / "gather.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            f"-DCUBIC={cubic}",
            f"-DABSOLUTE={absolute}",
            "-I",
            str(core),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compiled = ctypes.CDLL(str(library))
    function = compiled.evaluate
    floats = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int] * 10 + [floats, floats, integers, floats]
    function.restype = None
    interval = compiled.intervals
    interval.argtypes = [ctypes.c_int, floats, integers, integers]
    interval.restype = None
    volume = compiled.volume
    volume.argtypes = [ctypes.c_int] * 7 + [floats] * 4
    volume.restype = None

    def evaluate(coeff, images, samples, run, *, dense=False, layout=0, window=None):
        nv, nu = images.shape[-2:]
        wu, wv, u0, v0 = nu, nv, 0, 0
        if layout == 1:
            images = images.swapaxes(-1, -2)
        if window is not None:
            u0, v0, wu, wv = window
            images = images[:, v0 : v0 + wv, u0 : u0 + wu]
        out = np.empty((len(samples), run), np.float32)
        function(
            run,
            int(dense),
            layout,
            len(samples),
            nu,
            nv,
            wu,
            wv,
            u0,
            v0,
            np.ascontiguousarray(coeff),
            np.ascontiguousarray(images),
            np.ascontiguousarray(samples),
            out,
        )
        return out

    def clip(cases, bounds):
        out = np.empty_like(bounds)
        interval(len(cases), np.ascontiguousarray(cases), np.ascontiguousarray(bounds), out)
        return out

    def whole_volume(coeff, images, initial, *, tiled):
        nz, ny, nx = initial.shape
        nv, nu = images.shape[-2:]
        if z_fast:
            initial = np.ascontiguousarray(initial.transpose(2, 1, 0))
        out = np.empty_like(initial)
        if tiled:
            images = images.swapaxes(-1, -2)
        volume(
            int(tiled),
            len(coeff),
            nx,
            ny,
            nz,
            nu,
            nv,
            np.ascontiguousarray(coeff),
            np.ascontiguousarray(images),
            initial,
            out,
        )
        return out.transpose(2, 1, 0) if z_fast else out

    return SimpleNamespace(evaluate=evaluate, clip=clip, volume=whole_volume)


def coefficient_cases():
    """Invertible sheared ray lattices, all major axes and rounded cell boundaries."""
    rng = np.random.default_rng(381)
    count = 108
    coeff = np.zeros((count, 14), np.float32)
    coeff[:, 0] = np.arange(count) % 3
    for i in range(count):
        angle = rng.uniform(-np.pi, np.pi)
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        matrix = np.diag(rng.uniform(0.1, 2.7, 2)) @ matrix
        if i < 18:
            tiny = [0.0, 1e-8, -1e-8, 1e-4, -1e-4, 0.3][i // 3]
            matrix = np.array([[1.0, tiny], [tiny, (-1.0) ** i]])
        coeff[i, [1, 2, 5, 6]] = matrix.ravel()
        inverse = np.linalg.inv(coeff[i, [1, 2, 5, 6]].reshape(2, 2))
        coeff[i, 10:] = inverse.ravel()
        coeff[i, [3, 7]] = rng.uniform(-0.7, 0.7, 2)
        coeff[i, [4, 8]] = 8 - matrix @ [11, 9] - coeff[i, [3, 7]] * 8
        coeff[i, 9] = rng.uniform(0.3, 2.1)
    # Include coordinates immediately on either side of exact support edges.
    for i in range(18):
        shift = np.float32([-2, -1, 0, 1, 2, 7][i // 3])
        toward = [np.float32(-np.inf), shift, np.float32(np.inf)][i % 3]
        coeff[i, [4, 8]] = np.nextafter(shift, toward)
    samples = np.column_stack((np.arange(648) % count, rng.integers(0, 17, (648, 3))))
    images = rng.normal(size=(count, 19, 23)).astype(np.float32)
    return coeff, images, samples.astype(np.int32)


@pytest.mark.parametrize("run", [1, 2, 3, 4])
def test_gather_support_and_layout_equal_exhaustive_float32_ray_sum(gather_host, run):
    coeff, images, samples = coefficient_cases()
    expected = gather_host.evaluate(coeff, images, samples, run, dense=True)
    for layout in (0, 1):
        actual = gather_host.evaluate(coeff, images, samples, run, layout=layout)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_gather_window_uses_global_support_coordinates(gather_host):
    coeff, images, samples = coefficient_cases()
    window = (3, 2, 13, 11)
    masked = np.zeros_like(images)
    masked[:, 2:13, 3:16] = images[:, 2:13, 3:16]
    expected = gather_host.evaluate(coeff, masked, samples, 4, dense=True)
    actual = gather_host.evaluate(coeff, images, samples, 4, layout=2, window=window)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_interval_clipping_keeps_exact_rounded_coordinate_support(gather_host):
    rng = np.random.default_rng(37)
    cases = rng.uniform(-30, 30, (2048, 6)).astype(np.float32)
    cases[:, 1] *= 0.1
    cases[:, 2:4].sort(axis=1)
    cases[:, 4:6].sort(axis=1)
    # Rounded coordinates can be constant over an interval, even with a
    # nonzero slope. Tiny reciprocal overflow must not remove valid samples.
    cases[:64, 0] = np.float32(2**23)
    cases[:64, 2:4] = -np.float32(2**23)
    cases[:64, 4:6] = [-1, 1]
    cases[64:128, 1] = np.resize([0, 1e-8, -1e-8, 1e-40, -1e-40], 64)
    bounds = np.tile(np.array([0, 64], np.int32), (len(cases), 1))
    ranges = gather_host.clip(cases, bounds)
    v = np.arange(64, dtype=np.float32)[None]
    pixel = cases[:, :1] + cases[:, 1:2] * v
    low, high = pixel + cases[:, 2:3], pixel + cases[:, 3:4]
    expected = (high > cases[:, 4:5]) & (low < cases[:, 5:6])
    actual = (v >= ranges[:, :1]) & (v < ranges[:, 1:2])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("window", [None, (3, 2, 13, 11)])
def test_transverse_tile_keeps_every_independent_ray_weight(gather_host, window):
    coeff, images, samples = coefficient_cases()
    repeated = np.repeat(samples, 16, axis=0)
    b = samples[:, 2:3] + np.repeat(np.arange(4), 4)
    c = samples[:, 3:4] + np.tile(np.arange(4), 4)
    k = samples[:, 1:2] + np.zeros(16, dtype=int)
    a = coeff[samples[:, 0], 0:1]
    repeated[:, 1] = np.where(a == 0, k, np.where(a == 1, c, b)).ravel()
    repeated[:, 2] = np.where(a == 0, b, np.where(a == 1, k, c)).ravel()
    repeated[:, 3] = np.where(a == 0, c, np.where(a == 1, b, k)).ravel()
    reference_images = images.copy()
    if window is not None:
        reference_images[:] = 0
        reference_images[:, 2:13, 3:16] = images[:, 2:13, 3:16]
    expected = gather_host.evaluate(coeff, reference_images, repeated, 1, dense=True)
    for layout in (0, 1) if window is None else (2,):
        actual = gather_host.evaluate(coeff, images, samples, 16, layout=layout, window=window)
        np.testing.assert_array_equal(
            actual.view(np.uint32).ravel(), expected.view(np.uint32).ravel()
        )


@pytest.mark.parametrize("only_axis", [None, 1, 2])
def test_axis_grouped_tiles_only_change_roundoff_of_view_sums(gather_host, only_axis):
    coeff, images, _ = coefficient_cases()
    if only_axis is not None:
        selected = coeff[:, 0] == only_axis
        coeff, images = coeff[selected], images[selected]
    coeff, images = np.tile(coeff, (7, 1)), np.tile(images, (7, 1, 1))
    rng = np.random.default_rng(18)
    images *= rng.uniform(0.3, 2.1, (len(images), 1, 1)).astype(np.float32)
    initial = rng.normal(size=(5, 7, 9)).astype(np.float32)
    original = gather_host.volume(coeff, images, initial, tiled=False)
    tiled = gather_host.volume(coeff, images, initial, tiled=True)
    if only_axis is not None:
        np.testing.assert_array_equal(original.view(np.uint32), tiled.view(np.uint32))
    error = np.linalg.norm((original - tiled).astype(np.float64))
    assert error < 2e-6 * np.linalg.norm(original.astype(np.float64))
