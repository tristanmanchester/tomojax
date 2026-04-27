from __future__ import annotations

import tomojax.align as align_api
import tomojax.data as data_api
import tomojax.recon as recon_api


def test_alignment_facade_exports_documented_api() -> None:
    assert align_api.AlignConfig.__name__ == "AlignConfig"
    assert callable(align_api.align)
    assert callable(align_api.align_multires)


def test_reconstruction_facade_exports_documented_api() -> None:
    assert recon_api.FistaConfig.__name__ == "FistaConfig"
    assert recon_api.SPDHGConfig.__name__ == "SPDHGConfig"
    assert callable(recon_api.fbp)
    assert callable(recon_api.fista_tv)
    assert callable(recon_api.spdhg_tv)


def test_data_facade_exports_documented_api() -> None:
    assert data_api.LoadedNXTomo.__name__ == "LoadedNXTomo"
    assert data_api.NXTomoMetadata.__name__ == "NXTomoMetadata"
    assert data_api.SimConfig.__name__ == "SimConfig"
    assert callable(data_api.load_nxtomo)
    assert callable(data_api.save_nxtomo)
    assert callable(data_api.validate_nxtomo)
    assert callable(data_api.simulate)
    assert callable(data_api.sphere)
