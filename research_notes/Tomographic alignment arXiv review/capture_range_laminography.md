# Robust, large-capture-range tomographic alignment and laminography/tilted-axis alignment: literature notes for TomoJAX

Scope note: TomoJAX estimates 5 DOF per view (3 rotations, detector x/z shifts). Current pilot uses +/-0.25 deg, +/-0.5 px motion. Laminography: 0.22 deg rotation RMSE after 64 alternating outer iterations, image rel. L2 about 0.12 from the missing cone. Target: >=99% recovery on noisy cases with up to +/-3 deg and +/-10 px initial error.

Verification status of IDs. I confirmed these arXiv IDs by downloading the PDF and reading the title/author header: 1705.08678, 2008.03375, 2310.09567, 2401.09283, 2405.19079, 2304.04597, 1803.04495, 2006.10390. I confirmed these from arXiv listing/abstract pages: 2209.08450, 2401.11101, 2511.01893, 2009.09498, 2509.13863, 2601.07254. For the remaining papers my checks (Semantic Scholar externalIds, Europe PMC, web search) found no arXiv version, so they are listed by DOI. The arXiv API rate-limited me (HTTP 429) part-way through, so I could not search arXiv exhaustively by keyword. Some recent papers may have arXiv versions I missed.

## Q1. Projection matching / reprojection alignment for X-ray nano-tomography and ptycho-tomography (PSI, APS/Gürsoy, Nikitin): capture range, accuracy, what makes them robust

### Takeaway
The production pipelines at PSI and APS do not use one wide-basin optimiser. They chain a cheap coarse stage (cross-correlation, plus a common-line or vertical-mass-fluctuation step for the axial shift) with a multiresolution projection-matching refinement. Published numbers on translation-only problems are deep sub-pixel accuracy, about 0.2 px RMS or better, starting from about +/-10 px. Almost none of these papers report a success rate over many random trials. Papers that recover all three rotations are rarer. The best available evidence for large rotation errors (van Leeuwen et al.) shows that the rotation about the tomographic axis is the parameter that fails once its error exceeds the angular sampling.

### Cited Findings

**Odstrčil, Holler, Raabe, Guizar-Sicairos (2019), "Alignment methods for nanotomography with deep subpixel accuracy", Optics Express 27(25):36637. No arXiv version found. DOI 10.1364/OE.27.036637 (older but foundational, PSI cSAXS).**
- The toolkit combines deep-subpixel methods with a multiresolution scheme. The authors say this makes alignment "robust and accurate" at much lower compute and memory cost than common iterative alignment. It was demonstrated on simulated and measured data for both tomography and laminography geometries, and the GPU implementation is public. — [Europe PMC abstract / Optics Express](https://doi.org/10.1364/OE.27.036637)
- Pipeline: (1) cross-correlation alignment (XCA) for the initial guess, (2) the vertical-mass-fluctuation (VMF) common-line method for vertical alignment, independent of horizontal misalignment, (3) multi-resolution projection-matching alignment (MR-PMA) for the final result. The methods also apply to laminography and interior tomography. — [web search summary of the Optica paper/ResearchGate](https://www.researchgate.net/publication/337704331_Alignment_methods_for_nanotomography_with_deep_subpixel_accuracy)
- "MR-PMA reached sub-0.2 pixel RMS accuracy for all tested configurations in the final resolution level." This comes from a search-engine summary of the Optica full text; I could not load the full text (it is JS-gated), so treat the number as secondary. — [Optica page](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-25-36637&id=423764)
- The code and a simulated example dataset ("tomography alignment toolkit ... for reconstruction and alignment of tomography and laminography datasets", Matlab + CUDA) are on Zenodo. — [Zenodo 3360819](https://zenodo.org/records/3360819)
- Evidence: simulated and real (ptycho-tomography and laminography at PSI).

**Gürsoy et al. (2017), "Rapid alignment of nanotomography data using joint iterative reconstruction and reprojection", Sci. Rep. 7. No arXiv version found. DOI 10.1038/s41598-017-12141-9 (older, APS; the basis of TomoPy's `align_joint`).**
- Algorithm 2 interleaves a single reconstruction iteration with re-alignment by phase correlation ("avoids iterations in iterations"). The older Algorithm 1 runs a full reconstruction to convergence and then aligns. Algorithm 2 converges faster, is more robust, and is more accurate for limited-angle data. — [PMC5603591](https://pmc.ncbi.nlm.nih.gov/articles/PMC5603591/)
- Capture-range test: 100 projections over 0-180 deg on a 100x100 px detector. Transverse and axial shifts were drawn independently from U(-10, 10) px, and Gaussian noise was added. "The joint algorithm could satisfactorily estimate all axial shifts for up to 10% noise (and most of them for 20% noise) at sub-pixel accuracy." Residual alignment error was "a factor of a thousand smaller" than the sequential approach, which needed 40 SIRT iterations x 10 outer loops = 400 iterations. — [PMC5603591](https://pmc.ncbi.nlm.nih.gov/articles/PMC5603591/)
- Only translations were estimated. Real data: X-ray and electron nanotomography. The paper also notes that cross-correlation of adjacent views can remove frame-to-frame jitter but "cannot be relied upon to find a common rotation axis". — [PMC5603591](https://pmc.ncbi.nlm.nih.gov/articles/PMC5603591/)

**Pande, Donatelli, Parkinson, Yan, Sethian (2022), "Joint iterative reconstruction and 3D rigid alignment for X-ray tomography", Optics Express 30(6). No arXiv version found. DOI 10.1364/OE.443248 (LBNL/CAMERA).**
- Joint reconstruction plus alignment under a rigid-body deformation model. The authors claim it is "highly efficient in recovering relatively large alignment errors without prior knowledge of a low resolution approximation of the 3D structure or a reasonable estimate of alignment parameters". Tested on synthetic phantom and experimental data. — [OSTI 1847186](https://www.osti.gov/pages/biblio/1847186); [Europe PMC abstract](https://doi.org/10.1364/OE.443248)
- This is the paper closest to TomoJAX's problem (full 3D rigid motion per view), but I could not get its numeric capture range (see Gaps).

**Wang, C.-C. (2020), "Joint Iterative Fast Projection Matching for Fully Automatic Marker-free Alignment of Nano-tomography Reconstructions" (JI-Faproma), Sci. Rep. 10:7330. No arXiv version found. DOI 10.1038/s41598-020-62949-1 (NSRRC Taiwan).**
- Decomposes alignment by DOF: in-plane rotation from frequency-domain common lines, vertical shift from real-space common lines, horizontal shift from single-layer joint iterative reprojection. — [PMC7192921](https://pmc.ncbi.nlm.nih.gov/articles/PMC7192921/)
- Simulated results: noise-free data align perfectly. Up to 20% noise, the errors are "within 0.12°, 0 pixel, and 1 pixel" (in-plane rotation, vertical, horizontal). A 3x3 smoothing kernel cut horizontal RMSE from 1 to 0.6 px at 20% noise. Converges in fewer than 15 JIRRM iterations. — [PMC7192921](https://pmc.ncbi.nlm.nih.gov/articles/PMC7192921/)

**Sanders (2018), "Phase Based Alignment and Improved Projection Matching of Parallel Beam Tomography Data", arXiv:1803.04495 (older).**
- Recasts translation misregistration as multiplicative Fourier phase errors and estimates them inside the iterative reconstruction from discrepancies between data and reprojections. Regularisation (e.g. TV) helps recover the correct phases. The analysis also shows how to "notably improve the basic projection matching" to similar accuracy. — [arXiv:1803.04495](https://arxiv.org/abs/1803.04495)

**Aslan, Nikitin, Ching, Bicer, Leyffer, Gürsoy (2019), "Joint ptycho-tomography reconstruction through alternating direction method of multipliers", Optics Express 27:9128. No arXiv version found. DOI 10.1364/OE.27.009128. Follow-up with learned priors: Aslan et al., "Joint ptycho-tomography with deep generative priors", arXiv:2009.09498 (MLST 2021).**
- ADMM splits the joint ptycho-tomography problem into a ptychography sub-problem and a tomography sub-problem. Alignment is not the main focus. — [Semantic Scholar record](https://doi.org/10.1364/OE.27.009128); [arXiv:2009.09498](https://arxiv.org/abs/2009.09498)

**Nikitin, De Andrade, Slyamov, Gould, Zhang, Sampathkumar, Kasthuri, Gürsoy, De Carlo (2021), "Distributed optimization for nonrigid nano-tomography", arXiv:2008.03375 (IEEE TCI 7:272-287).**
- ADMM with three sub-problems: tomographic reconstruction, projected deformation estimation (dense 2D Farnebäck optical flow per projection), and regularisation. Deformation is estimated coarse-to-fine, "first for the coarsest scale, then ... propagated step-by-step to the finest levels". The authors say it is robust to Poisson and low-frequency background noise in synthetic tests. — [arXiv:2008.03375](https://arxiv.org/abs/2008.03375)
- Real TXM data (APS 32-ID, two independent sets of 1440 projections): FSC (1/2-bit) resolution was 195 nm with CG and 192 nm with pCG without deformation correction. With optical-flow methods it was 151 nm (3D OF), 152 nm (non-dense OF) and 127 nm (dense OF). — [arXiv:2008.03375 PDF](https://arxiv.org/pdf/2008.03375)
- The authors note that "most algorithms are given without formal study of convergence analysis", and that binning or filtering is sometimes needed to make the method converge. — [arXiv:2008.03375](https://arxiv.org/abs/2008.03375)

**Odstrčil et al. (2019), "Ab initio nonrigid X-ray nanotomography", Nat. Commun. 10. DOI 10.1038/s41467-019-10670-7 (older, PSI).**
- Real ptycho-tomography data: after the static model was used for the initial guess, an extra projection shift correction "with amplitude up to four pixels" was needed. Simulated DVF with maximum displacement 10 px was recovered with 0.8 px RMS error. 50 joint iterations were run, with Gaussian (sigma = 30 px) DVF regularisation. — [PMC6565693](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565693/)

**TomocuPy: Nikitin (2023), "TomocuPy: efficient GPU-based tomographic reconstruction with asynchronous data processing", arXiv:2209.08450 (JSR 2023).**
- GPU tomography and laminography reconstruction (Fourier-based laminography), with a full 2048^3 reconstruction in under 7 s on one A100. It does reconstruction, not alignment. Laminography geometry is calibrated by manual rotation-axis and pitch search (see Q3). — [arXiv:2209.08450](https://arxiv.org/abs/2209.08450)

### Inferences
- The +/-10 px translation part of the TomoJAX target is in line with the literature: Gürsoy reports sub-pixel recovery from U(-10, 10) px at <=10% noise. Rotations are different. None of the X-ray nano-tomography projection-matching papers I could read reports per-view 3-rotation recovery from +/-3 deg with a success rate. A >=99% success claim would therefore need TomoJAX's own Monte-Carlo protocol (see Q6).
- The robust pipelines all decouple identifiable sub-problems before the joint refinement: vertical shift from VMF or common lines, in-plane rotation from frequency-domain common lines, horizontal shift from reprojection. TomoJAX's phase-correlation translation seed matches the first step. Adding a common-line/VMF axial-shift estimate and a common-line in-plane-rotation estimate (JI-Faproma reports 0.12 deg at 20% noise) would give a closed-form, data-only seed for 2 of the 5 DOF before any reconstruction-based step.
- Gürsoy's "one reconstruction iteration per alignment step" result means that inner reconstruction accuracy is not needed early. This supports cheap, partially converged volume updates in the early outer iterations, as long as late iterations are accurate (van Leeuwen gives the convergence argument; see Q2).

### Gaps
- Pande et al. 2022 numeric capture range (initial rotation and shift magnitudes, final RMSE): the full text is not in PMC, and the Optica and OSTI pages only showed the abstract.
- Odstrčil 2019 tested misalignment magnitudes and laminography-specific accuracy: the full text was not accessible, so only the secondary "sub-0.2 px RMS" summary is available.
- No published success-rate statistics (fraction of random trials converging) were found for any of the X-ray projection-matching tools.
- Gürsoy's "tomoalign" and Nikitin's newer APS alignment work after 2023: I found no additional arXiv paper reporting rigid 3-rotation capture range.

## Q2. Multiresolution, coarse-to-fine, continuation, smoothing/low-pass objectives and robust losses; and variable projection (VarPro) / Schur-complement joint Gauss–Newton vs alternation

### Takeaway
The strongest theoretical and empirical support for a VarPro-style reduced problem comes from van Leeuwen, Maretzke and Batenburg (Inverse Problems 2018). They eliminate the volume, recover 5 DOF in parallel beam, use regularisation-strength continuation to widen the basin, and degrade gracefully up to 16x larger initial misalignment, except for the rotation about the tomographic axis. Guerrero et al. (IEEE TCI 2024) also use VarPro for geometry calibration. The usual basin-widening tools are multiresolution (Odstrčil, Nikitin), smoothing priors (van Leeuwen) and low-dimensional motion models (Thies). Thies et al. also show that over-smooth motion models cap the motion frequencies that can be recovered.

### Cited Findings

**van Leeuwen, Maretzke, Batenburg (2018), "Automatic alignment for three-dimensional tomographic reconstruction", arXiv:1705.08678 (Inverse Problems 34(2):024004) (older but essential).**
- Uses variable projection: "we eliminate the reconstructed object by setting u(a) := argmin_u f(a,u)" and then minimise the reduced function over alignment parameters a. The reconstruction is computed inexactly with tolerance epsilon_k. There are convergence guarantees to a local minimum when epsilon decreases to 0, or to within O(epsilon) of one for fixed epsilon. Three algorithms are compared: gradient on the reduced function, proximal-gradient with bound constraints, and alternating reconstruct/align. — [arXiv:1705.08678](https://arxiv.org/abs/1705.08678)
- The parallel-beam parameterisation is exactly TomoJAX's 5 DOF: in-plane (about z), nod/pitch (about x) and tomographic (about y) rotations, plus lateral and axial shifts. Cone-beam and dual-tilt setups are also covered. — [arXiv:1705.08678 PDF](https://arxiv.org/pdf/1705.08678)
- Robustness: initial misalignment was scaled by 2, 4, 8 and 16. "Even for the most severe initial misalignment ... all alignment parameters are accurately recovered, except for the tomographic angle. If the misalignment exceeds the angular sampling, it becomes exceedingly hard to find the correct projection angles as nearby projections are similar." The procedure "deteriorates gracefully as the initial misalignment increases". — [arXiv:1705.08678 PDF](https://arxiv.org/pdf/1705.08678)
- Continuation: the reconstruction uses a gradient (Tikhonov-type) penalty alpha*||grad u||^2. A heuristic assigns alpha of roughly 10^1 to 10^4, larger for larger misalignment. "A smoother reconstruction, as induced by a larger alpha, improves the initial convergence of the alignment", and "one could ... use a continuation, starting from large alpha and reducing it as the alignment improves". — [arXiv:1705.08678 PDF](https://arxiv.org/pdf/1705.08678)
- Inner accuracy: too loose a reconstruction tolerance (epsilon = 1e-1) "may lead to a premature stalling of the convergence, or even to divergence". More alignment sub-steps per outer iteration gave "slightly more optimal results" but no better reconstruction accuracy and a slightly larger alignment error. Truncated (ROI) data are also handled. — [arXiv:1705.08678 PDF](https://arxiv.org/pdf/1705.08678)
- Evidence: simulated data plus one real electron tomography dataset. — [arXiv:1705.08678](https://arxiv.org/abs/1705.08678)

**Guerrero, Bellens, Santander, Dewulf (2024), "Automatic and Computationally Efficient Alignment in Fan- and Cone-beam Tomography", arXiv:2310.09567 (IEEE TCI, DOI 10.1109/TCI.2024.3396385).**
- Uses VarPro ("removing a subset of variables from the loss function which are relatively easy to solve and then ... a reduced problem over the remaining variables") to estimate the detector u-shift and in-plane detector rotation in cone beam. Fan beam uses cheaper consistency-condition strategies (Helgason–Ludwig and epipolar conditions are discussed). Validated on simulated and experimental industrial CT, with code. — [arXiv:2310.09567](https://arxiv.org/abs/2310.09567)
- Global geometry only (a few parameters per scan), not per-view poses.

**Thies, Wagner, Maul, et al. (2024/2025), "A gradient-based approach to fast and accurate head motion compensation in cone-beam CT", arXiv:2401.09283 (IEEE TMI 2025).**
- Analytic derivatives of backprojection with respect to cone-beam geometry. Rigid 6-DOF per-view motion is modelled through spline nodes, cutting the DOF from 6*Np to 6*Nn and enforcing smooth motion. Gradient descent is run on a differentiable image-quality target in reconstruction space (a learned voxel-wise quality network; autoencoder-like architectures gave better gradient flow). — [arXiv:2401.09283](https://arxiv.org/abs/2401.09283)
- Results: reprojection error fell "from an initial average of 3 mm to 0.61 mm", with a 19x speed-up over existing methods. Evidence is realistic head-anatomy experiments (simulated motion on real head CT). — [arXiv:2401.09283](https://arxiv.org/abs/2401.09283)

**Thies, Wagner, Maul, Mei, Gu, Pfaff, Vysotskaya, Yu, Maier (2024), "On the Influence of Smoothness Constraints in Computed Tomography Motion Compensation", arXiv:2405.19079.**
- The choice of spline motion model "crucially influences recoverable frequencies". The optimiser fits spline nodes accurately, but motion above the model's bandwidth cannot be recovered. — [arXiv:2405.19079](https://arxiv.org/abs/2405.19079)

**Jiang, Wang, Uneri, Zbijewski, Stayman (2026), "Differentiable Forward and Back-Projector for Rigid Motion Estimation in X-Ray Imaging", IEEE TBME. No arXiv version found. DOI 10.1109/TBME.2025.3643742.**
- Analytic continuous-domain gradients of forward and back projection with respect to rigid pose, expressed through the projection operators themselves. It is about 8x faster than an existing differentiable forward projector at similar accuracy (2D/3D registration). Also applied to motion-compensated reconstruction and CBCT geometry calibration on real phantoms. — [Europe PMC abstract](https://doi.org/10.1109/tbme.2025.3643742)

**Riis, Dong, Hansen (2021), "Computed tomography with view angle estimation using uncertainty quantification", Inverse Problems. No arXiv version found. DOI 10.1088/1361-6420/abf5ba.**
- Joint image and view-angle estimation that quantifies view-angle uncertainty through a model-discrepancy term. The authors say it "generalizes in a straightforward way to other cases of uncertain geometry". — [Semantic Scholar/IOP abstract](https://doi.org/10.1088/1361-6420/abf5ba)

**Coarse-to-fine elsewhere**
- Odstrčil 2019 (MR-PMA) and Nikitin 2021 (coarsest-to-finest optical flow): see Q1. Gürsoy 2017 notes that early electron-tomography projection matching (Dengler 1989) already used multiscale downsampled reconstructions for a first alignment pass. — [PMC5603591](https://pmc.ncbi.nlm.nih.gov/articles/PMC5603591/)

### Inferences
- Support for a VarPro / Schur-complement joint Gauss–Newton step:
  - van Leeuwen et al. show that the reduced (volume-eliminated) problem is the right object to optimise. Its gradient equals the partial gradient at the inner optimum u(a). Inexact inner solves keep convergence as long as tolerances tighten.
  - Plain alternation is their Algorithm 3. They found that over-solving the alignment sub-problem at a fixed volume does not help and can slightly hurt.
  - TomoJAX's 64 outer alternating iterations stalling at 0.22 deg in laminography fit the known weakness of alternation on strongly coupled separable problems. When pose and volume are correlated, block-coordinate steps zig-zag. VarPro and Schur-complement GN account for that coupling through the reduced Jacobian.
  - This is my inference: none of these papers ran a direct alternation-vs-VarPro-GN comparison on laminography.
- Challenge to VarPro GN: van Leeuwen's basin argument depends on heavy smoothing of the inner reconstruction (alpha continuation). At +/-3 deg, the rotation about the tomographic axis is not identifiable by local methods once it exceeds the angular step. With N views over 180 deg the step is 180/N deg, so for N >= 60 a +/-3 deg tomographic-angle error exceeds it. A Gauss–Newton step alone will not fix this. It needs a global or ordering prior on the nominal angles: a monotonic/smooth prior on the tomographic-angle error, or a bounded proximal step as in van Leeuwen's Algorithm 2.
- Concrete continuation schedule for TomoJAX:
  - Pyramid (binned detector + binned volume) combined with a strong gradient or Tikhonov penalty on the volume at the coarse levels, relaxed as levels refine.
  - Optionally a low-pass filter on the residual.
  - Optionally a low-dimensional (spline/Fourier) pose model at coarse levels, released to per-view DOF only at the finest level. Thies 2405.19079 warns that keeping it too long caps recoverable high-frequency jitter.
- A robust loss (Huber/Cauchy on the residual) is not reported in the X-ray alignment papers I read. Its use rests on general M-estimation reasoning, not on cited evidence here.

### Gaps
- I found no paper that directly benchmarks VarPro/Schur GN against alternation for per-view rigid poses in laminography.
- I found no published quantitative basin-width study (success fraction vs initial error) for multiresolution vs single-resolution projection matching in X-ray tomography.

## Q3. Laminography- and tilted-axis-specific alignment: axis tilt, missing cone and pose identifiability, published pipelines

### Takeaway
Published laminography alignment is mostly instrument-side: interferometry at LamNI, mechanical two-step alignment at the APS TXM, and manual pitch and centre-of-rotation search in TomocuPy. Software-side, PSI uses projection-matching alignment (Odstrčil 2019 MR-PMA, which supports laminography) as a preprocessing step. I found no arXiv paper that reports per-view 3-rotation pose recovery accuracy for laminography or analyses how the missing cone limits identifiability of each DOF. That gap is real, and TomoJAX would need to characterise it itself.

### Cited Findings

**Holler, Odstrčil, Guizar-Sicairos, Lebugle, Frommherz, Lachat, Bunk, Raabe, Aeppli (2020), "LamNI – an instrument for X-ray scanning microscopy in laminography geometry", J. Synchrotron Rad. 27. No arXiv version found. DOI 10.1107/S1600577520003586.**
- The laminography angle between beam and rotation axis is 61 deg, reached by tilting the stage 15 deg about x and then 60 deg about y. Laminography reduces the missing wedge to a missing cone. Position metrology uses laser interferometry with 2 nm position stability, positioning errors below 5 nm and a 12 x 12 mm scan range. — [PMC7206541](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206541/)

**Holler et al. (2019), "Three-dimensional imaging of integrated circuits with macro- to nanoscale zoom", Nature Electronics 2. No arXiv version found. DOI 10.1038/s41928-019-0309-z.**
- The flagship ptycho-laminography demonstration with LamNI. I did not retrieve its alignment details. — [Semantic Scholar record](https://doi.org/10.1038/s41928-019-0309-z)

**Kang, Jiang, Holler, Guizar-Sicairos, Levi, Klug, Vogt, Barbastathis (2023), "Accelerated deep self-supervised ptycho-laminography for three-dimensional nanoscale imaging of integrated circuits", arXiv:2304.04597 (Optica, DOI 10.1364/OPTICA.492666).**
- Real LamNI IC data: 61 deg laminographic angle, 0.18 deg angular step, 2000 scans. The preprocessing is 100 iterations of ML ptychography, then "projection matching alignment (PMA)" (reference 23 = Odstrčil 2019), then a 256x256 crop, FBP, and recovery of missing-cone information with a self-supervised network. Alignment is treated as a solved preprocessing step. — [arXiv:2304.04597](https://arxiv.org/abs/2304.04597)

**Witte, Späth, Finizio, Donnelly, Watts, Sarafimov, Odstrčil, Guizar-Sicairos, Holler, Fink, Raabe (2020), "From 2D STXM to 3D Imaging: Soft X-ray Laminography of Thin Specimens", Nano Lett. 20. No arXiv version found. DOI 10.1021/acs.nanolett.9b04782.**
- Soft-X-ray laminography (270–1500 eV) at PolLux/SLS on extended thin samples. The abstract does not report quantitative alignment detail. — [Europe PMC abstract](https://doi.org/10.1021/acs.nanolett.9b04782)

**Nikitin, Mittone, Clark, Fezzaa, Wojcik, Deriy, Bean, De Carlo (2025), "Nano-laminography with a transmission X-ray microscope", J. Synchrotron Rad. No arXiv version found. DOI 10.1107/S1600577525007234.**
- The rotation axis is inclined 20 deg to the beam. Larger angles "typically introduce significant laminography artifacts due to the missing cone". 50 nm resolution. — [PMC12591070](https://pmc.ncbi.nlm.nih.gov/articles/PMC12591070/)
- Alignment is two-stage and mechanical: coarse alignment at micro-resolution with optics out of the beam, then at nano-resolution. Misalignment shows up as an elliptical feature trajectory. Simultaneously evaluating "artifacts caused by incorrect pitch, roll, and rotation axis alignment" was "impractical". — [PMC12591070](https://pmc.ncbi.nlm.nih.gov/articles/PMC12591070/)
- TomocuPy laminography reconstruction is "(1) manual rotation axis search ... (2) manual pitch angle search ... (3) full reconstruction". The authors list "reliability and automation of sample alignment" as needing improvement, since acquisition is faster than manual alignment. — [PMC12591070](https://pmc.ncbi.nlm.nih.gov/articles/PMC12591070/)

**Nikitin, Wildenberg, Mittone, Shevchenko, Deriy, De Carlo (2024), "Laminography as a tool for imaging large-size samples with high resolution", arXiv:2401.11101 (JSR 31, DOI 10.1107/S1600577524002923).**
- Laminography pipeline at APS 2-BM for samples larger than 1 cm at micrometre resolution: four sequential slabs of a whole osmium-stained mouse brain, about 12 TB raw data, with fast multi-GPU reconstruction. The abstract gives no automated pose-alignment results. — [arXiv:2401.11101](https://arxiv.org/abs/2401.11101)

**Ma, Nikitin, Wang, Bicer, Li (2025), "mLR: Scalable Laminography Reconstruction based on Memoization", arXiv:2511.01893 (SC'25).**
- Speeds up ADMM-FFT laminography reconstruction ("high reconstruction accuracy ... but ... excessive computation time and large memory") by memoising repeated FFTs. It is about reconstruction, not alignment. — [arXiv:2511.01893](https://arxiv.org/abs/2511.01893)

**Industrial cone-beam laminography geometry: Sun, Han, Tan, Xi, Li, Yan, Zhang (2023), "Geometric parameters sensitivity evaluation based on projection trajectories for X-ray cone-beam computed laminography", J. X-ray Sci. Technol. DOI 10.3233/XST-221338. No arXiv version found.**
- Defines a "Minimum Deviation Unit" from projection trajectories to rank geometric parameter sensitivity. At low magnification, three parameters (eta, u0, v0) are most sensitive. Analyses how axis-tilt angle and magnification change sensitivity. — [Europe PMC abstract](https://doi.org/10.3233/xst-221338)

**Other recent laminography arXiv papers (reconstruction only, no alignment): LamiGauss (Chen, Biguri, Morel, Chan, Schönlieb, Li, 2025), arXiv:2509.13863; LaminoDiff (Liu et al., 2026), arXiv:2601.07254.**
- Both address sparse-view or missing-cone artefacts with generative or Gaussian-splatting priors and assume known geometry. — [arXiv:2509.13863](https://arxiv.org/abs/2509.13863); [arXiv:2601.07254](https://arxiv.org/abs/2601.07254)

### Inferences
- Identifiability in laminography, from geometry rather than any cited paper. With a 30 deg tilt (TomoJAX) the projection directions sweep a cone. Shifts along the detector axis that maps to the rotation axis, and rotations whose effect lies mostly in the missing cone, are only weakly constrained. The volume can absorb part of the pose error, especially when the reconstruction is regularised by the missing-cone null space.
  - This explains a residual 0.22 deg RMSE that alternation cannot remove. The reduced-problem curvature (Schur complement) along those directions is small.
  - Recommendations for TomoJAX:
    - (a) Compute the Schur-complement / reduced Gauss–Newton Hessian at the solution and report its smallest eigenvalues and eigenvectors per DOF. This is an identifiability diagnostic.
    - (b) Fix global gauge modes explicitly: the global rotation and translation of the volume relative to all poses. TomoJAX already has `gauge.py`.
    - (c) Consider priors on the poses (smoothness over view index, known stage geometry) instead of fully free 5 DOF per view in laminography.
- PSI's practice (laminography IC imaging at 61 deg) is to align with 2D projection matching and fixed nominal stage geometry, plus nm-scale interferometry. This shows that the hard instrumental work makes rotations nearly known and only shifts are refined. A TomoJAX gate of 0.01 deg rotation RMSE in laminography therefore goes beyond what published laminography pipelines demonstrate.

### Gaps
- I found no published missing-cone identifiability analysis for per-view rigid poses.
- I found no arXiv paper by a "Cheng" group on laminography alignment. The requested "Witte/Cheng-type methods" could not be matched to a specific alignment paper. Witte 2020 is instrumentation plus reconstruction.
- No quantitative per-view rotation accuracy was found for any laminography alignment pipeline.

## Q4. Cryo-ET / electron-tomography alignment advances that transfer to X-ray rigid per-view alignment

### Takeaway
Cryo-ET alignment (AreTomo/AreTomo3/AreTomoLive, Markerfree, and the 2026 gradient-based refiners) uses the same coarse-to-fine recipe: cross-correlation and common-line coarse alignment, then projection-matching refinement, then local or patch motion. The 2026 work moves towards gradient-based refinement of per-tilt parameters and learned alignment scores. The transferable pieces are the staged pipeline, the restriction of which images enter the reference reconstruction, and the per-tilt gradient refinement.

### Cited Findings

**Zheng, Wolff, Greenan, Chen, Faas, Bárcena, Koster, Cheng, Agard (2022), "AreTomo: An integrated software package for automated marker-free, motion-corrected cryo-electron tomographic alignment and reconstruction", J. Struct. Biol. X 6:100068. bioRxiv only, no arXiv version found. DOI 10.1016/j.yjsbx.2022.100068.**
- GPU, fully automatic, marker-free. Corrects in-plane rotation, translation and local beam-induced motion between tilts. "The residual local motion after correction for global motion was found in the range of ± 80 Å." — [Europe PMC abstract](https://doi.org/10.1016/j.yjsbx.2022.100068)

**AreTomo3 / AreTomoLive (Peck et al., Nature Methods 2026, DOI 10.1038/s41592-026-03093-y; bioRxiv 2025.03.11.642690).**
- Three modules: per-tilt motion correction; iterative CTF estimation plus global alignment (tilt-axis angle and translations) plus local alignment; then SART/WBP reconstruction. It runs in real time with a scout-worker multi-GPU design. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.03.11.642690v1.full); [Nature Methods](https://www.nature.com/articles/s41592-026-03093-y)

**Xu, Liu, Niu, He, Zhang, Li, Han (2026), "Markerfree: GPU-accelerated marker-free alignment for improved cryo-ET reconstruction", Structure. DOI 10.1016/j.str.2026.01.007. No arXiv version found.**
- Iterative cross-correlation and common-lines coarse alignment, then global refinement by "enhanced projection matching, which limits the number of images participating in reconstruction while improving their quality". Validated on simulated and real data. — [Europe PMC abstract](https://doi.org/10.1016/j.str.2026.01.007)

**Chaillet, van Loenhout, Leung, Burt, Tegunov (2026), "MissAlignment Teaches Itself Better Cryo-ET Tilt-Series Alignment by Making It Worse", bioRxiv preprint, DOI 10.64898/2026.04.29.721716.**
- A CNN learns to score alignment accuracy with a contrastive loss that needs no well-aligned ground truth (it is trained by perturbing alignments). Gradients back-propagated through the score optimise per-image alignment parameters. Claimed to "significantly outperform" reference-free methods and "rival reference-based alignment". — [Europe PMC preprint abstract](https://doi.org/10.64898/2026.04.29.721716)

**Chen, M. (2026), "Gradient based refinement of CryoET tilt series alignment improves tomogram contrast and structure resolution", bioRxiv. DOI 10.64898/2026.01.16.699989.**
- Gradient-descent refinement of alignment parameters. Reports improved tomogram contrast and higher-resolution subtomogram averages from the same particles. — [Europe PMC preprint abstract](https://doi.org/10.64898/2026.01.16.699989)

**Other**
- Coray et al. 2024 (Dynamo automated fiducial alignment, Structure, DOI 10.1016/j.str.2024.07.003) and Xu et al. 2024 (Markerauto2, Structure, 2024; DOI truncated in my search output) are fiducial-based, so less transferable. — [Europe PMC search results](https://europepmc.org/search?query=tilt%20series%20alignment)
- teamtomo/tttsa provides automated tilt-series alignment in PyTorch. It is a differentiable-framework reference implementation (software, no paper found). — [GitHub tttsa](https://github.com/teamtomo/tttsa)

### Inferences
- Cryo-ET restricts the reference reconstruction to a subset of well-aligned or low-tilt images during projection matching (Markerfree). Applied to TomoJAX, this suggests a leave-one-out or held-out reference: realign view i against a volume reconstructed without view i, or with heavily down-weighted outlier views. This reduces the volume absorbing view i's own pose error, which matters for large initial errors.
- Cryo-ET tilt series are limited-angle (missing wedge), similar to laminography's missing cone. The field's reliance on tilt-axis-angle estimation as a global parameter, separate from per-tilt shifts, argues for estimating TomoJAX's global axis tilt and orientation as a separate low-dimensional parameter before freeing per-view rotations.

### Gaps
- No quantitative capture range in degrees or pixels was available from the abstracts of AreTomo3, Markerfree, MissAlignment or Chen 2026.
- I found no arXiv versions of these cryo-ET papers. They appear to be bioRxiv/journal only.

## Q5. Deep-learning / learned initialisation for pose and motion in CT, and trustworthiness for scientific use

### Takeaway
Learned components in CT motion estimation (Preuhs 2020, Huang 2022, Thies 2024, MissAlignment 2026) are mostly used as learned objectives or quality metrics inside a physics-based gradient optimisation, not as direct pose regressors. Accuracy gains are real: Preuhs reports 0.013 mm residual RPE. However, a learned metric breaks the data-consistency guarantee and can be biased toward the training distribution. For scientific X-ray work, the trustworthy pattern is a learned or heuristic initialiser followed by a final data-consistency (reprojection-residual) refinement and check.

### Cited Findings

**Preuhs, Manhart, Roser, Hoppe, Huang, Psychogios, Kowarschik, Maier (2020), "Appearance Learning for Image-based Motion Estimation in Tomography", arXiv:2006.10390 (IEEE TMI 39(11)).**
- Learns the appearance of rigid-motion artefacts (3 translations + 3 rotations per view), independent of the scanned object, as an objective for motion estimation. "Residual mean RPE of 0.013 mm with an inter-patient standard deviation of 0.022 mm", about twice as accurate as previous results. Applicability is shown on one motion-affected clinical scan. — [arXiv:2006.10390](https://arxiv.org/abs/2006.10390)
- The authors note that pure deep-learning reconstruction can make anatomical malformations vanish, since "the consistency of the reconstructed image to the acquired data is not guaranteed". They motivate a hybrid, physics-consistent approach for "data integrity ... of high importance in a clinical setting". — [arXiv:2006.10390 PDF](https://arxiv.org/pdf/2006.10390)

**(Authors not verified; 2022), "Reference-Free Learning-Based Similarity Metric for Motion Compensation in Cone-Beam CT", PMC9254028 (journal and authors not verified; found only as a search-result title).**
- A learned reference-free similarity metric for motion compensation. I could not retrieve the full text (Europe PMC returned HTTP 500) and found it only through the search listing. — [PMC9254028](https://pmc.ncbi.nlm.nih.gov/articles/PMC9254028/)

**Thies et al. (2024)**
- Uses a learned voxel-wise quality network as the differentiable target, with 3 mm -> 0.61 mm RPE (see Q2). — [arXiv:2401.09283](https://arxiv.org/abs/2401.09283)

**Wang, Yan, Pan, Zhang, Ng, Yu, Wang, Li (2026), "Data-driven deformation correction in X-ray spectro-tomography with implicit neural networks" (CANet), Patterns. DOI 10.1016/j.patter.2026.101515.**
- A self-supervised coordinate network maps projection angular and spectral coordinates to affine transforms. It needs no external training data. Shown on real TXM-XANES battery-cathode data. — [Europe PMC abstract](https://doi.org/10.1016/j.patter.2026.101515)

**Goldmann, Damm, Goldmann, Hornung, Manhart, Preuhs, Kowarschik, Maier (2026), "Measured and synthetic rigid head motion datasets via generative model for motion simulation and compensation in medical imaging", J. Med. Imaging.**
- Provides measured and generatively synthesised rigid-motion trajectories for benchmarking. — [Europe PMC search listing](https://europepmc.org/search?query=rigid%20head%20motion%20datasets%20generative)

**MissAlignment (2026)**
- A self-supervised learned score drives gradient refinement of per-tilt alignment (see Q4). — [bioRxiv preprint DOI](https://doi.org/10.64898/2026.04.29.721716)

### Inferences
- For TomoJAX's scientific-trust requirement, learned components should only propose: initialise, or generate multi-start candidates. Acceptance should be decided by data-consistency metrics: reprojection residual, held-out-view residual (TomoJAX has `validation_residuals.py`), and FSC between half-sets.
- Self-supervised, per-dataset networks (CANet, MissAlignment's contrastive score) avoid training-distribution mismatch better than supervised regressors. They are the more defensible learned option for synchrotron data.

### Gaps
- I found no study that measures failure or hallucination rates of learned pose initialisers on out-of-distribution X-ray nano-tomography or laminography data.
- I could not access the Huang 2022 full text.

## Q6. Benchmarks / protocols for declaring successful recovery and identifiability on noisy data

### Takeaway
I found no community benchmark or standard protocol for success rate of rigid per-view alignment at given initial error and noise in X-ray tomography. The papers report single-trial RMSE (Gürsoy, JI-Faproma), FSC resolution (Nikitin 2021), reprojection error (Preuhs, Thies) or qualitative "graceful degradation" curves (van Leeuwen). TomoJAX's ">=99% on noisy +/-3 deg / +/-10 px cases" gate would be stricter than anything published, so it needs its own protocol.

### Cited Findings
- Gürsoy 2017: shifts from U(-10, 10) px, Gaussian noise at several levels, per-parameter error plotted against a "single pixel bound". There are no success-rate statistics. — [PMC5603591](https://pmc.ncbi.nlm.nih.gov/articles/PMC5603591/)
- van Leeuwen 2018: initial misalignment scaled 2/4/8/16x, recovery judged per parameter type. The tomographic angle fails once error exceeds angular sampling. — [arXiv:1705.08678](https://arxiv.org/abs/1705.08678)
- JI-Faproma 2020: RMSE vs noise level (0–20%) per DOF: 0.12 deg in-plane rotation, 0 px vertical, 1 px horizontal at 20% noise. — [PMC7192921](https://pmc.ncbi.nlm.nih.gov/articles/PMC7192921/)
- Nikitin 2021: half-dataset FSC (1/2-bit criterion) on real data as the quality metric (127 nm vs 192 nm). — [arXiv:2008.03375](https://arxiv.org/abs/2008.03375)
- Thies 2024: mean reprojection error (mm), 3 mm -> 0.61 mm. Thies 2405.19079 characterises recoverable motion frequencies as a function of the motion model. — [arXiv:2401.09283](https://arxiv.org/abs/2401.09283); [arXiv:2405.19079](https://arxiv.org/abs/2405.19079)
- Riis 2021 adds view-angle uncertainty quantification, a route to per-parameter credible intervals. — [Inverse Problems DOI](https://doi.org/10.1088/1361-6420/abf5ba)
- Goldmann 2026 supplies realistic rigid-motion trajectory datasets, which could be used to sample non-uniform motion. — [Europe PMC listing](https://europepmc.org/search?query=rigid%20head%20motion%20datasets%20generative)

### Inferences
Proposed protocol, synthesised from the above:
- (i) Sample at least 300 independent trials per geometry. That gives a 95% Wilson lower bound of about 98.7% when 300/300 succeed; a strict ">=99% with confidence" claim needs about 460 zero-failure trials. Sample per-view errors uniformly in +/-3 deg and +/-10 px, plus smooth-drift and jitter components following Thies' frequency finding.
- (ii) Define success after gauge fixing: per-DOF RMSE below a threshold (for example 0.05 px and 0.01 deg) and no single view off by more than k sigma.
- (iii) Separately report the tomographic-angle DOF, since van Leeuwen shows it is ill-posed beyond the angular step.
- (iv) Report identifiability via the smallest eigenvalues of the reduced (Schur) Hessian, or Riis-style posterior widths, especially for laminography.
- (v) Add real-data checks via half-set FSC and held-out-view residuals.

### Gaps
- No published success-rate benchmark exists for X-ray rigid alignment. TomoBank and similar repositories hold data, but I found no standard misalignment-recovery protocol attached to them.
- No published 0.01 deg per-view rotation accuracy result was found for any real-data X-ray nano-tomography or laminography pipeline. TomoJAX's gate may exceed what the literature demonstrates.
