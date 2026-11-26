# TomoJAX Studio

<!-- impeccable:product-schema 1 -->

## Platform

Native desktop, built with Rune and TomoJAX's Python engine.

## Users

Imaging scientists preparing, reconstructing and comparing tomography data.
Assumed working context: a desktop or laptop, often with multiple acquisitions
open and a numerical job running while the next scan is prepared.

## Product Purpose

Make a persistent project the home of a tomography workflow: its referenced or
copied acquisitions, preparation settings, reusable recipes, submitted jobs,
derived volumes, comparisons and exports. Reopening a project should restore
useful context. Its files should remain understandable outside the application.

## Capabilities and Constraints

The computational authority is the current TomoJAX implementation and its public
engine/schema contract. The interface discovers supported operations and
parameters there. Inputs and submitted settings are captured for each job;
later editing does not change work already queued. Originals are preserved,
outputs retain provenance, missing sources can be reconnected, and interrupted
jobs can be inspected and retried.

Volumes use `(z, y, x)` and projections use `(view, v, u)`. Geometry and units
remain attached to scientific arrays. Display previews and rendering quality
are distinguishable from quantitative exports. Results, durations, histograms,
measurements and progress come from actual data and events.

## Brand Commitments

The supplied Terrarium brief pins an image-led light interface, warm whites,
restrained sage and forest greens, excellent typography, contextual controls,
and calm, purposeful motion. Scientific images may use dark neutral canvases.
The visual reference is inspiration; the user delegated composition and
interaction decisions.

## Product Principles

- Keep the specimen, its source and the consequences of processing in view.
- Let common tasks be immediate and advanced settings discoverable in context.
- Preserve work across sessions and preserve submitted work while settings change.
- Make navigation responsive while loading, rendering and reconstruction proceed.
- Explain limitations and failure recovery without inventing successful results.

## Accessibility and Inclusion

The complete workflow needs keyboard navigation, visible focus, discoverable and
customizable shortcuts, labeled controls and states that do not rely on color.
Windows should remain useful at the supported minimum size.

## Evidence on Hand

The current TomoJAX source, its public APIs and tests, the v0.1 design brief,
the Terrarium reference image, and actual generated or loaded scientific data.
Screenshots of attractive data are not substitutes for a working renderer.
