---
name: TomoJAX Studio
description: A calm native workspace for persistent tomography projects.
colors:
  forest: "#315D46"
  sage: "#E2EADF"
  background: "#F5F6F0"
  paper: "#FCFDF9"
  panel: "#F0F3EC"
  ink: "#24382D"
  muted: "#596A5D"
  line: "#DCE3D8"
  canvas: "#111A17"
  canvas-ink: "#EDF2EA"
  canvas-muted: "#B1BEB4"
  warning: "#8B5E27"
  error: "#A3493B"
  axis-z: "#94BE9A"
  axis-y: "#D6B991"
  axis-x: "#A9B4D6"
typography:
  headline:
    fontFamily: "Inter, sans-serif"
    fontSize: "24px"
    fontWeight: 600
  title:
    fontFamily: "Inter, sans-serif"
    fontSize: "22px"
    fontWeight: 600
  body:
    fontFamily: "Inter, sans-serif"
    fontSize: "12px"
    fontWeight: 400
  action:
    fontFamily: "Inter, sans-serif"
    fontSize: "12px"
    fontWeight: 600
  label:
    fontFamily: "Inter, sans-serif"
    fontSize: "11px"
    fontWeight: 400
  numeric:
    fontFamily: "Inter, sans-serif"
    fontSize: "12px"
    fontWeight: 450
    fontFeature: "'tnum' 1"
rounded:
  field: "5px"
  control: "6px"
  thumbnail: "7px"
  canvas: "10px"
  dialog: "12px"
spacing:
  tight: "4px"
  control: "8px"
  group: "12px"
  section: "18px"
  workspace: "20px"
  inspector: "22px"
components:
  button-primary:
    backgroundColor: "{colors.forest}"
    textColor: "{colors.paper}"
    typography: "{typography.action}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "36px"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.action}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "36px"
  text-field:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.numeric}"
    rounded: "{rounded.field}"
    padding: "0 10px"
    height: "36px"
  image-canvas:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.canvas-ink}"
    rounded: "{rounded.canvas}"
  selected-view:
    backgroundColor: "{colors.sage}"
    textColor: "{colors.forest}"
    rounded: "{rounded.thumbnail}"
---

# Design System: TomoJAX Studio

## Overview

**Creative North Star: "Terrarium"**

Terrarium is a light desktop workspace for sustained scientific work. Warm
surfaces and restrained greens give the project and controls a consistent
identity. Scientific images occupy dark, neutral canvases with their own clear
labels and interaction hints.

The image remains the visual center of the application. The project navigator
explains where it came from; the inspector exposes the next useful operation.
Keep controls compact enough for an expert workflow and separate groups with
space, alignment and quiet seams.

**Key Characteristics:**

- Warm white chrome with forest actions and sage selections.
- Large scientific canvases surrounded by stable project context.
- Compact native controls with visible focus and explicit labels.
- Tabular values and descriptive states rather than decorative metrics.

This document records the implementation in [tokens.rs](src/ui/tokens.rs),
[controls.rs](src/ui/controls.rs) and the adjacent native views. Measurements in
the interface are logical desktop points. The token format uses `px` to preserve
the same numeric values in portable design tooling.

## Colors

Forest carries primary actions and active navigation. Sage identifies selected
rows, view labels and restrained grouping. Paper, panel and background create
subtle layers without changing the overall light appearance.

Ink is the main text color; muted is for secondary information that remains
readable. Line separates persistent regions and outlines secondary controls.
Warning and error always accompany words or a labeled icon. Axis colors
distinguish the canonical Z, Y and X slice directions; axis names remain visible.

The canvas palette belongs to scientific viewing. Its background establishes a
consistent surround, while image colors come from the selected scientific
colormap. Changing a colormap must change the rendered data mapping explicitly.

## Typography

Rune supplies embedded Inter. Use its sans family throughout the application;
numbers use tabular figures in the same family. The headline and title roles
establish the active specimen and inspector. Body and label roles carry controls
and technical context without uppercase section ornaments.

Functional text starts at 11 points with line height at 1.35 times the font size.
Use semibold for actions, selected navigation and short headings. Keep explanatory
text regular. Truncate long asset names in dense lists, and expose the complete
name or path through native help. Wrap explanations, validation errors and
required parameter guidance.

## Layout

The default window is 1540 × 980 points; the minimum is 1180 × 780. The topbar is
64 points high, the project navigator is 228 points wide, the inspector is 304
points wide, and the status line is 28 points high. The image workspace takes the
remaining space. Navigator and inspector can be hidden independently.

Below 1320 points, the command search label shortens to Search and navigation
spacing contracts. The inspector scrolls internally while its primary action
stays visible. Its paired fields shrink within the available width; labels and
input text must never overlap their neighbor. The project tree, jobs list and
modal results have their own bounded scrolling regions.

The central composition uses one large active image and a linked strip below it.
Promoting a slice changes the main canvas while preserving the project and
controls. Fit 2D scientific images according to their physical aspect ratio.
Axis labels follow array metadata, including arrays whose storage order differs
from the usual `(z, y, x)` convention.

## Elevation & Depth

Persistent regions use tonal surfaces and one-point seams. Buttons and job rows
use a border without a shadow. Command search and shortcut dialogs float above a
dimmed workspace and use Rune's native shadow treatment. The jobs drawer retains
the current specimen as context.

Pointer-driven image changes commit immediately. Progress is determined by real
worker events; an indeterminate stage stays textual until a progress value is
available. Avoid decorative motion around scientific data or status changes.

## Shapes

Controls have modest rounded corners. Text fields are slightly tighter than
buttons; image surfaces and dialogs have the larger radii. Keep rectangular
images rectangular inside their fitted viewport. The outer workspace can clip
its corners without cropping the scientific image itself.

Icons are authored line paths in a consistent 24-unit view box with 1.5-unit
strokes. Use the same set for the tree, toolbar, jobs and dialogs. Every icon-only
action has a visible focus state, an accessible name and native help.

## Components

### Actions and fields

Primary actions use forest and paper. Secondary actions use paper with a quiet
border. Standard buttons are 36 points high; fixed inspector actions are 40
points high. Icon actions use a 32-point target. Hover and press states are
restrained overlays, with a distinct forest focus ring for keyboard use.

Fields have persistent labels, optional unit suffixes and a focus-within outline.
Common numeric values remain directly editable. Structured operation parameters
retain an explicit JSON form; guided correction controls use the engine schema
and preserve correction order when the user switches editing modes.

### Navigation and lists

Prepare, Reconstruct, Explore, Compare and Jobs are workspaces inside one project.
The active workspace has forest text and a short underline. The asset tree keeps
derived results under their source. Selection, missing sources and saved state
must be described with text as well as color.

Job rows show the submitted inputs, status, worker stage and applicable controls.
Pause, resume and cancel stay beside the heading, above long task histories.
The default running record shows its current task tree and solver plots;
expanding a row reveals the full history, metric values and captured settings.
Errors remain attached to the failed
job so another successful run cannot erase the processing record.

### Scientific views

Main and linked views use the same scientific data and axis vocabulary.
Thumbnail overlays use the thumbnail transform; main overlays use its zoom and
pan. Measurement overlays belong to a particular asset and displayed slice.
Display window, colormap, opacity and clipping controls do not change the stored
quantitative values.

Image-quality labels distinguish sampled previews from full-resolution planes.
Comparison states disclose a grid or unit mismatch before suggesting a shared
physical interpretation. Loading, missing data and empty states explain the next
available action.

### Command and shortcut dialogs

Command search opens with focus in the query field. Its results show the action,
category and current binding. Arrow keys select; Enter runs; Escape dismisses.
Shortcut editing uses the same command vocabulary, shows conflicts before saving,
and offers restoration of the default binding.

## Do's and Don'ts

### Do

- Do preserve the image, source context and next action when adding a workspace.
- Do label physical axes and units from scientific metadata.
- Do keep primary actions visible when their inspector content scrolls.
- Do use real data, worker events and stored provenance for scientific content.
- Do check keyboard focus, long names and the minimum supported window size.

### Don't

- Don't turn display preferences into changes to quantitative data.
- Don't imply registration when compared arrays use different grids or units.
- Don't replace real scientific images or progress with illustrative placeholders.
- Don't add shadows to persistent panels or decorate each control as a card.
- Don't hide invalid input behind an apparently successful operation.
