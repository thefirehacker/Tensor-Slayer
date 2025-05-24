# Tensor Slaying Toolkit - UI Mockups V2 (Single-Page Monster Toolkit)

This document presents a revised series of ASCII art mockups focusing on a single-page, integrated toolkit with a global view of tensors and dynamic interaction, moving away from a chat-first interface.

## Mockup 1: Initial View - The Tensor Galaxy/Filmstrip

```
+----------------------------------------------------------------------------------------------------------------------+
| Tensor Slaying Toolkit                                                     Model: Qwen_0.6B (Editable) | PatchFile: none |
+----------------------------------------------------------------------------------------------------------------------+
| [Global Controls: Load Model | Load PatchFile | Save PatchFile | Export Patched Model | Settings | Help ]                |
+----------------------------------------------------------------------------------------------------------------------+
| [ Tensor Canvas (Zoomable/Pannable - scroll L/R or Up/Down or Zoom In/Out) ]                                         |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [L.0.SA.Q]---[L.0.SA.K]---[L.0.SA.V]---[L.0.SA.O]---[L.0.MLP.G]---[L.0.MLP.U]---[L.0.MLP.D]---[L.1.SA.Q] ...   | |
| |  (tiny heatmap) (th)       (th)       (th)       (th)        (th)        (th)        (th)                     | |
| |                                                                                                                  | |
| | [L.1.SA.K]---[L.1.SA.V]---[L.1.SA.O]---[L.1.MLP.G]---[L.1.MLP.U]---[L.1.MLP.D]---[L.2.SA.Q]---[L.2.SA.K] ...   | |
| |  (th)       (th)       (th)       (th)        (th)        (th)        (th)        (th)                     | |
| | ... (potentially many rows/columns of tensor thumbnails if laid out on a 2D grid) ...                              | |
| |                                                                                                                  | |
| | [embed_tokens]--------------------[lm_head]----------------------[final_norm]                                   | |
| |  (larger thumbnail due to size?) (larger th)                      (smaller th)                                   | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
| [ Bottom Dock: Quick Query / Filter ]                                                                                |
| [ Query: [Highlight tensors where mean > 0.01_________________________________________] [Go] ]                     |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels (Tabs or Accordion): [Focus View] [Investigation] [Patch Editor] [Patch Manager] ]             |
+----------------------------------------------------------------------------------------------------------------------+
```
(th) = tiny heatmap thumbnail. Names are abbreviated for space.

## Mockup 2: Focus View - Tensor Drill-Down (Right Panel Activated)

(User clicks `[L.0.MLP.G]` thumbnail on the Canvas)
```
+----------------------------------------------------------------------------------------------------------------------+
| Tensor Slaying Toolkit                                                                                               |
+----------------------------------------------------------------------------------------------------------------------+
| [ Tensor Canvas (L.0.MLP.G is highlighted/slightly larger, others dimmed/smaller) ]                                    |
| +------------------------------------------------------------------------------------------------------------------+ |
| | ... [L.0.SA.O]---{#L.0.MLP.G#}---[L.0.MLP.U] ...                                                                  | |
| |                   ( FOCUSED )                                                                                      | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
| [ Bottom Dock: Quick Query / Filter ]                                                                                |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels ]                                                                                              |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [Focus: L.0.MLP.G] [Investigation] [Patch Editor] [Patch Manager]                                                  | |
| |------------------------------------------------------------------------------------------------------------------| |
| | Slice: D0 [  0] to [ 32] D1 [  0] to [ 32] [View] | [ Full Tensor Stats ]                                        | |
| | +---------------------------------------------+  | [ Associated Vocab (if embed) ]                            | |
| | | Heatmap of L.0.MLP.G (32x32 slice)          |  | [ X-Ref with other tensors... ]                            | |
| | | ############**####                        |  |                                                            | |
| | | ###**#######**##                        |  | [ > SLAYING TOOLS < ]                                      | |
|   | | **##########**####                        |  |   - Edit Value at (X,Y): [ ] New: [_______] [Set]        | |
| | | (Hover for value/idx)                     |  |   - Select Region [Drag on Heatmap]                        | |
| | +---------------------------------------------+  |   - Analyze Selection / Apply Op to Selection...           | |
| |                                               |  |   - View/Edit Raw Hex for Slice...                       | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
```

## Mockup 3: Integrated Toolkit - Investigation Panel

```
+----------------------------------------------------------------------------------------------------------------------+
| ... (Canvas remains, L.0.MLP.G might still be in focus or canvas reset) ...                                          |
+----------------------------------------------------------------------------------------------------------------------+
| [ Bottom Dock: Quick Query / Filter ]                                                                                |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels ]                                                                                              |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [Focus View] [Investigation (Active)] [Patch Editor] [Patch Manager]                                               | |
| |------------------------------------------------------------------------------------------------------------------| |
| | Query Input:                                                                                                     | |
| | [ What tensors are most involved in negating prompts about Tiananmen Square?____________________] [Investigate] | |
| |                                                                                                                  | |
| | Results:                                                                                                         | |
| | - Sys: Analysis suggests high activation in these layers when prompt is encountered:                             | |
| |   - [model.layers.27.mlp.up_proj] (Confidence: 0.85) - Click to Focus                                          | |
| |   - [model.layers.15.self_attn.o_proj] (Confidence: 0.72) - Click to Focus                                     | |
| |   - Specific patterns in [model.embed_tokens.weight] also noted.                                                 | |
| | - Sys: Values in highlighted regions of `mlp.up_proj` seem to suppress positive sentiment tokens.                | |
| | [Export Investigation Log]                                                                                       | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
```

## Mockup 4: Integrated Toolkit - Patch Editor & Staging (Hex Diff Focus)

```
+----------------------------------------------------------------------------------------------------------------------+
| ... (Canvas with L.0.MLP.G in focus, showing a small visual marker for patched region) ...                           |
+----------------------------------------------------------------------------------------------------------------------+
| [ Bottom Dock: Quick Query / Filter ]                                                                                |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels ]                                                                                              |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [Focus View] [Investigation] [Patch Editor (Active)] [Patch Manager]                                               | |
| |------------------------------------------------------------------------------------------------------------------| |
| | Current Tensor Focus: model.layers.27.mlp.up_proj (Slice: 0-32, 0-32)                                            | |
| |                                                                                                                  | |
| | Staged Hex Diffs for this Tensor (from direct edits or applied ops):                                             | |
| | - Offset: 0x001A0 (Slice Coord: 5,10) Old: FF3C0A11 New: 00000000 [Revert]                                        | |
| | - Offset: 0x002B8 (Slice Coord: 10,22) Old: AB10CDFF New: 00000000 [Revert]                                        | |
| |                                                                                                                  | |
| | [ Direct Hex Edit for Focused Slice (Advanced) ]                                                                   | |
| |   Address | Hex Values (Editable)                 | ASCII (Attempted)                                          | |
| |   0x001A0 | [00 00 00 00] 01 23 45 67 89 AB CD EF | ...E.sUV....                                               | |
| |   0x001B0 | DE AD BE EF CA FE BA BE ...           | ........                                                   | |
| | [Apply Direct Hex Edits to Stage]                                                                                | |
| |                                                                                                                  | |
| | [Clear Staged Diffs for this Tensor] [Commit All Staged Diffs to PatchFile]                                      | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
```

## Mockup 5: Visual Query on Canvas (Multi-Select & Analyze)

```
+----------------------------------------------------------------------------------------------------------------------+
| ...                                                                                                                  |
| [ Tensor Canvas (User has Ctrl-Clicked/Box-Selected multiple tensor thumbnails) ]                                    |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [*L.0.SA.Q*]---[L.0.SA.K]---[*L.0.SA.V*]---[L.0.SA.O]---[*L.0.MLP.G*]---[L.0.MLP.U] ... (Selected marked by *) | |
| |                                                                                                                  | |
| | ... other rows also with potential selections ...                                                                  | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
| [ Bottom Dock: Contextual Multi-Select Query ]                                                                       |
| [ Query for 3 Selected Tensors: [What statistical properties do these share?________________] [Analyze Group] ]     |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels: Investigation (showing group analysis results) ]                                              |
| > Sys: Comparing selected tensors: L.0.SA.Q, L.0.SA.V, L.0.MLP.G                                                     |
|   - All have similar sparsity levels (~20-25%).                                                                      |
|   - L.0.SA.Q and L.0.SA.V show strong negative correlation in their diagonal elements.                               |
|   - [View Combined Heatmap (Overlay/Diff)] [Export Group Stats]                                                      |
+----------------------------------------------------------------------------------------------------------------------+
```

## Mockup 6: Linking Heatmap Colors to Values/Tokens (Focus View)

```
+----------------------------------------------------------------------------------------------------------------------+
| ... (Canvas, Bottom Dock) ...                                                                                        |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels ]                                                                                              |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [Focus: model.embed_tokens.weight] [Investigation] [Patch Editor] ...                                            | |
| |------------------------------------------------------------------------------------------------------------------| |
| | Slice: D0 [1000] to [1032] D1 [  0] to [ 32] [View] (Rows are token embeddings)                                   | |
| | +---------------------------------------------+  | [ Value Inspector (Drag selection on heatmap) ]            | |
| | | Heatmap of embed_tokens                   |  |------------------------------------------------------------| |
| | | Row 1005: ##[XX]#### (XX selected by user) |  | Selected Region (1005, 5-7):                               | |
|   | | Row 1006: ###YY##### (YY other color)     |  |   - Avg Color Value (Normalized): 0.8 (Greenish)           | |
| | | ...                                         |  |   - Actual Values: [0.75, 0.81, 0.79]                      | |
| | +---------------------------------------------+  |   - Hex: [3F400000, 3F4F0000, 3F4A0000]                  | |
| |                                               |  |   - Token ID for Row 1005: 872 -> "Tiananmen"            | |
| |                                               |  |   [Find other tensors with similar value patterns?]        | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
```

## Mockup 7: Global Filter/Highlight on Canvas (Bottom Dock Query)

```
+----------------------------------------------------------------------------------------------------------------------+
| ... (Global Controls) ...                                                                                            |
+----------------------------------------------------------------------------------------------------------------------+
| [ Tensor Canvas (Some thumbnails are highlighted/glowing, others are dimmed) ]                                       |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [L.0.SA.Q]---[dim:L.0.SA.K]---[L.0.SA.V]---[glow:L.0.SA.O]---[L.0.MLP.G]---[dim:L.0.MLP.U] ...                   | |
| |                                                                                                                  | |
| | [glow:embed_tokens]--------------[dim:lm_head]----------------------[final_norm]                                   | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
| [ Bottom Dock: Quick Query / Filter ]                                                                                |
| [ Query: [Show tensors with >50% zero values AND dtype=float16_________________________] [Apply Filter/Highlight] ] |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels: (Possibly a summary of filtered tensors) ]                                                    |
+----------------------------------------------------------------------------------------------------------------------+
```

## Mockup 8: Side-by-Side Tensor Comparison View (New Tab in Right Panel?)

```
+----------------------------------------------------------------------------------------------------------------------+
| ... (Canvas, Bottom Dock) ...                                                                                        |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels ]                                                                                              |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [Focus View] [Investigation] [Patch Editor] [Patch Manager] [Compare (Active)]                                   | |
| |------------------------------------------------------------------------------------------------------------------| |
| | Select Tensor A: [model.layers.0.mlp.gate_proj ▼] | Select Tensor B: [model.layers.1.mlp.gate_proj ▼]            | |
| | +---------------------------+ +---------------------------+ | [ Stats Diff ] [ Value Corr. ]                   | |
| | | Heatmap A (Slice)         | | Heatmap B (Slice)         | |                                                  | |
| | | ##########                | | **********                | |                                                  | |
| | | ##**####                  | | **####**##                | |                                                  | |
| | +---------------------------+ +---------------------------+ |                                                  | |
| | Stats A: Min -0.5, Max 1.2  | Stats B: Min -0.3, Max 1.0  |                                                  | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
```

## Mockup 9: "Tensor Autopsy" Mode (Dedicated Layout for One Tensor)

(Could be a modal, or a special state of the Focus View)
```
+----------------------------------------------------------------------------------------------------------------------+
| Tensor Autopsy: model.layers.27.mlp.up_proj                                                   [ Close Autopsy ]    |
+----------------------------------------------------------------------------------------------------------------------+
| [ Basic Info: Shape, DType, SizeMB ] [ Full Stats ] [ Raw Hex View (Scrollable) ] [ Associated Vocab (if any) ]      |
+----------------------------------------------------------------------------------------------------------------------+
| +-------------------------------------------+ +--------------------------------------------------------------------+ |
| | Heatmap (Full or Large Slice)             | | Investigation Notes / User Annotations (Scrollable)                | |
| |                                           | |--------------------------------------------------------------------| |
| | (Interactive heatmap)                     | | - 2024-05-25: High values in quadrant 2 linked to query "CCP".     | |
| |                                           | | - UserTag: #censorship_vector                                      | |
| |                                           | | - AutoNote: Correlates strongly with layer.26.o_proj on X metric.  | |
| +-------------------------------------------+ | [ Add Note/Tag... ]                                                | |
|                                             +--------------------------------------------------------------------+ |
| +-------------------------------------------+ +--------------------------------------------------------------------+ |
| | Staged Patches for this Tensor            | | Related Tensors (via Investigation/Similarity)                     | |
| |-------------------------------------------| |--------------------------------------------------------------------| |
| | - Slice(50,100) set to 0.0 [Revert]     | | - [model.layers.26.o_proj] (Similarity: 0.8)                       | |
| | ...                                       | | - [model.embed_tokens] (Influence Score: 0.65 for current query)   | |
| +-------------------------------------------+ +--------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
```

## Mockup 10: Hex-Diff Patch File Manager Panel

```
+----------------------------------------------------------------------------------------------------------------------+
| ... (Canvas, Bottom Dock) ...                                                                                        |
+----------------------------------------------------------------------------------------------------------------------+
| [ Right Docked Panels ]                                                                                              |
| +------------------------------------------------------------------------------------------------------------------+ |
| | [Focus View] [Investigation] [Patch Editor] [Patch Manager (Active)]                                               | |
| |------------------------------------------------------------------------------------------------------------------| |
| | Current Base Model: Qwen_0.6B.safetensors                                                                        | |
| | Loaded Patch File: [ccp_censorship_fix_v3.hexpatch] [Browse...] [New PatchFile]                                  | |
| |                                                                                                                  | |
| | Description for ccp_censorship_fix_v3.hexpatch:                                                                  | |
| | [Attempt to reduce CCP keyword suppression, focusing on MLP layers._________________________________________] [Save Desc]|
| |                                                                                                                  | |
| | Patches in this file (Applied to Base Model):                                                                    | |
| | 1. Tensor: model.layers.27.mlp.up_proj, Offset: 0x001A0, Len: 4, Old: FF3C0A11, New: 00000000 [View on Heatmap]   |
| | 2. Tensor: model.embed_tokens.weight, Offset: 0x15320, Len: 12, Old: ..., New: ... [View on Heatmap]              |
| | ...                                                                                                              | |
| | [Verify Patch Integrity] [Export Patched Model using this PatchFile...]                                          | |
| +------------------------------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------------------------+
``` 