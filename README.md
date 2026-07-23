# blender-leaf-gen

An add-on to create leaf retopology and procedural leaf shading in Blender

<img width="1200" height="600" alt="Leaf_Gen_1" src="https://github.com/user-attachments/assets/89782012-ccc3-4622-949e-4829ce9ae858" />


#

**Leaf Retopo — Grid Projection + Geometry Nodes**

This add-on provides a complete leaf retopology workflow combined with a procedural shading system. It lets you import a leaf PNG, generate clean quad topology from its silhouette using grid projection, sculpt leaf veins as bump maps, and apply a physically accurate Thin Wall translucency shader — all from within Blender's N panel.

---

# Requirements

The contour detection system (used for precise leaf silhouette extraction) works best with the following Python libraries:

- `numpy`
- `opencv-python`
- `Pillow`

<img width="1897" height="1000" alt="leaf_gen_1" src="https://github.com/user-attachments/assets/ecc5c506-550d-466d-823d-1a922f81bb16" />


**These can be installed directly from the add-on preferences** — no terminal or manual pip commands needed. Go to **Edit** > **Preferences** > **Add-ons** > **Leaf Retopo** and click **Install numpy + opencv-python + Pillow**.

Without these libraries the add-on falls back to a built-in minimal contour method — the workflow remains fully functional, only silhouette precision is reduced.

---

# Installation

1. Download the ZIP file.
2. Open Blender and go to **Edit** > **Preferences** > **Add-ons**.
3. Click **Install from Disk**, select the ZIP file, and confirm.
4. Enable the add-on by checking the corresponding box.
5. *(Optional)* Go to **Edit** > **Preferences** > **Add-ons** > **Leaf Retopo** and click **Install numpy + opencv-python + Pillow** for precise contour detection.
6. Access **Leaf Retopo** in the **N panel** (sidebar) under the **Leaf Retopo** tab.

---

# How to Use — Import & Retopo

### 1. Import Leaf PNG
Select a leaf image (PNG with transparency). The add-on reads the alpha channel to extract the leaf silhouette.

<img width="1897" height="998" alt="leaf_gen_5" src="https://github.com/user-attachments/assets/aa3bfcef-c011-44b1-93ce-038dd72a83e6" />

### 2. Grid Projection
Projects a quad grid onto the leaf surface, snapping to the mesh and trimming to the silhouette boundary. Adjust **Grid Density** to control topology resolution.

<img width="1897" height="998" alt="leaf_gen_6" src="https://github.com/user-attachments/assets/93d15de2-f53a-4a6b-91ed-3cd3fe593a71" />

### 3. Curve Modifier (Optional)
Convert the retopo result to a curve, simplify its control points, and loft it back to quads using the **Loft Curve → Quads** operator.

---

# How to Use — Leaf Veins Relief

Add procedural vein geometry to your leaf using the bump shader workflow.

### Add Veins (Solidify)
Creates a solidified vein structure driven by the leaf texture's luminance channel.

<img width="1897" height="998" alt="leaf_gen_7" src="https://github.com/user-attachments/assets/c34a13cf-daaa-4e3f-8413-9289b9196b35" />

### + Bump Shader
Connects the vein normal map into the material's Bump input, giving the appearance of raised veins without extra geometry.

### Toggle Invert / Remove
Invert the vein direction or remove the entire vein setup in one click.

---

# How to Use — Thin Wall Translucency

Applies a physically accurate two-sided translucency shader that makes leaves react correctly to backlit conditions.

<img width="1897" height="998" alt="leaf_gen_11 (1)" src="https://github.com/user-attachments/assets/68b00ee2-d45e-4fc2-9ff7-14083a3e68e7" />

### Blender 5.2+ (native Thin Wall)
Uses the native **Thin Wall** property on the Principled BSDF, introduced in Blender 5.2. Creates two Principled BSDF nodes mixed via Geometry **Backfacing**:

| | Roughness | IOR | Thin Wall | Transmission |
|---|---|---|---|---|
| **Front** | 0.114 | 3.2 | ✓ | 1.0 |
| **Back** | 0.332 | 1.1 | ✓ | 1.0 |

Both faces receive the leaf texture (Color + Alpha).

### Blender 5.0 / 5.1 (fallback)
Builds an equivalent setup using two Principled BSDF nodes (Roughness 0.673, IOR 1.1, Transmission 1.0) mixed via Backfacing, with a Transparent BSDF alpha mix for correct leaf clipping.

### Remove Thin Wall
Removes the entire Thin Wall node setup and reconnects the original material in one click.

---

# How to Use — Geometry Nodes

Apply procedural deformations to the leaf mesh using the built-in Geometry Nodes presets.

### Create & Apply Bend
Adds a natural curvature to the leaf along its main axis.

<img width="1897" height="998" alt="leaf_gen_8" src="https://github.com/user-attachments/assets/cac4f305-a8ae-448d-a148-03a1181a606a" />


### Create & Apply Subdivide
Applies a Geometry Nodes subdivision pass for smooth silhouettes without baking.

---

# Advanced Options

- **One-click Dependency Installation** : numpy, opencv-python and Pillow install directly from the add-on preferences — no terminal required. Three fallback strategies ensure installation works across all Blender Python environments.
- **Automatic lib detection** : The add-on searches the local `deps/` folder, user site-packages and the standard Python path — libraries are found regardless of which install strategy succeeded.
- **Version-aware Thin Wall** : The shading system detects the running Blender version at click time and automatically picks the native 5.2 path or the 5.0 fallback — no user configuration needed.
- **Non-destructive workflow** : The original Principled BSDF is preserved and repositioned, never deleted — Remove Thin Wall fully restores the previous material state.
- **Shared image path** : The image selected via Import Leaf PNG is shared across the retopo and shading workflows — select once, use everywhere.
- **Blender 5.0 → 5.2 compatible** : Tested on Blender 5.0 and 5.2 LTS. Minimum supported version: 3.0.

---

# Compatibility

| Blender | Thin Wall | Contour (cv2) | Contour (fallback) |
|---|---|---|---|
| 3.x / 4.x | — | ✓ | ✓ |
| 5.0 / 5.1 | Fallback (Translucent) | ✓ | ✓ |
| 5.2 LTS | ✓ Native | ✓ | ✓ |
