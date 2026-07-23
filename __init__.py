# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "Leaf Retopo - Grid Projection + GN (inline create & apply)",
    "author": "Guillaume Bissieres",
    "version": (1, 0, 3),
    "blender": (5, 0, 1),
    "location": "View3D > Sidebar > Leaf Retopo",
    "description": "An add-on to create leaf retopology and procedural leaf shading in Blender",
    "category": "3D View",
    "doc_url": "https://bissieres.gumroad.com/l/LeafGen",
}

import bpy
import bmesh
import typing
import math
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy.props import IntProperty, FloatProperty, BoolProperty, StringProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper

# -----------------------------------------------------------------------
# Dependency management
# Packages: numpy, opencv-python, Pillow
# -----------------------------------------------------------------------
import subprocess
import importlib

# (import_name, pip_name)
_LEAF_DEPS = [
    ("numpy",  "numpy"),
    ("cv2",    "opencv-python"),
    ("PIL",    "Pillow"),
]

cv2         = None
np          = None
PIL_Image   = None
PIL_ImageFilter = None


def _try_import_deps():
    """Try to import optional deps; updates module-level globals.
    Also ensures the local deps/ folder and user site-packages are on sys.path."""
    global cv2, np, PIL_Image, PIL_ImageFilter
    import sys, os

    # Add local deps/ folder (--target install fallback)
    local_deps = os.path.join(os.path.dirname(__file__), "deps")
    if os.path.isdir(local_deps) and local_deps not in sys.path:
        sys.path.insert(0, local_deps)

    # Add user site-packages (--user install)
    try:
        import site
        user_site = site.getusersitepackages()
        if user_site and os.path.isdir(user_site) and user_site not in sys.path:
            sys.path.insert(0, user_site)
    except Exception:
        pass

    try:
        import importlib
        import cv2 as _cv2
        import numpy as _np
        cv2 = _cv2
        np  = _np
    except ImportError:
        cv2 = None
        np  = None

    try:
        from PIL import Image as _PilImage, ImageFilter as _PilImageFilter
        PIL_Image       = _PilImage
        PIL_ImageFilter = _PilImageFilter
    except ImportError:
        PIL_Image       = None
        PIL_ImageFilter = None


def _dep_status():
    """Return list of (pip_name, is_installed) for all leaf deps."""
    results = []
    for imp, pip in _LEAF_DEPS:
        try:
            importlib.import_module(imp)
            results.append((pip, True))
        except ImportError:
            results.append((pip, False))
    return results


def _all_deps_ok():
    return all(ok for _, ok in _dep_status())


def _pip_target_dir():
    """Return the user-writable site-packages dir where Blender extensions can install."""
    import site, sys, os
    # Prefer user site dir (writable without admin rights)
    user_site = site.getusersitepackages()
    if user_site and os.path.exists(os.path.dirname(user_site)):
        os.makedirs(user_site, exist_ok=True)
        return user_site
    # Fallback: alongside the __init__.py of this addon
    return os.path.join(os.path.dirname(__file__), "deps")


def _install_dep(pip_name, report_fn=None):
    import sys, os
    python = sys.executable
    target = _pip_target_dir()

    # Make sure target is on sys.path so the import works right after install
    if target not in sys.path:
        sys.path.insert(0, target)

    # Try several strategies in order
    strategies = [
        # 1. Plain upgrade (works if pip is not locked)
        [python, "-m", "pip", "install", "--upgrade", pip_name],
        # 2. --user (most common workaround for Blender-managed Python)
        [python, "-m", "pip", "install", "--upgrade", "--user", pip_name],
        # 3. --target into addon/deps (last resort, always writable)
        [python, "-m", "pip", "install", "--upgrade",
         "--target", target, pip_name],
    ]

    for cmd in strategies:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                if report_fn:
                    report_fn(f"✓ {pip_name} installed")
                return True
            # If "externally-managed" error, try next strategy
            if "externally-managed" in result.stderr or "externally managed" in result.stderr:
                if report_fn:
                    report_fn(f"  → pip blocked, retrying with next strategy…")
                continue
        except Exception as e:
            if report_fn:
                report_fn(f"  → error: {e}")
            continue

    # All strategies failed — report the last error
    if report_fn:
        report_fn(f"✗ {pip_name}: all install strategies failed. "
                  f"Last stderr: {result.stderr[-300:]}")
    return False


# Run on first import
_try_import_deps()

# Constants
ALPHA_THRESHOLD = 10
BLUR_RADIUS = 1.2
DEFAULT_SOLIDIFY_THICKNESS = 0.001

# ----------------------------
# Helpers
# ----------------------------

def ensure_object_mode():
    if bpy.ops.object.mode_set.poll():
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass


def safe_vec(seq, fallback=(0.0, 0.0, 1.0)):
    try:
        if seq is None:
            return Vector(fallback)
        if isinstance(seq, Vector):
            return seq.copy()
        return Vector((float(seq[0]), float(seq[1]), float(seq[2])))
    except Exception:
        return Vector(fallback)


def object_bounds_local(obj):
    """Return local-space min and max Vector of the object's bounding box.

    Safe fallback if bound_box is unavailable.
    """
    try:
        bb = [Vector(b) for b in obj.bound_box]
        min_v = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
        max_v = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
        return min_v, max_v
    except Exception:
        return Vector((-0.5, -0.5, -0.5)), Vector((0.5, 0.5, 0.5))


# ----------------------------
# Image -> Leaf (NGON) import (REPLACED BY Version21 improved implementation)
# ----------------------------

# --- helpers: RDP, Chaikin, resample ---
def rdp(points, eps):
    if not points or len(points) < 3:
        return points
    first = points[0]; last = points[-1]
    x1, y1 = first; x2, y2 = last
    denom = math.hypot(x2 - x1, y2 - y1)
    max_dist = -1.0; index = -1
    for i in range(1, len(points) - 1):
        x0, y0 = points[i]
        if denom == 0:
            dist = math.hypot(x0 - x1, y0 - y1)
        else:
            dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denom
        if dist > max_dist:
            index = i; max_dist = dist
    if max_dist > eps:
        left = rdp(points[:index + 1], eps)
        right = rdp(points[index:], eps)
        return left[:-1] + right
    else:
        return [first, last]

def chaikin_smooth(points, iterations=2):
    if not points or len(points) < 3:
        return points
    pts = points[:]
    for _ in range(iterations):
        new = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]; p1 = pts[(i + 1) % n]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new.append(q); new.append(r)
        pts = new
    return pts

def resample_arc_length(points, target_count):
    if not points or len(points) < 2:
        return points
    dists = []
    total = 0.0
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        dists.append(dist); total += dist
    if total == 0:
        return [points[0]] * target_count
    out = []
    step = total / float(target_count)
    accum = 0.0
    idx = 0
    for k in range(target_count):
        t = k * step
        while idx < len(dists) and (accum + dists[idx]) < t:
            accum += dists[idx]; idx += 1
        if idx >= len(dists):
            out.append(points[-1]); continue
        a, b = points[idx], points[(idx + 1) % len(points)]
        seg_len = dists[idx] if dists[idx] != 0 else 1e-12
        seg_t = (t - accum) / seg_len
        x = a[0] + (b[0] - a[0]) * seg_t
        y = a[1] + (b[1] - a[1]) * seg_t
        out.append((x, y))
    return out

# --- small morphological open (3x3) using numpy (fast) ---
def morph_open_3x3(mask):
    if np is None:
        return mask
    pad = np.pad(mask, 1, mode='constant', constant_values=False)
    H, W = mask.shape
    eroded = np.ones_like(mask, dtype=bool)
    eroded &= pad[0:H, 0:W]; eroded &= pad[0:H, 1:W+1]; eroded &= pad[0:H, 2:W+2]
    eroded &= pad[1:H+1, 0:W]; eroded &= pad[1:H+1, 1:W+1]; eroded &= pad[1:H+1, 2:W+2]
    eroded &= pad[2:H+2, 0:W]; eroded &= pad[2:H+2, 1:W+1]; eroded &= pad[2:H+2, 2:W+2]
    pad2 = np.pad(eroded, 1, mode='constant', constant_values=False)
    dilated = np.zeros_like(eroded, dtype=bool)
    dilated |= pad2[0:H, 0:W]; dilated |= pad2[0:H, 1:W+1]; dilated |= pad2[0:H, 2:W+2]
    dilated |= pad2[1:H+1, 0:W]; dilated |= pad2[1:H+1, 1:W+1]; dilated |= pad2[1:H+1, 2:W+2]
    dilated |= pad2[2:H+2, 0:W]; dilated |= pad2[2:H+2, 1:W+1]; dilated |= pad2[2:H+2, 2:W+2]
    return dilated

# Moore trace
def _moore_boundary_trace(mask, start_y, start_x):
    H, W = mask.shape
    nbrs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    s_y, s_x = start_y, start_x
    b_y, b_x = s_y, s_x - 1
    boundary = []
    current_y, current_x = s_y, s_x
    first = True
    guard = 0
    init_prev = (b_y, b_x)
    while True:
        boundary.append((current_x, current_y))
        found_index = None
        for idx,(dy,dx) in enumerate(nbrs):
            if current_y + dy == b_y and current_x + dx == b_x:
                found_index = idx; break
        if found_index is None: found_index = 0
        next_found = None
        for k in range(1,9):
            idx = (found_index + k) % 8
            dy,dx = nbrs[idx]
            ny, nx = current_y + dy, current_x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                next_found = (ny, nx)
                prev_idx = (idx - 1) % 8
                b_y = current_y + nbrs[prev_idx][0]; b_x = current_x + nbrs[prev_idx][1]
                break
        if next_found is None: break
        if next_found[0] == s_y and next_found[1] == s_x and b_y == init_prev[0] and b_x == init_prev[1] and not first:
            break
        first = False
        current_y, current_x = next_found
        guard += 1
        if guard > (H * W * 4): break
    return boundary

# Robust mesh creation (bmesh) and exact layout mapping
def create_leaf_mesh_main(outer, holes, w, h, image_path, name,
                          solidify_thickness=0.0,
                          add_solidify=False,
                          blend_method='CLIP',
                          alpha_threshold=0.05,
                          pixels_per_unit=100.0,
                          preserve_exact_layout=True,
                          layout_mode='CENTERED',
                          auto_fit=True,
                          max_dimension=0.5,
                          simplify_tolerance=1.5,
                          smooth_iterations=2,
                          upsample_factor=2):
    # This function is an improved, robust mesh creation ported from Version21.
    # It creates an ngon mesh from an outer contour + optional holes, maps UVs and material.
    import bmesh

    def contour_to_xy(cnt):
        pts = []
        for p in cnt:
            try:
                if isinstance(p[0], (list, tuple)) or (np is not None and hasattr(p[0], "dtype")):
                    px = float(p[0][0]); py = float(p[0][1])
                else:
                    px = float(p[0]); py = float(p[1])
            except Exception:
                try:
                    px = float(p[0][0]); py = float(p[0][1])
                except Exception:
                    px,py = 0.0,0.0
            pts.append((px,py))
        return pts

    outer_pts = contour_to_xy(outer) if outer is not None else []
    holes_pts = [contour_to_xy(h) for h in (holes or [])]
    if not outer_pts:
        return None

    # Simplify small spikes
    eps = max(0.0, float(simplify_tolerance))
    if eps > 0.0:
        try:
            outer_pts = rdp(outer_pts, eps)
            holes_pts = [rdp(h, eps) for h in holes_pts]
        except Exception:
            pass

    # resample and smooth for stable ngons
    perim = 0.0
    for i in range(len(outer_pts)):
        a = outer_pts[i]; b = outer_pts[(i+1)%len(outer_pts)]
        perim += math.hypot(b[0]-a[0], b[1]-a[1])
    target = min(max(64, int(perim / 2)), 8192)
    try:
        outer_pts = resample_arc_length(outer_pts, target)
    except Exception:
        pass
    try:
        outer_pts = chaikin_smooth(outer_pts, iterations=max(0, int(smooth_iterations)))
    except Exception:
        pass

    # compute component center (pixel coords)
    xs = [p[0] for p in outer_pts]; ys = [p[1] for p in outer_pts]
    minx_c, maxx_c = min(xs), max(xs)
    miny_c, maxy_c = min(ys), max(ys)
    comp_cx = (minx_c + maxx_c) / 2.0
    comp_cy = (miny_c + maxy_c) / 2.0
    width_px = maxx_c - minx_c; height_px = maxy_c - miny_c

    # image reference: either full image or bounding box of non-transparent region
    img_minx, img_miny, img_maxx, img_maxy = 0.0, 0.0, float(w), float(h)
    if not preserve_exact_layout:
        # try compute non-transparent bbox if PIL & numpy present
        if PIL_Image is not None and np is not None:
            try:
                pil = PIL_Image.open(image_path).convert("RGBA")
                alpha_arr = np.array(pil.split()[-1])
                nz = np.argwhere(alpha_arr > ALPHA_THRESHOLD)
                if nz.size:
                    miny_px, minx_px = nz.min(axis=0)
                    maxy_px, maxx_px = nz.max(axis=0)
                    img_minx = float(minx_px); img_miny = float(miny_px)
                    img_maxx = float(maxx_px); img_maxy = float(maxx_px)
            except Exception:
                img_minx, img_miny, img_maxx, img_maxy = 0.0, 0.0, float(w), float(h)
    # else preserve_exact_layout True => use full image extents (img_minx... = 0..w/h)

    # scale px->BU — auto_fit uses FULL IMAGE size so all leaves share the same scale
    ppu = max(1e-12, float(pixels_per_unit))
    base_scale = 1.0 / ppu
    scale_auto = 1.0
    if auto_fit and float(max_dimension) > 0.0:
        # Use full image dimensions so every leaf is scaled identically
        img_w_bu = float(w) * base_scale
        img_h_bu = float(h) * base_scale
        maxdim = max(img_w_bu, img_h_bu)
        if maxdim > 0.0 and maxdim > float(max_dimension):
            scale_auto = float(max_dimension) / maxdim
    final_scale = base_scale * scale_auto

    # local verts centered on component center
    verts_local = []
    for (px,py) in outer_pts:
        lx = (px - comp_cx) * final_scale
        ly = (comp_cy - py) * final_scale
        verts_local.append((lx, ly, 0.0))
    holes_local = []
    for hpts in holes_pts:
        loop = []
        for (px,py) in hpts:
            lx = (px - comp_cx) * final_scale
            ly = (comp_cy - py) * final_scale
            loop.append((lx, ly, 0.0))
        holes_local.append(loop)

    # build mesh robustly via bmesh
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm_verts_outer = [bm.verts.new(co) for co in verts_local]
    bm.verts.ensure_lookup_table()
    outer_edges = []
    n = len(bm_verts_outer)
    for i in range(n):
        a = bm_verts_outer[i]; b = bm_verts_outer[(i+1)%n]
        try:
            e = bm.edges.new((a,b))
        except ValueError:
            e = bm.edges.get((a,b)) or bm.edges.get((b,a))
        outer_edges.append(e)
    for loop in holes_local:
        bm_h_verts = [bm.verts.new(co) for co in loop]
        bm.verts.ensure_lookup_table()
        for i in range(len(bm_h_verts)):
            a = bm_h_verts[i]; b = bm_h_verts[(i+1)%len(bm_h_verts)]
            try:
                e = bm.edges.new((a,b))
            except ValueError:
                e = bm.edges.get((a,b)) or bm.edges.get((b,a))
    face_created = False
    try:
        res = bmesh.ops.contextual_create(bm, geom=outer_edges)
        if res and 'faces' in res and len(res['faces'])>0:
            face_created = True
    except Exception:
        pass
    if not face_created:
        try:
            b_edges = [e for e in bm.edges if e.is_boundary]
            if b_edges:
                res2 = bmesh.ops.holes_fill(bm, edges=b_edges, sides=0)
                if res2 and 'faces' in res2:
                    face_created = True
        except Exception:
            pass
    if not face_created:
        try:
            bm.faces.new(bm_verts_outer); face_created = True
        except Exception:
            pass
    bm.normal_update()
    bm.to_mesh(mesh); bm.free()

    # create object and compute world location using chosen mapping
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # ALWAYS use full image pixel dimensions as reference so ALL leaves from
    # the same image land in the same coordinate space → relative layout preserved.
    # comp_cx/comp_cy are in full image pixel coords (0..w, 0..h).
    full_w_bu = float(w) * final_scale
    full_h_bu = float(h) * final_scale

    # In Blender: X = right, Y = up. Image: X = right, Y = down.
    # Flip Y: pixel_y=0 (top) → world_y = +full_h_bu/2 when centered.
    if layout_mode == 'CENTERED':
        world_x =  comp_cx * final_scale - full_w_bu / 2.0
        world_y = -comp_cy * final_scale + full_h_bu / 2.0
    elif layout_mode == 'TOPLEFT_AT_ORIGIN':
        world_x =  comp_cx * final_scale
        world_y = -comp_cy * final_scale
    else:  # BOTTOMLEFT_AT_ORIGIN
        world_x =  comp_cx * final_scale
        world_y =  full_h_bu - comp_cy * final_scale

    obj.location = (world_x, world_y, 0.0)

    # cleanup
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=1e-6)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    # UVs (map to full image so textures align)
    try:
        mesh_uv = obj.data
        uv_layer = mesh_uv.uv_layers.new(name="UVMap")
        raw_px = outer_pts[:]
        for hpts in holes_pts:
            raw_px.extend(hpts)
        for loop in mesh_uv.loops:
            vidx = loop.vertex_index
            if vidx < len(raw_px):
                px,py = raw_px[vidx]
                u = px / float(w) if w else 0.0
                v = 1.0 - (py / float(h) if h else 0.0)
                uv_layer.data[loop.index].uv = (u,v)
        mesh_uv.uv_layers.active = uv_layer
    except Exception:
        pass

    # material
    try:
        mat = bpy.data.materials.new(name + "_Mat"); mat.use_nodes = True
        nodes = mat.node_tree.nodes; links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf_front = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf_back  = nodes.new("ShaderNodeBsdfPrincipled")
        tex = nodes.new("ShaderNodeTexImage")
        geom = nodes.new("ShaderNodeNewGeometry")
        mix = nodes.new("ShaderNodeMixShader")
        uv_node = nodes.new("ShaderNodeUVMap")

        # Matte leaf — no reflections, texture always visible under light
        for bsdf in (bsdf_front, bsdf_back):
            try: bsdf.inputs['Roughness'].default_value = 1.0
            except (KeyError, AttributeError): pass
            try: bsdf.inputs['Specular IOR Level'].default_value = 0.0  # Blender 4+
            except (KeyError, AttributeError):
                try: bsdf.inputs['Specular'].default_value = 0.0        # Blender 3
                except (KeyError, AttributeError): pass
            try: bsdf.inputs['IOR'].default_value = 1.0
            except (KeyError, AttributeError): pass
            try: bsdf.inputs['Metallic'].default_value = 0.0
            except (KeyError, AttributeError): pass
            try: bsdf.inputs['Sheen Weight'].default_value = 0.0        # Blender 4+
            except (KeyError, AttributeError):
                try: bsdf.inputs['Sheen'].default_value = 0.0           # Blender 3
                except (KeyError, AttributeError): pass
            try: bsdf.inputs['Coat Weight'].default_value = 0.0         # Blender 4+
            except (KeyError, AttributeError):
                try: bsdf.inputs['Clearcoat'].default_value = 0.0       # Blender 3
                except (KeyError, AttributeError): pass
        try:
            if obj.data.uv_layers.active:
                uv_node.uv_map = obj.data.uv_layers.active.name
        except Exception:
            pass
        img = None
        try:
            img = bpy.data.images.load(image_path)
        except Exception:
            img = bpy.data.images.get(bpy.path.basename(image_path))
        if img is None:
            obj.data.materials.append(mat)
        else:
            try:
                if hasattr(img, "alpha_mode"):
                    img.alpha_mode = 'STRAIGHT'
            except Exception:
                pass
            tex.image = img
            try:
                tex.image.colorspace_settings.name = 'sRGB'
            except Exception:
                pass
            try:
                links.new(uv_node.outputs.get("UV"), tex.inputs.get("Vector"))
                links.new(tex.outputs.get("Color"), bsdf_front.inputs.get("Base Color"))
                links.new(tex.outputs.get("Color"), bsdf_back.inputs.get("Base Color"))
            except Exception:
                pass
            if "Alpha" in tex.outputs:
                try:
                    links.new(tex.outputs["Alpha"], bsdf_front.inputs["Alpha"])
                    links.new(tex.outputs["Alpha"], bsdf_back.inputs["Alpha"])
                except Exception:
                    pass
            try:
                links.new(bsdf_front.outputs.get("BSDF"), mix.inputs[1])
                links.new(bsdf_back.outputs.get("BSDF"), mix.inputs[2])
                links.new(geom.outputs.get("Backfacing"), mix.inputs[0])
                links.new(mix.outputs.get("Shader"), out.inputs.get("Surface"))
            except Exception:
                pass
            try:
                method = blend_method.upper() if isinstance(blend_method, str) else 'CLIP'
                if method not in ('CLIP','HASHED','BLEND'): method='CLIP'
                mat.blend_method = method
                if mat.blend_method == 'CLIP':
                    try: mat.alpha_threshold = float(alpha_threshold)
                    except Exception: pass
                mat.use_backface_culling = False
                try: mat.shadow_method = 'NONE'
                except Exception: pass
            except Exception:
                pass
            obj.data.materials.append(mat)
    except Exception:
        pass

    if add_solidify and solidify_thickness and solidify_thickness != 0.0:
        try:
            mod = obj.modifiers.new(name="Leaf_Solidify", type='SOLIDIFY')
            mod.thickness = float(solidify_thickness)
            mod.offset = 0.0; mod.use_even_offset = True; mod.use_rim = True
        except Exception:
            pass

    return obj

# PIL fallback improved with upsampling refinement and morpho cleaning
def detect_and_create_leaves_pil(path, create_fn,
                                 simplify_tolerance=1.5,
                                 morph_clean=True,
                                 blur_radius=1.2,
                                 upsample_factor=2):
    if PIL_Image is None or np is None or PIL_ImageFilter is None:
        return False
    try:
        pil = PIL_Image.open(path).convert("RGBA")
    except Exception:
        return False
    # upsample to reduce quantization and improve edge accuracy
    uf = max(1, int(upsample_factor))
    W, H = pil.size
    if uf > 1:
        try:
            pil_up = pil.resize((W*uf, H*uf), resample=PIL_Image.Resampling.LANCZOS)
        except Exception:
            pil_up = pil
    else:
        pil_up = pil
    try:
        pil_blur = pil_up.filter(PIL_ImageFilter.GaussianBlur(radius=blur_radius))
    except Exception:
        pil_blur = pil_up
    alpha = np.array(pil_blur.split()[-1])
    mask = alpha > ALPHA_THRESHOLD
    # morphological cleaning on upsampled mask
    if morph_clean:
        try:
            mask = morph_open_3x3(mask)
        except Exception:
            pass
    H2, W2 = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions = []
    for y in range(H2):
        for x in range(W2):
            if mask[y, x] and not visited[y, x]:
                stack = [(y, x)]; visited[y, x] = True; comp = []
                while stack:
                    cy, cx = stack.pop()
                    comp.append((cy, cx))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0<=ny<H2 and 0<=nx<W2 and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True; stack.append((ny, nx))
                regions.append(comp)
    if not regions:
        return True
    # compute overall bbox on upsampled mask
    nz = np.argwhere(mask)
    if nz.size:
        miny_up, minx_up = nz.min(axis=0); maxy_up, maxx_up = nz.max(axis=0)
    else:
        minx_up, miny_up, maxx_up, maxy_up = 0,0,W2-1,H2-1
    # process each region: trace boundary on upsampled mask, then scale down coords
    for i, comp in enumerate(regions):
        start = None
        for (cy, cx) in comp:
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = cy+dy, cx+dx
                if not (0<=ny<H2 and 0<=nx<W2 and mask[ny,nx]):
                    start = (cy, cx); break
            if start: break
        if start is None:
            start = comp[0]
        boundary_up = _moore_boundary_trace(mask, start[0], start[1])
        if not boundary_up:
            continue
        # scale down by upsample factor to original pixel coords (center of upsample pixel -> /uf)
        pts = [ (float(x)/uf, float(y)/uf) for (x,y) in boundary_up ]
        # resample uniformly to keep density manageable
        perim = 0.0
        for k in range(len(pts)):
            a = pts[k]; b = pts[(k+1)%len(pts)]
            perim += math.hypot(b[0]-a[0], b[1]-a[1])
        target = min(max(64, int(perim / 2)), 8192)
        try:
            pts = resample_arc_length(pts, target)
        except Exception:
            pass
        # simplify and smooth
        if simplify_tolerance and simplify_tolerance > 0.0:
            try:
                pts = rdp(pts, simplify_tolerance)
            except Exception:
                pass
        try:
            pts = chaikin_smooth(pts, iterations=max(0, int(2)))
        except Exception:
            pass
        contour = [[[int(round(x)), int(round(y))]] for (x,y) in pts]
        try:
            create_fn(contour, [], W, H, path, f"Leaf_{i}")
        except Exception:
            pass
    return True

# minimal fallback if no libs
def detect_and_create_leaves_simple(path, create_fn):
    try:
        img = bpy.data.images.load(path)
    except Exception:
        img = bpy.data.images.get(bpy.path.basename(path))
    if img is None:
        return False
    try:
        pixels = list(img.pixels)
    except Exception:
        return False
    w = img.size[0]; h = img.size[1]
    if w==0 or h==0:
        return False
    minx=w;miny=h;maxx=0;maxy=0;found=False
    for y in range(h):
        for x in range(w):
            idx = (y*w + x)*4
            a = pixels[idx+3]
            if a > (ALPHA_THRESHOLD/255.0):
                found=True
                minx=min(minx,x); maxx=max(maxx,x)
                miny=min(miny,y); maxy=max(maxy,y)
    if not found:
        return False
    contour = [[[minx,miny]],[[maxx,miny]],[[maxx,maxy]],[[minx,maxy]]]
    try: create_fn(contour, [], w, h, path, "Leaf_simple")
    except Exception: return False
    return True

# dispatcher: prefer cv2, else improved PIL fallback, else simple
def detect_and_create_leaves(path, create_fn, simplify_tolerance=1.5, upsample_factor=2):
    # cv2 path if available
    if cv2 is not None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim>=3 and img.shape[2]>=4:
                h,w = img.shape[:2]
                alpha = img[:,:,3]
                alpha = cv2.GaussianBlur(alpha, (0,0), 1.2)
                _, mask = cv2.threshold(alpha, ALPHA_THRESHOLD, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                if contours:
                    for i in range(len(contours)):
                        if hierarchy is None or hierarchy[0][i][3] != -1:
                            continue
                        outer = contours[i]
                        holes = []
                        child = hierarchy[0][i][2]
                        while child != -1:
                            holes.append(contours[child]); child = hierarchy[0][child][0]
                        # optionally simplify cv2 contour
                        if simplify_tolerance and simplify_tolerance > 0.0:
                            try:
                                pts = [(int(p[0][0]), int(p[0][1])) for p in outer]
                                pts = rdp(pts, simplify_tolerance)
                                outer = [[[x,y]] for (x,y) in pts]
                            except Exception:
                                pass
                        create_fn(outer, holes, w, h, path, f"Leaf_{i}")
                    return
        except Exception:
            pass
    # PIL fallback improved
    if PIL_Image is not None and np is not None and PIL_ImageFilter is not None:
        try:
            if detect_and_create_leaves_pil(path, create_fn, simplify_tolerance, morph_clean=True, blur_radius=1.2, upsample_factor=upsample_factor):
                return
        except Exception:
            pass
    # minimal fallback
    detect_and_create_leaves_simple(path, create_fn)

# ----------------------------
# Import operator (replaced with improved operator from Version21,
# but keep the UI name requested "Import Leaf PNG (Pure NGON)")
# ----------------------------

class ImportLeafClean(bpy.types.Operator, ImportHelper):
    bl_idname = "import_image.leaf_clean"
    bl_label = "Import Leaf PNG (Pure NGON)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png", options={'HIDDEN'})

    add_solidify: BoolProperty(
        name="Add Solidify Modifier",
        default=False,
        description="Add a small Solidify modifier to avoid transparency z-sorting issues"
    )
    solidify_thickness: FloatProperty(
        name="Solidify Thickness",
        default=DEFAULT_SOLIDIFY_THICKNESS,
        description="Thickness to use for the Solidify modifier (scene units)"
    )
    blend_method: EnumProperty(
        name="Blend Mode",
        items=(('CLIP', "Alpha Clip", "Fast, no sorting artefacts (hard edges)"),
               ('HASHED', "Alpha Hashed", "Dithered alpha"),
               ('BLEND', "Alpha Blend", "Smooth alpha but can exhibit sorting artefacts in Eevee"),),
        default='CLIP'
    )
    alpha_threshold: FloatProperty(
        name="Alpha Clip Threshold",
        default=0.05, min=0.0, max=1.0
    )

    pixels_per_unit: FloatProperty(
        name="Pixels per Unit",
        default=500.0,
        min=1e-6,
        description="Number of image pixels that equal 1 Blender unit (200 => 200 px = 1 BU)"
    )
    preserve_exact_layout: BoolProperty(
        name="Preserve Exact Layout (use full image coords)",
        default=True,
        description="Map objects to the full image pixel coordinates (no bbox cropping)."
    )
    layout_mode: EnumProperty(
        name="Layout Mode",
        items=(('CENTERED','Centered at origin',''),
               ('TOPLEFT_AT_ORIGIN','Top-left at origin',''),
               ('BOTTOMLEFT_AT_ORIGIN','Bottom-left at origin','')),
        default='CENTERED'
    )
    auto_fit: BoolProperty(name="Auto Fit", default=True)
    max_dimension: FloatProperty(name="Max Dimension (BU)", default=2.0, min=1e-6)
    simplify_tolerance: FloatProperty(name="Simplify Tolerance (px)", default=1.0, min=0.0)
    smooth_iterations: IntProperty(name="Smooth Iterations", default=2, min=0, max=6)
    upsample_factor: IntProperty(name="Upsample Factor (PIL)", default=2, min=1, max=8)

    def execute(self, context):
        if cv2 is not None:
            self.report({'INFO'}, "Using OpenCV for contour extraction.")
        elif PIL_Image is not None and np is not None:
            self.report({'INFO'}, "Using Pillow + numpy fallback (refined).")
        else:
            self.report({'WARNING'}, "No cv2/Pillow+numpy available — using minimal fallback (less precise).")

        def create_fn(outer, holes, w, h, image_path, name):
            return create_leaf_mesh_main(
                outer, holes, w, h, image_path, name,
                solidify_thickness=self.solidify_thickness,
                add_solidify=self.add_solidify,
                blend_method=self.blend_method,
                alpha_threshold=self.alpha_threshold,
                pixels_per_unit=self.pixels_per_unit,
                preserve_exact_layout=self.preserve_exact_layout,
                layout_mode=self.layout_mode,
                auto_fit=self.auto_fit,
                max_dimension=self.max_dimension,
                simplify_tolerance=self.simplify_tolerance,
                smooth_iterations=self.smooth_iterations,
                upsample_factor=self.upsample_factor
            )

        detect_and_create_leaves(self.filepath, create_fn, simplify_tolerance=self.simplify_tolerance, upsample_factor=self.upsample_factor)

        # ensure normals consistent for created objects
        for obj in list(bpy.context.selected_objects):
            try:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                try:
                    bpy.ops.object.mode_set(mode='OBJECT')
                except Exception:
                    pass

        return {'FINISHED'}


# ------------------ Project / retopo operator (kept) ------------------
class OBJECT_OT_project_grid_retopo(bpy.types.Operator):
    """Projeter une grille (soit issue du cutter, soit générée) sur la feuille via raycast BVH."""
    bl_idname = "object.project_grid_retopo"
    bl_label = "Project Grid Retopo (Final - robust fix)"
    bl_options = {'REGISTER', 'UNDO'}

    fallback_rows: IntProperty(name="Fallback Rows", default=51, min=2, max=2048)
    fallback_cols: IntProperty(name="Fallback Cols", default=51, min=2, max=2048)
    merge_distance: FloatProperty(name="Merge Distance", default=0.000001)
    remove_cutter_after: BoolProperty(name="Remove Cutter After", default=False)
    copy_material: BoolProperty(name="Copy Material from Target", default=True)
    start_offset_multiplier: FloatProperty(name="Ray start offset multiplier", default=2.0)

    def execute(self, context):
        ensure_object_mode()
        sel_meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if len(sel_meshes) < 1:
            self.report({'ERROR'}, "Sélectionne la feuille (objet actif).")
            return {'CANCELLED'}

        target = context.view_layer.objects.active
        if not target or target.type != 'MESH':
            self.report({'ERROR'}, "L'objet actif doit être la feuille (mesh).")
            return {'CANCELLED'}

        cutters = [o for o in sel_meshes if o != target]
        cutter = cutters[0] if cutters else None
        if cutter is None:
            for o in context.scene.objects:
                if o.type == 'MESH' and ("LeafCutter" in o.name or "LeafCutter_Edges" in o.name):
                    cutter = o
                    break
        if cutter is None:
            self.report({'ERROR'}, "Aucun cutter trouvé. Crée le cutter via 'Create Edge Grid Cutter' ou sélectionne le cutter correct.")
            return {'CANCELLED'}

        rows = cutter.get("grid_rows", None)
        cols = cutter.get("grid_cols", None)
        use_generated_grid = False
        if rows is None or cols is None:
            rows = int(self.fallback_rows)
            cols = int(self.fallback_cols)
            use_generated_grid = True

        world_points = []
        if (not use_generated_grid) and ("grid_rows" in cutter.keys() and "grid_cols" in cutter.keys()):
            try:
                world_points = [cutter.matrix_world @ v.co for v in cutter.data.vertices]
            except Exception:
                use_generated_grid = True

        if use_generated_grid:
            try:
                if len(cutter.data.vertices) > 0:
                    local_coords = [v.co.copy() for v in cutter.data.vertices]
                    xs = [v.x for v in local_coords]
                    ys = [v.y for v in local_coords]
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                else:
                    bb = cutter.bound_box
                    coords_bb = [Vector(b) for b in bb]
                    xs = [v.x for v in coords_bb]
                    ys = [v.y for v in coords_bb]
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                xrange = max(1e-6, maxx - minx)
                yrange = max(1e-6, maxy - miny)
                pts_local = []
                for r in range(rows):
                    fy = 0.0 if rows == 1 else (r / (rows - 1))
                    y = miny + fy * yrange
                    for c in range(cols):
                        fx = 0.0 if cols == 1 else (c / (cols - 1))
                        x = minx + fx * xrange
                        pts_local.append(Vector((x, y, 0.0)))
                world_points = [cutter.matrix_world @ p for p in pts_local]
            except Exception:
                self.report({'ERROR'}, "Impossible de générer la grille depuis le cutter.")
                return {'CANCELLED'}

        deps = context.evaluated_depsgraph_get()
        eval_obj = target.evaluated_get(deps)
        try:
            bvh = BVHTree.FromObject(target, deps)
        except Exception:
            try:
                mesh_eval = eval_obj.to_mesh()
                bvh = BVHTree.FromMesh(mesh_eval)
                try:
                    eval_obj.to_mesh_clear()
                except Exception:
                    pass
            except Exception:
                self.report({'ERROR'}, "Impossible de construire le BVHTree pour la feuille.")
                return {'CANCELLED'}

        try:
            cutter_normal_world = cutter.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
            if cutter_normal_world.length == 0.0:
                cutter_normal_world = Vector((0.0, 0.0, 1.0))
            cutter_normal_world.normalize()
        except Exception:
            cutter_normal_world = Vector((0.0, 0.0, 1.0))

        bbox = getattr(target, "dimensions", None)
        if not bbox:
            bbox = Vector((1.0, 1.0, 1.0))
        maxdim = max(bbox.x, bbox.y, bbox.z, 1.0)
        start_offset = maxdim * max(0.0001, float(self.start_offset_multiplier))

        projected_positions = [None] * len(world_points)
        for i, wpt in enumerate(world_points):
            try:
                if not isinstance(wpt, Vector):
                    wpt = safe_vec(wpt)
            except Exception:
                projected_positions[i] = None
                continue

            origin = wpt + cutter_normal_world * start_offset
            direction = -cutter_normal_world
            try:
                hit = bvh.ray_cast(origin, direction)
            except Exception:
                hit = None
            if hit is None:
                origin2 = wpt - cutter_normal_world * start_offset
                try:
                    hit = bvh.ray_cast(origin2, cutter_normal_world)
                except Exception:
                    hit = None
                if hit is None:
                    projected_positions[i] = None
                else:
                    loc, nrm, idx, dist = hit
                    projected_positions[i] = Vector(loc)
            else:
                loc, nrm, idx, dist = hit
                projected_positions[i] = Vector(loc)

        new_mesh = bpy.data.meshes.new(f"LeafRetopo_{target.name}_mesh")
        new_obj = bpy.data.objects.new(f"LeafRetopo_{target.name}", new_mesh)
        bpy.context.collection.objects.link(new_obj)

        verts = []
        vert_map = {}
        for i, pos in enumerate(projected_positions):
            if pos is None:
                vert_map[i] = -1
            else:
                vert_map[i] = len(verts)
                verts.append((pos.x, pos.y, pos.z))

        faces = []
        def idx_rc(r, c):
            return r * cols + c

        for r in range(rows - 1):
            for c in range(cols - 1):
                i0 = idx_rc(r, c)
                i1 = idx_rc(r, c + 1)
                i2 = idx_rc(r + 1, c + 1)
                i3 = idx_rc(r + 1, c)
                vi0 = vert_map.get(i0, -1)
                vi1 = vert_map.get(i1, -1)
                vi2 = vert_map.get(i2, -1)
                vi3 = vert_map.get(i3, -1)
                if vi0 >= 0 and vi1 >= 0 and vi2 >= 0 and vi3 >= 0:
                    faces.append((vi0, vi1, vi2, vi3))

        if len(verts) == 0 or len(faces) == 0:
            try:
                bpy.data.objects.remove(new_obj, do_unlink=True)
            except Exception:
                pass
            self.report({'ERROR'}, "Projection n'a renvoyé aucune face — vérifiez la grille/couverture du cutter sur la feuille.")
            return {'CANCELLED'}

        new_mesh.from_pydata(verts, [], faces)
        new_mesh.update()

        try:
            new_obj.matrix_world = Matrix.Identity(4)
        except Exception:
            pass

        bpy.context.view_layer.objects.active = new_obj
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.merge_by_distance(threshold=self.merge_distance)
            if self.copy_material and len(target.data.materials) > 0:
                new_mesh.materials.append(target.data.materials[0])
            try:
                bpy.ops.mesh.tris_convert_to_quads(quad_method='BEAUTY', ngon_method='BEAUTY')
            except Exception:
                pass
            try:
                bpy.ops.mesh.normals_make_consistent(inside=False)
            except Exception:
                pass
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass

        if self.remove_cutter_after:
            try:
                bpy.data.objects.remove(cutter, do_unlink=True)
            except Exception:
                try:
                    cutter.select_set(True)
                    bpy.ops.object.delete()
                except Exception:
                    pass

        for o in list(bpy.context.scene.objects):
            o.select_set(False)
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj

        self.report({'INFO'}, f"Retopo terminé: '{new_obj.name}' ({len(verts)} verts, {len(faces)} faces).")
        return {'FINISHED'}


# -------------------------------------------------------
# Curve simplify utilities & operators
# -------------------------------------------------------

def mesh_from_evaluated_curve(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(deps)
    me = eval_obj.to_mesh()

    verts = [eval_obj.matrix_world @ v.co for v in me.vertices]

    adj = {i: set() for i in range(len(verts))}
    for e in me.edges:
        a, b = e.vertices
        adj[a].add(b)
        adj[b].add(a)

    loops = []
    visited = set()

    for start in range(len(verts)):
        if start in visited or len(adj[start]) < 2:
            continue

        loop = [start]
        prev = None
        cur = start

        while True:
            visited.add(cur)
            nxt = None
            for v in adj[cur]:
                if v != prev:
                    nxt = v
                    break
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt

        if len(loop) >= 3:
            loops.append([verts[i] for i in loop])

    eval_obj.to_mesh_clear()
    return loops


def polygon_arc_lengths(pts):
    cum = [0.0]
    s = 0.0
    for i in range(len(pts)):
        s += (pts[(i + 1) % len(pts)] - pts[i]).length
        cum.append(s)
    return cum, s


def resample_closed_polygon(pts, count):
    cum, total = polygon_arc_lengths(pts)
    res = []
    if total == 0:
        return [pts[0]] * count
    for i in range(count):
        t = (i / count) * total
        if t >= total:
            t = total - 1e-12
        for j in range(len(pts)):
            if cum[j] <= t < cum[j + 1]:
                a = pts[j]
                b = pts[(j + 1) % len(pts)]
                seg_len = (cum[j + 1] - cum[j])
                if seg_len == 0:
                    f = 0.0
                else:
                    f = (t - cum[j]) / seg_len
                res.append(a.lerp(b, f))
                break
    return res


class OBJECT_OT_simplify_curve_points(bpy.types.Operator):
    bl_idname = "object.simplify_curve_points"
    bl_label = "Simplify Curve"
    bl_description = "Reduce number of points on the selected curve by resampling (keeps shape)"
    bl_options = {'REGISTER', 'UNDO'}

    target_points: IntProperty(name="Target Points", default=128, min=4, max=8192)
    create_backup: BoolProperty(name="Create Backup", default=True,
                                description="Duplicate the original object before modifying it")

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'ERROR'}, "No active object")
            return {'CANCELLED'}

        loops = mesh_from_evaluated_curve(obj)
        if not loops:
            self.report({'ERROR'}, "Could not evaluate a closed contour from the active object")
            return {'CANCELLED'}

        outer = loops[0]
        resampled = resample_closed_polygon(outer, self.target_points)

        if self.create_backup:
            backup = obj.copy()
            if obj.data:
                backup.data = obj.data.copy()
            backup.name = obj.name + "_backup"
            context.collection.objects.link(backup)

        inv = obj.matrix_world.inverted()
        local_pts = [inv @ p for p in resampled]

        if obj.type == 'CURVE' and isinstance(obj.data, bpy.types.Curve):
            crv = obj.data
            crv.splines.clear()
            spline = crv.splines.new('POLY')
            spline.points.add(len(local_pts) - 1)
            for i, p in enumerate(local_pts):
                spline.points[i].co = (p.x, p.y, p.z, 1.0)
            spline.use_cyclic_u = True
            crv.dimensions = '3D'
            self.report({'INFO'}, f"Curve simplified to {len(local_pts)} points")
            return {'FINISHED'}
        else:
            new_curve_data = bpy.data.curves.new(obj.name + "_simplified_crv", type='CURVE')
            new_curve_data.dimensions = '3D'
            spline = new_curve_data.splines.new('POLY')
            spline.points.add(len(local_pts) - 1)
            for i, p in enumerate(local_pts):
                spline.points[i].co = (p.x, p.y, p.z, 1.0)
            spline.use_cyclic_u = True
            new_obj = bpy.data.objects.new(obj.name + "_simplified", new_curve_data)
            new_obj.matrix_world = obj.matrix_world.copy()
            context.collection.objects.link(new_obj)
            bpy.ops.object.select_all(action='DESELECT')
            new_obj.select_set(True)
            context.view_layer.objects.active = new_obj
            self.report({'INFO'}, f"Created new simplified curve '{new_obj.name}' with {len(local_pts)} points")
            return {'FINISHED'}


# -------------------------------------------------------
# Loft operator
# -------------------------------------------------------

def make_loft_mesh(rings, name, cap_center_ngon):
    verts = []
    faces = []
    ring_len = len(rings[0])

    for r in rings:
        for v in r:
            verts.append(v)

    for i in range(len(rings) - 1):
        a = i * ring_len
        b = (i + 1) * ring_len
        for j in range(ring_len):
            faces.append((
                a + j,
                a + (j + 1) % ring_len,
                b + (j + 1) % ring_len,
                b + j
            ))

    if cap_center_ngon:
        base = (len(rings) - 1) * ring_len
        faces.append(tuple(range(base, base + ring_len)))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


class OBJECT_OT_curve_loft_to_quads(bpy.types.Operator):
    bl_idname = "object.curve_loft_to_quads"
    bl_label = "Loft Curve → Quad Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    # Defaults adjusted to match screenshot:
    rings: IntProperty(name="Rings", default=3, min=2, max=256)
    points_per_ring: IntProperty(name="Points per Ring", default=1000, min=8, max=2048)
    shrink_ratio: FloatProperty(name="Shrink Strength", default=0.03, min=0.0, max=5.0)
    cap_center: BoolProperty(name="Cap Center (n-gon)", default=True)
    cap_center_quads: BoolProperty(name="Cap Center (Quads)", default=False)
    center_ratio: FloatProperty(name="Center Ring Ratio", default=0.0, min=0.0001, max=1.0)
    output_name: StringProperty(name="Output Name", default="LoftedQuad")

    def execute(self, context):
        curve = context.active_object
        if curve is None or (curve.type != 'CURVE' and curve.type != 'MESH'):
            self.report({'ERROR'}, "Select a curve (or converted curve mesh)")
            return {'CANCELLED'}

        loops = mesh_from_evaluated_curve(curve)
        if not loops:
            self.report({'ERROR'}, "No closed contour found")
            return {'CANCELLED'}

        outer = resample_closed_polygon(loops[0], self.points_per_ring)

        centroid = Vector((0.0, 0.0, 0.0))
        for p in outer:
            centroid += p
        centroid /= len(outer)

        rings = []
        for k in range(self.rings):
            s = k / max(1, self.rings - 1)
            scale = 1.0 - s * self.shrink_ratio
            if scale < 0.0:
                scale = 0.0
            if self.cap_center_quads and k == (self.rings - 1):
                cr = max(self.center_ratio, 1e-6)
                scale = cr

            ring = []
            for p in outer:
                ring.append(centroid + (p - centroid) * scale)
            rings.append(ring)

        add_ngon_cap = self.cap_center and not self.cap_center_quads

        obj = make_loft_mesh(rings, self.output_name, add_ngon_cap)

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        return {'FINISHED'}


# ------------------ Convert to Curve ------------------

class OBJECT_OT_convert_to_curve(bpy.types.Operator):
    """Convert the active mesh object to a Curve (Blender's convert)"""
    bl_idname = "object.convert_to_curve"
    bl_label = "Convert to Curve"
    bl_description = "Convert the active mesh object to a Curve using Blender's convert operator"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ensure_object_mode()
        obj = context.view_layer.objects.active
        if obj is None:
            self.report({'ERROR'}, "Aucun objet actif")
            return {'CANCELLED'}
        if obj.type != 'MESH':
            self.report({'ERROR'}, "L'objet actif doit être un mesh pour être converti en courbe")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        try:
            bpy.ops.object.convert(target='CURVE')
        except Exception as e:
            self.report({'ERROR'}, f"Échec de la conversion: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"'{obj.name}' converti en Curve")
        return {'FINISHED'}


# ------------------ Decimate & grid cut & tris->quads etc. ------------------

def find_view3d_override():
    wm = bpy.context.window_manager
    for window in wm.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        override = {
                            'window': window,
                            'screen': screen,
                            'area': area,
                            'region': region,
                            'scene': bpy.context.scene,
                            'edit_object': bpy.context.edit_object,
                            'active_object': bpy.context.active_object,
                        }
                        return override
    return None


class MESH_OT_delta_quad_plus(bpy.types.Operator):
    """Pipeline: triangulate ngons -> tris->quads -> dissolve limited -> optional decimate"""
    bl_idname = "mesh.delta_quad_plus"
    bl_label = "Decimate"
    bl_options = {'REGISTER', 'UNDO'}

    apply_decimate: BoolProperty(
        name="Apply Decimate",
        default=True,
        description="Apply the Decimate modifier after the pipeline"
    )
    decimate_ratio: FloatProperty(
        name="Ratio",
        default=0.05,
        min=0.0, max=1.0,
        description="Decimate ratio"
    )
    decimate_type: EnumProperty(
        name="Mode",
        items=(
            ('COLLAPSE', "Collapse", ""),
            ('UNSUBDIV', "Un-Subdivide", ""),
            ('PLANAR', "Planar", ""),
        ),
        default='COLLAPSE',
        description="Decimate method"
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh")
            return {'CANCELLED'}

        face_threshold = 0.0
        shape_threshold = 0.0

        prev_mode = obj.mode
        if prev_mode != 'EDIT':
            try:
                bpy.ops.object.mode_set(mode='EDIT')
            except Exception:
                pass

        me = obj.data
        try:
            bm = bmesh.from_edit_mesh(me)
        except Exception as e:
            self.report({'ERROR'}, f"Could not get bmesh in Edit mode: {e}")
            return {'CANCELLED'}

        try:
            for f in bm.faces:
                f.select = (len(f.verts) > 4)
            bmesh.ops.triangulate(bm,
                                  faces=[f for f in bm.faces if f.select],
                                  quad_method='BEAUTY', ngon_method='BEAUTY')
            bmesh.update_edit_mesh(me)
        except Exception as e:
            self.report({'WARNING'}, f"Triangulation failed: {e}")

        override = find_view3d_override()
        try:
            if override:
                bpy.ops.mesh.tris_convert_to_quads(override,
                                                   uvs=False, seam=False,
                                                   face_threshold=face_threshold,
                                                   shape_threshold=shape_threshold)
            else:
                bpy.ops.mesh.tris_convert_to_quads(uvs=False, seam=False,
                                                   face_threshold=face_threshold,
                                                   shape_threshold=shape_threshold)
        except Exception as e:
            self.report({'WARNING'}, f"tris_convert_to_quads failed: {e}")
        finally:
            bmesh.update_edit_mesh(me)

        if self.apply_decimate:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
                mod = obj.modifiers.new(name="DeltaQuad_Decimate", type='DECIMATE')
                mod.decimate_type = self.decimate_type
                if self.decimate_type == 'COLLAPSE':
                    mod.ratio = self.decimate_ratio
                elif self.decimate_type == 'UNSUBDIV':
                    mod.iterations = max(1, int((1.0 - self.decimate_ratio) * 10))
                elif self.decimate_type == 'PLANAR':
                    mod.angle_limit = math.radians((1.0 - self.decimate_ratio) * 45.0)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as e:
                self.report({'WARNING'}, f"Decimate failed: {e}")
            finally:
                if prev_mode == 'EDIT':
                    try:
                        bpy.ops.object.mode_set(mode='EDIT')
                    except Exception:
                        pass
        else:
            if prev_mode != 'EDIT':
                try:
                    bpy.ops.object.mode_set(mode=prev_mode)
                except Exception:
                    pass

        try:
            if prev_mode == 'EDIT':
                bpy.ops.mesh.normals_make_consistent(inside=False)
            else:
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.object.mode_set(mode=prev_mode)
        except Exception:
            pass

        self.report({'INFO'}, "DeltaQuadPlus pipeline finished")
        return {'FINISHED'}


class MESH_OT_grid_cut(bpy.types.Operator):
    """Cut the active mesh into a grid (safe mode with clamping)."""
    bl_idname = "mesh.grid_cut"
    bl_label = "Grid Cut"
    bl_options = {'REGISTER', 'UNDO'}

    divisions_x: IntProperty(name="Divisions X", default=4, min=1, max=4096, options={'HIDDEN'})
    divisions_y: IntProperty(name="Divisions Y", default=4, min=1, max=4096, options={'HIDDEN'})
    divisions_z: IntProperty(name="Divisions Z", default=4, min=1, max=4096, options={'HIDDEN'})

    use_cell_size: BoolProperty(name="Use Cell Size", default=False, options={'HIDDEN'})
    cell_size: FloatProperty(name="Cell Size", default=0.2, min=1e-9, soft_max=10.0, options={'HIDDEN'})

    # Simpler alternative: Grid Density (1=coarse, 100=fine)
    use_density: BoolProperty(
        name="Use Density",
        default=True,
        description="Use a simple density slider instead of manual division counts"
    )
    grid_density: IntProperty(
        name="Grid Density",
        default=10, min=1, max=100,
        description=(
            "Grid density: 1 = very coarse (few cuts), "
            "100 = very fine (many cuts). "
            "Automatically adapts to the object size"
        )
    )

    merge_threshold: FloatProperty(name="Merge Distance", default=1e-6, min=0.0, max=1.0)
    separate_parts: BoolProperty(name="Separate Loose Parts", default=False)

    def cut_positions(self, divisions, min_val, max_val):
        cuts = max(0, divisions - 1)
        if cuts == 0:
            return []
        step = (max_val - min_val) / float(divisions)
        return [min_val + step * i for i in range(1, divisions)]

    def execute(self, context):
        scene = context.scene
        max_div = max(1, getattr(scene, "gridcut_max_divisions", 256))

        obj = context.view_layer.objects.active
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Sélectionne un mesh et rends-le actif.")
            return {'CANCELLED'}

        prev_mode = obj.mode
        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass

        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.verts.ensure_lookup_table()

        min_v, max_v = object_bounds_local(obj)
        bbox_size = max_v - min_v
        bbox_max = max(bbox_size.x, bbox_size.y, bbox_size.z)

        if bbox_max <= 0.0:
            self.report({'ERROR'}, "Bounding box invalide (taille nulle).")
            try:
                bpy.ops.object.mode_set(mode=prev_mode)
            except Exception:
                pass
            return {'CANCELLED'}

        cell_limit = getattr(scene, "gridcut_cells_limit", 2000000)

        def divisions_from_cell_for_axis(axis_len, c):
            if axis_len <= 0.0:
                return 1
            n = int(math.ceil(axis_len / c))
            n = max(1, min(n, max_div))
            return n

        # --- Density mode: convert density (1-100) to cell size automatically ---
        if self.use_density:
            # density=1 → cell = bbox_max (1 cut), density=100 → cell = bbox_max/100
            density = max(1, min(100, self.grid_density))
            # Map density non-linearly so low values = coarse, high = fine
            # cell_size = bbox_max / density (proportional to object)
            auto_cell = bbox_max / float(density)
            auto_cell = max(auto_cell, bbox_max / float(max_div))
            eff_div_x = divisions_from_cell_for_axis(bbox_size.x, auto_cell)
            eff_div_y = divisions_from_cell_for_axis(bbox_size.y, auto_cell)
            eff_div_z = divisions_from_cell_for_axis(bbox_size.z, auto_cell)
            estimated_cells = eff_div_x * eff_div_y * eff_div_z
        elif self.use_cell_size:
            min_allowed_cell = bbox_max / float(max_div)
            cell = max(self.cell_size, min_allowed_cell)
            if cell != self.cell_size:
                self.report({'WARNING'}, f"Cell Size trop petite -> ajustée à {cell:.6g} pour limiter divisions à {max_div}.")

            eff_div_x = divisions_from_cell_for_axis(bbox_size.x, cell)
            eff_div_y = divisions_from_cell_for_axis(bbox_size.y, cell)
            eff_div_z = divisions_from_cell_for_axis(bbox_size.z, cell)

            estimated_cells = eff_div_x * eff_div_y * eff_div_z

            if estimated_cells > cell_limit:
                for _ in range(50):
                    scale = (estimated_cells / float(cell_limit)) ** (1.0 / 3.0)
                    if scale <= 1.0:
                        scale = 1.1
                    cell *= scale
                    eff_div_x = divisions_from_cell_for_axis(bbox_size.x, cell)
                    eff_div_y = divisions_from_cell_for_axis(bbox_size.y, cell)
                    eff_div_z = divisions_from_cell_for_axis(bbox_size.z, cell)
                    estimated_cells = eff_div_x * eff_div_y * eff_div_z
                    if estimated_cells <= cell_limit:
                        break

                if estimated_cells > cell_limit:
                    eff_div_x = eff_div_y = eff_div_z = 1
                    estimated_cells = 1
                    self.report({'WARNING'}, "Impossible de respecter pleinement la limite, divisions forcées à 1.")
                else:
                    self.report({'WARNING'}, f"Cell Size ajustée à {cell:.6g} pour respecter limite de {cell_limit} cellules (est. {estimated_cells}).")
        else:
            eff_div_x = min(max(1, self.divisions_x), max_div)
            eff_div_y = min(max(1, self.divisions_y), max_div)
            eff_div_z = min(max(1, self.divisions_z), max_div)
            if self.divisions_x != eff_div_x or self.divisions_y != eff_div_y or self.divisions_z != eff_div_z:
                self.report({'WARNING'}, f"Divisions clampées à max {max_div} pour éviter surcharge.")

            estimated_cells = eff_div_x * eff_div_y * eff_div_z

            if estimated_cells > cell_limit:
                orig_x, orig_y, orig_z = eff_div_x, eff_div_y, eff_div_z
                for _ in range(50):
                    scale = (estimated_cells / float(cell_limit)) ** (1.0 / 3.0)
                    if scale <= 1.0:
                        scale = 1.1
                    eff_div_x = max(1, int(eff_div_x / scale))
                    eff_div_y = max(1, int(eff_div_y / scale))
                    eff_div_z = max(1, int(eff_div_z / scale))
                    eff_div_x = min(eff_div_x, max_div)
                    eff_div_y = min(eff_div_y, max_div)
                    eff_div_z = min(eff_div_z, max_div)
                    estimated_cells = eff_div_x * eff_div_y * eff_div_z
                    if estimated_cells <= cell_limit:
                        break

                if estimated_cells > cell_limit:
                    eff_div_x = eff_div_y = eff_div_z = 1
                    estimated_cells = 1
                    self.report({'WARNING'}, "Impossible de respecter la limite, divisions forcées à 1.")
                else:
                    self.report({'WARNING'}, f"Divisions réduites à X={eff_div_x} Y={eff_div_y} Z={eff_div_z} pour respecter limite de {cell_limit} cellules (est. {estimated_cells}).")

        estimated_cells = eff_div_x * eff_div_y * eff_div_z
        if estimated_cells > cell_limit:
            self.report({'ERROR'}, f"Opération trop lourde: {estimated_cells} cellules estimées > limite {cell_limit}. Réduis la Cell Size ou Divisions.")
            try:
                bpy.ops.object.mode_set(mode=prev_mode)
            except Exception:
                pass
            return {'CANCELLED'}

        axes = [
            (Vector((1.0, 0.0, 0.0)), eff_div_x, 0),
            (Vector((0.0, 1.0, 0.0)), eff_div_y, 1),
            (Vector((0.0, 0.0, 1.0)), eff_div_z, 2),
        ]

        def current_geom():
            return list(bm.verts) + list(bm.edges) + list(bm.faces)

        for axis_vec, divisions, idx in axes:
            positions = self.cut_positions(divisions, min_v[idx], max_v[idx])
            for pos in positions:
                plane_co_local = Vector((0.0, 0.0, 0.0))
                plane_co_local[idx] = pos
                plane_co = plane_co_local
                plane_no = axis_vec.normalized()

                try:
                    bmesh.ops.bisect_plane(
                        bm,
                        geom=current_geom(),
                        plane_co=plane_co,
                        plane_no=plane_no,
                        use_snap_center=False,
                        clear_inner=False,
                        clear_outer=False
                    )
                except Exception as e:
                    self.report({'WARNING'}, f"bisect_plane failed at pos {pos}: {e}")
                bm.verts.ensure_lookup_table()

        try:
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=self.merge_threshold)
        except Exception:
            pass

        try:
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        except Exception:
            for f in bm.faces:
                f.normal_update()

        # Tag new edges created by bisect with a custom attribute for later removal
        try:
            tag_layer = bm.edges.layers.float.get('gridcut_tag')
            if tag_layer is None:
                tag_layer = bm.edges.layers.float.new('gridcut_tag')
            for e in bm.edges:
                if e[tag_layer] != 1.0:
                    # Edges that are interior (not boundary) and not already tagged
                    # were created by the grid cut
                    if not e.is_boundary:
                        e[tag_layer] = 1.0
        except Exception:
            pass

        bmesh.update_edit_mesh(me, loop_triangles=False)

        if self.separate_parts:
            try:
                bpy.ops.mesh.separate(type='LOOSE')
            except Exception:
                self.report({'WARNING'}, "Séparation échouée ou aucune partie créée.")

        try:
            bpy.ops.object.mode_set(mode=prev_mode)
        except Exception:
            pass

        self.report({'INFO'}, f"Grid Cut: X={eff_div_x} Y={eff_div_y} Z={eff_div_z} (est. cellules={estimated_cells})")
        try:
            bpy.context.scene.gridcut_active = True
        except Exception:
            pass
        # Auto-enable wireframe overlay so cuts are visible
        try:
            _set_view3d_overlay_wireframe(True, opacity=1.0)
        except Exception:
            pass
        return {'FINISHED'}


class MESH_OT_remove_grid_cut(bpy.types.Operator):
    """Remove all grid cut edges by dissolving interior edges tagged by Grid Cut."""
    bl_idname  = 'mesh.remove_grid_cut'
    bl_label   = 'Remove Grid Cut'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        ensure_object_mode()

        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass

        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.edges.ensure_lookup_table()

        tag_layer = bm.edges.layers.float.get('gridcut_tag')

        if tag_layer:
            # Dissolve tagged edges
            edges_to_dissolve = [e for e in bm.edges if e[tag_layer] == 1.0]
            if edges_to_dissolve:
                try:
                    bmesh.ops.dissolve_edges(bm, edges=edges_to_dissolve,
                                             use_verts=True, use_face_split=False)
                    bmesh.update_edit_mesh(me)
                    # Remove the custom layer
                    bm.edges.layers.float.remove(tag_layer)
                    bmesh.update_edit_mesh(me)
                    n = len(edges_to_dissolve)
                    try:
                        bpy.ops.object.mode_set(mode='OBJECT')
                        bpy.context.scene.gridcut_active = False
                    except Exception:
                        pass
                    self.report({'INFO'}, f"Grid cut removed — {n} edges dissolved.")
                    return {'FINISHED'}
                except Exception as e:
                    self.report({'WARNING'}, f"Dissolve failed: {e}")
            else:
                self.report({'WARNING'}, "No tagged grid cut edges found.")
        else:
            # Fallback: dissolve all interior non-boundary edges (less precise)
            bpy.ops.mesh.select_all(action='DESELECT')
            for e in bm.edges:
                if not e.is_boundary and len(e.link_faces) == 2:
                    e.select = True
            bmesh.update_edit_mesh(me)
            try:
                bpy.ops.mesh.dissolve_edges(use_verts=True)
            except Exception as e:
                self.report({'WARNING'}, f"Fallback dissolve failed: {e}")

        try:
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.scene.gridcut_active = False
        except Exception:
            pass
        self.report({'INFO'}, "Grid cut edges removed.")
        return {'FINISHED'}


# ------------------ Tris -> Quads operator (added) ------------------
class OBJECT_OT_call_tris_convert_to_quads(bpy.types.Operator):
    """Call mesh.tris_convert_to_quads with a robust 3D View override (falls back if needed)."""
    bl_idname = "object.call_tris_convert_to_quads"
    bl_label = "Tris → Quads"
    bl_options = {'REGISTER', 'UNDO'}

    uvs: BoolProperty(
        name="Respect UVs",
        default=False,
        description="Do not convert triangles across UV islands"
    )
    seam: BoolProperty(
        name="Respect Seams",
        default=False,
        description="Do not convert triangles across seams"
    )
    face_threshold: FloatProperty(
        name="Face angle (rad)",
        default=0.523599,
        min=0.0, max=3.14159,
        description="Max angle between face normals to allow conversion (radians)"
    )
    shape_threshold: FloatProperty(
        name="Shape threshold (rad)",
        default=0.523599,
        min=0.0, max=3.14159,
        description="Shape similarity threshold for triangle pairs (radians)"
    )

    @classmethod
    def poll(cls, context):
        obj = context.view_layer.objects.active
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        active = context.view_layer.objects.active
        if active is None or active.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh.")
            return {'CANCELLED'}

        started_in_edit = (context.mode == 'EDIT_MESH')
        try:
            if not started_in_edit:
                bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            self.report({'WARNING'}, "Could not switch to Edit Mode automatically — ensure target is in Edit Mode.")

        base_override = None
        wm = bpy.context.window_manager
        for window in wm.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    space = area.spaces.active
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            base_override = {
                                'window': window,
                                'screen': screen,
                                'area': area,
                                'region': region,
                                'scene': bpy.context.scene,
                                'space_data': space,
                            }
                            break
                    if base_override:
                        break
            if base_override:
                break

        override = None
        if base_override:
            override = base_override.copy()
            override['edit_object'] = active
            override['active_object'] = active

        kwargs = {
            "uvs": self.uvs,
            "seam": self.seam,
            "face_threshold": self.face_threshold,
            "shape_threshold": self.shape_threshold,
        }

        err = None
        if override is not None:
            try:
                bpy.ops.mesh.tris_convert_to_quads(override, **kwargs)
                return {'FINISHED'}
            except Exception as e:
                err = str(e)

        try:
            bpy.ops.mesh.tris_convert_to_quads(**kwargs)
            return {'FINISHED'}
        except Exception as e2:
            try:
                if not started_in_edit:
                    bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
            msg = f"tris_convert_to_quads failed: {err or ''} {e2}"
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}


# ------------------ Geometry Node groups and apply operators (RESTORED) ------------------

def leaf_bend_gn_2_node_group(node_tree_names: dict[typing.Callable, str]):
    """Create LEAF_BEND_GN: simple robust bend + roll + edge waviness."""
    name = "LEAF_BEND_GN"
    existing = bpy.data.node_groups.get(name)
    if existing:
        bpy.data.node_groups.remove(existing)

    ng = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)
    ng.is_modifier = True
    ng.show_modifier_manage_panel = True

    iface = ng.interface

    def _geo(label, in_out):
        s = iface.new_socket(name=label, in_out=in_out, socket_type='NodeSocketGeometry')
        s.attribute_domain = 'POINT'; s.default_input = 'VALUE'; s.structure_type = 'AUTO'
        return s

    def _flt(label, default=0.0, mn=-100.0, mx=100.0):
        s = iface.new_socket(name=label, in_out='INPUT', socket_type='NodeSocketFloat')
        s.attribute_domain = 'POINT'; s.default_input = 'VALUE'; s.structure_type = 'AUTO'
        try: s.default_value = default
        except Exception: pass
        try: s.min_value = mn
        except Exception: pass
        try: s.max_value = mx
        except Exception: pass
        return s

    def _int(label, default=0, mn=0, mx=10):
        s = iface.new_socket(name=label, in_out='INPUT', socket_type='NodeSocketInt')
        s.attribute_domain = 'POINT'; s.default_input = 'VALUE'; s.structure_type = 'AUTO'
        try: s.default_value = default
        except Exception: pass
        try: s.min_value = mn
        except Exception: pass
        try: s.max_value = mx
        except Exception: pass
        return s

    def _sep(label):
        s = iface.new_socket(name=label, in_out='INPUT', socket_type='NodeSocketString')
        s.attribute_domain = 'POINT'; s.default_input = 'VALUE'; s.structure_type = 'AUTO'
        try: s.default_value = label
        except Exception: pass
        return s

    _geo("Geometry", 'OUTPUT')
    _geo("Geometry", 'INPUT')
    _sep("SUBDIVISIONS"); _int("Subdivisions", default=0, mn=0, mx=6)
    _sep("BEND");         _flt("Bend X", 0.0, -10.0, 10.0); _flt("Bend Y", 0.0, -10.0, 10.0)
    _sep("ROLL");         _flt("Roll",   0.0,   0.0,  6.28)
    _sep("EDGE WAVINESS")
    _flt("Wave Strength", 0.0, 0.0, 0.5)
    _flt("Wave Scale",    8.0, 0.1, 50.0)
    _flt("Wave Falloff",  2.0, 0.1, 10.0)

    nd = ng.nodes
    lk = ng.links

    def N(bl_id, x, y):
        node = nd.new(bl_id); node.location = (x, y); return node

    def inp(name):
        gi = nd.new("NodeGroupInput"); gi.location = (-800, 0)
        return gi

    # single Group Input
    gi  = N("NodeGroupInput",  -800,  0)
    go  = N("NodeGroupOutput", 3400,  0); go.is_active_output = True

    def sock(name):
        for o in gi.outputs:
            if o.name == name: return o
        return None

    # 1 — Subdivide
    sub = N("GeometryNodeSubdivideMesh", -560, 0)
    lk.new(gi.outputs[0],        sub.inputs[0])
    lk.new(sock("Subdivisions"), sub.inputs[1])

    # 2 — Bend X  (rotate each vertex around Y by pos.x * BendX)
    p1  = N("GeometryNodeInputPosition", -560, -250)
    sx1 = N("ShaderNodeSeparateXYZ",     -360, -250)
    m1  = N("ShaderNodeMath",            -160, -250); m1.operation = 'MULTIPLY'
    vr1 = N("ShaderNodeVectorRotate",     40,  -150)
    vr1.rotation_type = 'AXIS_ANGLE'
    try: vr1.inputs[1].default_value = (0.0, 0.0, 0.0)
    except Exception: pass
    try: vr1.inputs[2].default_value = (0.0, 1.0, 0.0)   # Y axis
    except Exception: pass
    sp1 = N("GeometryNodeSetPosition",  240, 0)
    try: sp1.inputs[1].default_value = True
    except Exception: pass

    lk.new(sub.outputs[0],   sp1.inputs[0])
    lk.new(p1.outputs[0],    sx1.inputs[0])
    lk.new(sx1.outputs[0],   m1.inputs[0])
    lk.new(sock("Bend X"),   m1.inputs[1])
    lk.new(p1.outputs[0],    vr1.inputs[0])
    lk.new(m1.outputs[0],    vr1.inputs[3])
    lk.new(vr1.outputs[0],   sp1.inputs[2])

    # 3 — Bend Y  (rotate around X by pos.y * BendY)
    p2  = N("GeometryNodeInputPosition", -560, -550)
    sx2 = N("ShaderNodeSeparateXYZ",     -360, -550)
    m2  = N("ShaderNodeMath",            -160, -550); m2.operation = 'MULTIPLY'
    vr2 = N("ShaderNodeVectorRotate",     640, -350)
    vr2.rotation_type = 'AXIS_ANGLE'
    try: vr2.inputs[1].default_value = (0.0, 0.0, 0.0)
    except Exception: pass
    try: vr2.inputs[2].default_value = (1.0, 0.0, 0.0)   # X axis
    except Exception: pass
    sp2 = N("GeometryNodeSetPosition",  840, 0)
    try: sp2.inputs[1].default_value = True
    except Exception: pass

    lk.new(sp1.outputs[0],   sp2.inputs[0])
    lk.new(p2.outputs[0],    sx2.inputs[0])
    lk.new(sx2.outputs[1],   m2.inputs[0])
    lk.new(sock("Bend Y"),   m2.inputs[1])
    lk.new(p2.outputs[0],    vr2.inputs[0])
    lk.new(m2.outputs[0],    vr2.inputs[3])
    lk.new(vr2.outputs[0],   sp2.inputs[2])

    # 4 — Roll  (rotate around X by pos.x * Roll)
    p3  = N("GeometryNodeInputPosition", -560, -850)
    sx3 = N("ShaderNodeSeparateXYZ",     -360, -850)
    m3  = N("ShaderNodeMath",            -160, -850); m3.operation = 'MULTIPLY'
    vr3 = N("ShaderNodeVectorRotate",    1240, -550)
    vr3.rotation_type = 'AXIS_ANGLE'
    try: vr3.inputs[1].default_value = (0.0, 0.0, 0.0)
    except Exception: pass
    try: vr3.inputs[2].default_value = (1.0, 0.0, 0.0)
    except Exception: pass
    sp3 = N("GeometryNodeSetPosition",  1440, 0)
    try: sp3.inputs[1].default_value = True
    except Exception: pass

    lk.new(sp2.outputs[0],  sp3.inputs[0])
    lk.new(p3.outputs[0],   sx3.inputs[0])
    lk.new(sx3.outputs[0],  m3.inputs[0])
    lk.new(sock("Roll"),    m3.inputs[1])
    lk.new(p3.outputs[0],   vr3.inputs[0])
    lk.new(m3.outputs[0],   vr3.inputs[3])
    lk.new(vr3.outputs[0],  sp3.inputs[2])

    # 5 — Edge Waviness  (noise * radius^falloff * strength → Z offset)
    p4  = N("GeometryNodeInputPosition", 1440, -400)
    sx4 = N("ShaderNodeSeparateXYZ",     1640, -400)
    # x²+y² → sqrt → radius
    mx4 = N("ShaderNodeMath", 1840, -350); mx4.operation = 'MULTIPLY'
    my4 = N("ShaderNodeMath", 1840, -500); my4.operation = 'MULTIPLY'
    add = N("ShaderNodeMath", 2040, -420); add.operation = 'ADD'
    sqr = N("ShaderNodeMath", 2240, -420); sqr.operation = 'SQRT'
    pw  = N("ShaderNodeMath", 2440, -420); pw.operation  = 'POWER'

    # noise: scale pos by Wave Scale using CombineXYZ trick
    csc = N("ShaderNodeCombineXYZ",  1440, -700)
    vmul= N("ShaderNodeVectorMath",  1640, -700); vmul.operation = 'MULTIPLY'
    nz  = N("ShaderNodeTexNoise",    1840, -700); nz.noise_dimensions = '3D'
    try: nz.inputs['Detail'].default_value = 6.0
    except (KeyError, AttributeError): pass
    try: nz.inputs['Roughness'].default_value = 0.6
    except (KeyError, AttributeError): pass

    # remap 0..1 → -1..1
    sh  = N("ShaderNodeMath", 2040, -700); sh.operation = 'SUBTRACT'
    try: sh.inputs[1].default_value = 0.5
    except Exception: pass
    mt  = N("ShaderNodeMath", 2240, -700); mt.operation = 'MULTIPLY'
    try: mt.inputs[1].default_value = 2.0
    except Exception: pass

    # noise * radius^falloff * strength → Z
    mw  = N("ShaderNodeMath", 2640, -560); mw.operation = 'MULTIPLY'
    ms  = N("ShaderNodeMath", 2840, -560); ms.operation = 'MULTIPLY'
    cb  = N("ShaderNodeCombineXYZ", 3040, -560)
    try: cb.inputs[0].default_value = 0.0
    except Exception: pass
    try: cb.inputs[1].default_value = 0.0
    except Exception: pass
    sp4 = N("GeometryNodeSetPosition", 3240, 0)
    try: sp4.inputs[1].default_value = True
    except Exception: pass

    lk.new(p4.outputs[0],           sx4.inputs[0])
    lk.new(sx4.outputs[0],          mx4.inputs[0]); lk.new(sx4.outputs[0], mx4.inputs[1])
    lk.new(sx4.outputs[1],          my4.inputs[0]); lk.new(sx4.outputs[1], my4.inputs[1])
    lk.new(mx4.outputs[0],          add.inputs[0]); lk.new(my4.outputs[0], add.inputs[1])
    lk.new(add.outputs[0],          sqr.inputs[0])
    lk.new(sqr.outputs[0],          pw.inputs[0])
    lk.new(sock("Wave Falloff"),    pw.inputs[1])

    lk.new(sock("Wave Scale"),      csc.inputs[0])
    lk.new(sock("Wave Scale"),      csc.inputs[1])
    lk.new(sock("Wave Scale"),      csc.inputs[2])
    lk.new(p4.outputs[0],           vmul.inputs[0])
    lk.new(csc.outputs[0],          vmul.inputs[1])
    lk.new(vmul.outputs[0],         nz.inputs[0])
    lk.new(nz.outputs[0],           sh.inputs[0])
    lk.new(sh.outputs[0],           mt.inputs[0])

    lk.new(mt.outputs[0],           mw.inputs[0])
    lk.new(pw.outputs[0],           mw.inputs[1])
    lk.new(mw.outputs[0],           ms.inputs[0])
    lk.new(sock("Wave Strength"),   ms.inputs[1])
    lk.new(ms.outputs[0],           cb.inputs[2])
    lk.new(cb.outputs[0],           sp4.inputs[3])

    lk.new(sp3.outputs[0],          sp4.inputs[0])
    lk.new(sp4.outputs[0],          go.inputs[0])

    return ng


class OBJECT_OT_apply_leaf_geometry_node(bpy.types.Operator):
    """Créer le Node Group LEAF_BEND_GN (si manquant) et l'ajouter/appliquer sur l'objet actif."""
    bl_idname = "object.apply_leaf_geometry_node"
    bl_label = "Create & Apply LEAF_BEND_GN"
    bl_options = {'REGISTER', 'UNDO'}

    node_group_name: StringProperty(
        name="Node Group Name",
        default="LEAF_BEND_GN",
    )
    add_modifier: BoolProperty(
        name="Add Geometry Nodes modifier if missing",
        default=True,
    )
    apply_modifier: BoolProperty(
        name="Apply modifier after adding",
        default=False,
    )

    def ensure_node_group(self):
        ng = bpy.data.node_groups.get(self.node_group_name)
        if ng is None:
            try:
                ng = leaf_bend_gn_2_node_group({})
            except Exception:
                ng = None
        return ng

    def execute(self, context):
        ensure_object_mode()
        obj = context.view_layer.objects.active
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "L'objet actif doit être un mesh.")
            return {'CANCELLED'}

        ng = self.ensure_node_group()
        if ng is None:
            self.report({'ERROR'}, f"Impossible de créer ou trouver le Node Group '{self.node_group_name}'.")
            return {'CANCELLED'}

        mod = None
        for m in obj.modifiers:
            try:
                if getattr(m, "type", "") == 'NODES' and getattr(m, "node_group", None) is not None:
                    if m.node_group.name == ng.name:
                        mod = m
                        break
            except Exception:
                continue

        if mod is None and self.add_modifier:
            try:
                mod = obj.modifiers.new(name=self.node_group_name, type='NODES')
                mod.node_group = ng
            except Exception as e:
                self.report({'ERROR'}, f"Impossible de créer/assigner le modificateur GN : {e}")
                return {'CANCELLED'}
        elif mod is None:
            self.report({'ERROR'}, "Aucun modificateur Geometry Nodes trouvé pour ce node group.")
            return {'CANCELLED'}

        if self.apply_modifier:
            for o in context.view_layer.objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as e:
                self.report({'ERROR'}, f"Échec de l'application du modificateur : {e}")
                return {'CANCELLED'}

        self.report({'INFO'}, f"Node Group '{self.node_group_name}' appliqué sur '{obj.name}'.")
        return {'FINISHED'}


def subdvz_gn_2_node_group(node_tree_names: dict[typing.Callable, str]):
    name = "SUBDVZ GN"
    existing = bpy.data.node_groups.get(name)
    if existing:
        return existing

    subdvz_gn_2 = bpy.data.node_groups.new(type='GeometryNodeTree', name=name)

    subdvz_gn_2.color_tag = 'NONE'
    subdvz_gn_2.description = ""
    subdvz_gn_2.default_group_node_width = 140
    subdvz_gn_2.is_modifier = True
    subdvz_gn_2.show_modifier_manage_panel = True

    geometry_socket = subdvz_gn_2.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    geometry_socket.attribute_domain = 'POINT'
    geometry_socket.default_input = 'VALUE'
    geometry_socket.structure_type = 'AUTO'

    geometry_socket_1 = subdvz_gn_2.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    geometry_socket_1.attribute_domain = 'POINT'
    geometry_socket_1.default_input = 'VALUE'
    geometry_socket_1.structure_type = 'AUTO'

    _socket = subdvz_gn_2.interface.new_socket(name="", in_out='INPUT', socket_type='NodeSocketString')
    _socket.default_value = ""
    _socket.subtype = 'NONE'
    _socket.attribute_domain = 'POINT'
    _socket.description = "SUBDIVISIONS"
    _socket.default_input = 'VALUE'
    _socket.structure_type = 'AUTO'

    level_socket = subdvz_gn_2.interface.new_socket(name="Level", in_out='INPUT', socket_type='NodeSocketInt')
    level_socket.default_value = 2
    level_socket.min_value = 0
    level_socket.max_value = 6
    level_socket.subtype = 'NONE'
    level_socket.attribute_domain = 'POINT'
    level_socket.force_non_field = True
    level_socket.default_input = 'VALUE'
    level_socket.structure_type = 'SINGLE'

    edge_crease_socket = subdvz_gn_2.interface.new_socket(name="Edge Crease", in_out='INPUT', socket_type='NodeSocketFloat')
    edge_crease_socket.default_value = 0.0
    edge_crease_socket.min_value = 0.0
    edge_crease_socket.max_value = 1.0
    edge_crease_socket.subtype = 'FACTOR'
    edge_crease_socket.attribute_domain = 'POINT'
    edge_crease_socket.force_non_field = True
    edge_crease_socket.default_input = 'VALUE'
    edge_crease_socket.structure_type = 'SINGLE'

    vertex_crease_socket = subdvz_gn_2.interface.new_socket(name="Vertex Crease", in_out='INPUT', socket_type='NodeSocketFloat')
    vertex_crease_socket.default_value = 0.0
    vertex_crease_socket.min_value = 0.0
    vertex_crease_socket.max_value = 1.0
    vertex_crease_socket.subtype = 'FACTOR'
    vertex_crease_socket.attribute_domain = 'POINT'
    vertex_crease_socket.force_non_field = True
    vertex_crease_socket.default_input = 'VALUE'
    vertex_crease_socket.structure_type = 'SINGLE'

    _socket_1 = subdvz_gn_2.interface.new_socket(name="", in_out='INPUT', socket_type='NodeSocketString')
    _socket_1.default_value = ""
    _socket_1.subtype = 'NONE'
    _socket_1.attribute_domain = 'POINT'
    _socket_1.description = "SMOOTH"
    _socket_1.default_input = 'VALUE'
    _socket_1.structure_type = 'AUTO'

    limit_surface_socket = subdvz_gn_2.interface.new_socket(name="Limit Surface", in_out='INPUT', socket_type='NodeSocketBool')
    limit_surface_socket.default_value = True
    limit_surface_socket.attribute_domain = 'POINT'
    limit_surface_socket.force_non_field = True
    limit_surface_socket.default_input = 'VALUE'
    limit_surface_socket.structure_type = 'SINGLE'

    shade_smooth_socket = subdvz_gn_2.interface.new_socket(name="Shade Smooth", in_out='INPUT', socket_type='NodeSocketBool')
    shade_smooth_socket.default_value = False
    shade_smooth_socket.attribute_domain = 'POINT'
    shade_smooth_socket.force_non_field = True
    shade_smooth_socket.default_input = 'VALUE'
    shade_smooth_socket.structure_type = 'SINGLE'

    group_input = subdvz_gn_2.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"
    group_output = subdvz_gn_2.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    subdivision_surface = subdvz_gn_2.nodes.new("GeometryNodeSubdivisionSurface")
    subdivision_surface.name = "Subdivision Surface"
    try:
        subdivision_surface.inputs[5].default_value = 'Keep Boundaries'
        subdivision_surface.inputs[6].default_value = 'All'
    except Exception:
        pass

    set_shade_smooth = subdvz_gn_2.nodes.new("GeometryNodeSetShadeSmooth")
    set_shade_smooth.name = "Set Shade Smooth"
    set_shade_smooth.domain = 'FACE'
    try:
        set_shade_smooth.inputs[1].default_value = True
    except Exception:
        pass

    frame = subdvz_gn_2.nodes.new("NodeFrame")
    frame.label = "SUBDVZ GN"
    frame.name = "Frame"
    frame.use_custom_color = True
    frame.color = (0.0, 0.0, 0.0)
    frame.label_size = 20
    frame.shrink = True

    clamp = subdvz_gn_2.nodes.new("ShaderNodeClamp")
    clamp.name = "Clamp"
    clamp.clamp_type = 'MINMAX'
    try:
        clamp.inputs[1].default_value = 0.0
        clamp.inputs[2].default_value = 11.0
    except Exception:
        pass

    reroute = subdvz_gn_2.nodes.new("NodeReroute")
    reroute.name = "Reroute"
    reroute.socket_idname = "NodeSocketGeometry"
    reroute_001 = subdvz_gn_2.nodes.new("NodeReroute")
    reroute_001.name = "Reroute.001"
    reroute_001.socket_idname = "NodeSocketFloatFactor"
    reroute_002 = subdvz_gn_2.nodes.new("NodeReroute")
    reroute_002.name = "Reroute.002"
    reroute_002.socket_idname = "NodeSocketFloatFactor"
    reroute_003 = subdvz_gn_2.nodes.new("NodeReroute")
    reroute_003.name = "Reroute.003"
    reroute_003.socket_idname = "NodeSocketBool"
    reroute_004 = subdvz_gn_2.nodes.new("NodeReroute")
    reroute_004.name = "Reroute.004"
    reroute_004.socket_idname = "NodeSocketBool"

    try:
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Set Shade Smooth"].outputs[0], subdvz_gn_2.nodes["Group Output"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Subdivision Surface"].outputs[0], subdvz_gn_2.nodes["Set Shade Smooth"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Group Input"].outputs[2], subdvz_gn_2.nodes["Clamp"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Clamp"].outputs[0], subdvz_gn_2.nodes["Subdivision Surface"].inputs[1])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Group Input"].outputs[0], subdvz_gn_2.nodes["Reroute"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Reroute"].outputs[0], subdvz_gn_2.nodes["Subdivision Surface"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Group Input"].outputs[3], subdvz_gn_2.nodes["Reroute.001"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Reroute.001"].outputs[0], subdvz_gn_2.nodes["Subdivision Surface"].inputs[2])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Group Input"].outputs[4], subdvz_gn_2.nodes["Reroute.002"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Reroute.002"].outputs[0], subdvz_gn_2.nodes["Subdivision Surface"].inputs[3])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Group Input"].outputs[6], subdvz_gn_2.nodes["Reroute.003"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Reroute.003"].outputs[0], subdvz_gn_2.nodes["Subdivision Surface"].inputs[4])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Group Input"].outputs[7], subdvz_gn_2.nodes["Reroute.004"].inputs[0])
        subdvz_gn_2.links.new(subdvz_gn_2.nodes["Reroute.004"].outputs[0], subdvz_gn_2.nodes["Set Shade Smooth"].inputs[2])
    except Exception:
        pass

    return subdvz_gn_2


class OBJECT_OT_apply_subdvz_geometry_node(bpy.types.Operator):
    """Créer le Node Group 'SUBDVZ GN' (si manquant) et l'ajouter/appliquer sur l'objet actif."""
    bl_idname = "object.apply_subdvz_geometry_node"
    bl_label = "Create & Apply SUBDVZ_GN"
    bl_options = {'REGISTER', 'UNDO'}

    node_group_name: StringProperty(
        name="Node Group Name",
        default="SUBDVZ GN",
    )
    add_modifier: BoolProperty(
        name="Add Geometry Nodes modifier if missing",
        default=True,
    )
    apply_modifier: BoolProperty(
        name="Apply modifier after adding",
        default=False,
    )

    def ensure_node_group(self):
        ng = bpy.data.node_groups.get(self.node_group_name)
        if ng is None:
            try:
                ng = subdvz_gn_2_node_group({})
            except Exception:
                ng = None
        return ng

    def execute(self, context):
        ensure_object_mode()
        obj = context.view_layer.objects.active
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "L'objet actif doit être un mesh.")
            return {'CANCELLED'}

        ng = self.ensure_node_group()
        if ng is None:
            self.report({'ERROR'}, f"Impossible de créer ou trouver le Node Group '{self.node_group_name}'.")
            return {'CANCELLED'}

        mod = None
        for m in obj.modifiers:
            try:
                if getattr(m, "type", "") == 'NODES' and getattr(m, "node_group", None) is not None:
                    if m.node_group.name == ng.name:
                        mod = m
                        break
            except Exception:
                continue

        if mod is None and self.add_modifier:
            try:
                mod = obj.modifiers.new(name=self.node_group_name, type='NODES')
                mod.node_group = ng
            except Exception as e:
                self.report({'ERROR'}, f"Impossible de créer/assigner le modificateur GN : {e}")
                return {'CANCELLED'}
        elif mod is None:
            self.report({'ERROR'}, "Aucun modificateur Geometry Nodes trouvé pour ce node group.")
            return {'CANCELLED'}

        if self.apply_modifier:
            for o in context.view_layer.objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as e:
                self.report({'ERROR'}, f"Échec de l'application du modificateur : {e}")
                return {'CANCELLED'}

        self.report({'INFO'}, f"Node Group '{self.node_group_name}' appliqué sur '{obj.name}'.")
        return {'FINISHED'}


# ------------------ Wireframe overlay toggle (uses overlay.show_wireframes) ------------------

def _set_view3d_overlay_wireframe(state: bool, opacity: float = None):
    """Set overlay.show_wireframes for all VIEW_3D areas and optionally set wireframe opacity if available."""
    wm = bpy.context.window_manager
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                try:
                    if hasattr(space, "overlay"):
                        space.overlay.show_wireframes = bool(state)
                        if opacity is not None and hasattr(space.overlay, "wireframe_opacity"):
                            try:
                                val = max(0.0, min(1.0, float(opacity)))
                                space.overlay.wireframe_opacity = val
                            except Exception:
                                pass
                except Exception:
                    pass


def _any_view3d_wireframe_active():
    """Return True if any VIEW_3D overlay.show_wireframes is True (used to draw depressed state)."""
    wm = bpy.context.window_manager
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                try:
                    if hasattr(space, "overlay") and getattr(space.overlay, "show_wireframes", False):
                        return True
                except Exception:
                    pass
    return False


class VIEW3D_OT_toggle_wireframe(bpy.types.Operator):
    """Toggle the Viewport Overlay 'Wireframe' checkbox in all 3D Views.
    Operator is REGISTER so the Adjust Last Operation shows the opacity slider.
    """
    bl_idname = "view3d.toggle_wireframe"
    bl_label = "Toggle Wireframe"
    bl_options = {'REGISTER', 'UNDO'}

    opacity: FloatProperty(
        name="Opacity",
        default=1.0,
        min=0.0,
        max=1.0,
        description="Wireframe opacity (if supported)"
    )

    def execute(self, context):
        current = _any_view3d_wireframe_active()
        new_state = not current
        try:
            _set_view3d_overlay_wireframe(new_state, opacity=self.opacity if new_state else None)
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        try:
                            area.tag_redraw()
                        except Exception:
                            pass
        except Exception as e:
            self.report({'WARNING'}, f"Could not update View3D overlays: {e}")

        self.report({'INFO'}, f"Wireframe overlay {'enabled' if new_state else 'disabled'}")
        return {'FINISHED'}


# ------------------ Leaf Veins from Texture (Modifier-based, visible in Solid) ------------------

def _get_leaf_material(obj):
    for slot in obj.material_slots:
        if slot.material and slot.material.use_nodes:
            return slot.material
    return None


def _find_tex_image_node(mat):
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            return node
    return None


def _extract_green_channel_texture(image, name='LeafVeins_Tex'):
    """Create a Blender Texture (type IMAGE) from the green channel of an image.
    Returns an existing texture with the same name if already present."""
    existing = bpy.data.textures.get(name)
    if existing:
        return existing

    tex = bpy.data.textures.new(name=name, type='IMAGE')
    tex.image = image
    tex.use_color_ramp = False
    # Use green channel by mapping RGB → intensity via color ramp on green
    tex.factor_red   = 0.0
    tex.factor_blue  = 0.0
    tex.use_alpha    = False
    return tex


class OBJECT_OT_leaf_veins_bump(bpy.types.Operator):
    """Add leaf vein relief visible in Solid viewport using a Displace modifier.
    Reads the leaf texture and applies real geometry displacement — no render needed."""
    bl_idname  = 'object.leaf_veins_bump'
    bl_label   = 'Add Leaf Veins Relief'
    bl_options = {'REGISTER', 'UNDO'}

    method: EnumProperty(
        name='Method',
        items=[
            ('DISPLACE', 'Displace Modifier',
             'Real geometry — visible in Solid viewport, works everywhere'),
            ('BUMP',     'Bump (Shader)',
             'Fake relief via normals — Material/Rendered view only, no geometry'),
            ('BOTH',     'Both',
             'Displace modifier + Bump shader combined'),
        ],
        default='DISPLACE',
    )
    strength: FloatProperty(
        name='Strength',
        default=0.005, min=0.0, max=0.010, precision=4,
        description='Displacement strength in scene units',
    )
    midlevel: FloatProperty(
        name='Midlevel',
        default=0.0, min=0.0, max=1.0,
        description='Midlevel: 0.0 = veins pushed outward only, 0.5 = symmetric',
    )
    subdivisions: IntProperty(
        name='Subdivisions',
        default=4, min=1, max=8,
        description='Subdivision levels before displacement (more = smoother veins)',
    )
    invert: BoolProperty(
        name='Invert',
        default=False,
        description='Invert so veins are recessed instead of raised',
    )
    bump_strength: FloatProperty(
        name='Bump Strength',
        default=0.8, min=0.0, max=2.0,
        description='Additional bump strength (shader only)',
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        mat = _get_leaf_material(obj)

        # Get image from material or first image texture in scene
        image = None
        if mat:
            tex_node = _find_tex_image_node(mat)
            if tex_node:
                image = tex_node.image

        if image is None:
            self.report({'ERROR'},
                "No image texture found on the material. "
                "Import a leaf PNG first.")
            return {'CANCELLED'}

        # --- Remove existing veins setup ---
        _remove_veins_setup(obj, mat)

        # --- 1. Subdivision modifier (must come before Displace) ---
        if self.method in ('DISPLACE', 'BOTH'):
            sub = obj.modifiers.new('LeafVeins_Sub', 'SUBSURF')
            sub.levels        = self.subdivisions
            sub.render_levels = self.subdivisions
            sub.use_limit_surface = False

        # --- 2. Displace modifier with texture ---
        if self.method in ('DISPLACE', 'BOTH'):
            tex = _extract_green_channel_texture(image, name='LeafVeins_Tex')

            disp_mod = obj.modifiers.new('LeafVeins_Displace', 'DISPLACE')
            disp_mod.texture        = tex
            disp_mod.texture_coords = 'UV'
            disp_mod.uv_layer       = (obj.data.uv_layers.active.name
                                       if obj.data.uv_layers.active else '')
            disp_mod.direction      = 'NORMAL'
            disp_mod.mid_level      = self.midlevel
            disp_mod.strength       = -self.strength if self.invert else self.strength

        # --- 3. Optional bump shader ---
        if self.method in ('BUMP', 'BOTH') and mat:
            _add_veins_bump_shader(mat, self.bump_strength)

        self.report({'INFO'},
            f"Leaf veins ({self.method}) applied on '{obj.name}' — "
            f"visible in Solid viewport")
        return {'FINISHED'}


def _add_veins_bump_shader(mat, strength=0.8):
    """Add bump shader nodes (visible in Material/Rendered view)."""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    tex_node = _find_tex_image_node(mat)
    bsdf     = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not tex_node or not bsdf:
        return

    if any(n.name == 'LeafVeins_Bump' for n in nodes):
        nodes['LeafVeins_Bump'].inputs['Strength'].default_value = strength
        return

    sep = nodes.new('ShaderNodeSeparateColor')
    sep.name = 'LeafVeins_SepRGB'
    sep.mode = 'RGB'
    sep.location = (tex_node.location.x + 300, tex_node.location.y - 300)

    gamma = nodes.new('ShaderNodeGamma')
    gamma.name = 'LeafVeins_Gamma'
    gamma.location = (sep.location.x + 200, sep.location.y)
    gamma.inputs['Gamma'].default_value = 2.2

    bump = nodes.new('ShaderNodeBump')
    bump.name = 'LeafVeins_Bump'
    bump.label = 'Leaf Veins Bump'
    bump.location = (gamma.location.x + 200, gamma.location.y - 50)
    bump.inputs['Strength'].default_value = strength
    bump.inputs['Distance'].default_value = 0.02

    links.new(tex_node.outputs['Color'], sep.inputs['Color'])
    links.new(sep.outputs['Green'],      gamma.inputs['Color'])
    links.new(gamma.outputs['Color'],    bump.inputs['Height'])
    links.new(bump.outputs['Normal'],    bsdf.inputs['Normal'])


def _remove_veins_setup(obj, mat=None):
    """Remove LeafVeins_ modifiers and shader nodes."""
    # Remove modifiers
    to_remove = [m for m in obj.modifiers if m.name.startswith('LeafVeins_')]
    for m in to_remove:
        obj.modifiers.remove(m)

    # Remove texture
    tex = bpy.data.textures.get('LeafVeins_Tex')
    if tex:
        bpy.data.textures.remove(tex)

    # Remove shader nodes
    if mat and mat.use_nodes:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf:
            for link in list(links):
                if link.to_node == bsdf and link.to_socket.name == 'Normal':
                    links.remove(link)
        to_del = [n for n in nodes if n.name.startswith('LeafVeins_')]
        for n in to_del:
            nodes.remove(n)


class OBJECT_OT_leaf_veins_toggle_invert(bpy.types.Operator):
    """Invert leaf veins displacement direction (flips the Displace strength sign)."""
    bl_idname  = 'object.leaf_veins_toggle_invert'
    bl_label   = 'Invert Leaf Veins'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        disp_mod = next((m for m in obj.modifiers
                         if m.name == 'LeafVeins_Displace'), None) if obj else None
        if not disp_mod:
            self.report({'WARNING'}, "No active leaf veins modifier found.")
            return {'CANCELLED'}
        disp_mod.strength = -disp_mod.strength
        return {'FINISHED'}


class OBJECT_OT_leaf_veins_remove(bpy.types.Operator):
    """Remove all leaf veins modifiers and shader nodes."""
    bl_idname  = 'object.leaf_veins_remove'
    bl_label   = 'Remove Leaf Veins'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        has_mod = any(m.name.startswith('LeafVeins_') for m in obj.modifiers)
        mat = _get_leaf_material(obj)
        has_nodes = mat and any(n.name.startswith('LeafVeins_')
                                for n in mat.node_tree.nodes)
        return has_mod or has_nodes

    def execute(self, context):
        obj = context.active_object
        mat = _get_leaf_material(obj)
        _remove_veins_setup(obj, mat)
        self.report({'INFO'}, "Leaf veins removed.")
        return {'FINISHED'}




def _set_thin_wall(node, value):
    """Enable or disable Thin Wall on a Principled BSDF (Blender 5.2+).
    Returns True if the property was found and set."""
    # In Blender 5.2 Thin Wall is input index 5
    if 'Thin Wall' in node.inputs:
        try:
            node.inputs['Thin Wall'].default_value = value
            return True
        except Exception:
            pass
    for inp_name in ('Thin Walled',):
        if inp_name in node.inputs:
            try:
                node.inputs[inp_name].default_value = value
                return True
            except Exception:
                pass
    for attr in ('thin_wall', 'thin_walled'):
        if hasattr(node, attr):
            setattr(node, attr, value)
            return True
    return False


def _has_thin_wall_support():
    """Check if current Blender version supports Thin Wall (5.2+)."""
    return bpy.app.version >= (5, 2, 0)


def _is_thin_wall_active(mat):
    """Return True if the material has a ThinWall_ tagged node setup."""
    if not mat or not mat.use_nodes:
        return False
    return any(n.name.startswith('ThinWall_') for n in mat.node_tree.nodes)


def _remove_thin_wall_setup(mat):
    """Remove all ThinWall_ nodes and restore the original Principled BSDF → Output link."""
    if not mat or not mat.use_nodes:
        return
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    # Find the original principled (kept but disconnected)
    original_principled = None
    for n in nodes:
        if n.type == 'BSDF_PRINCIPLED' and not n.name.startswith('ThinWall_'):
            original_principled = n
            break

    # Find Material Output
    mat_output = None
    for n in nodes:
        if n.type == 'OUTPUT_MATERIAL' and n.is_active_output:
            mat_output = n
            break
    if mat_output is None:
        for n in nodes:
            if n.type == 'OUTPUT_MATERIAL':
                mat_output = n
                break

    # Remove all ThinWall_ nodes
    to_remove = [n for n in nodes if n.name.startswith('ThinWall_')]
    for n in to_remove:
        nodes.remove(n)

    # Reconnect original principled to output
    if original_principled and mat_output:
        links.new(original_principled.outputs['BSDF'], mat_output.inputs['Surface'])


def _apply_common_bsdf_settings(node, roughness, ior, thin_wall):
    """Apply shared Principled BSDF settings for leaf translucency."""
    node.distribution = 'MULTI_GGX'
    try:
        node.subsurface_method = 'RANDOM_WALK'
    except Exception:
        pass
    node.inputs['Roughness'].default_value = roughness
    if 'IOR' in node.inputs:
        node.inputs['IOR'].default_value = ior
    if thin_wall:
        _set_thin_wall(node, True)
    if 'Transmission Weight' in node.inputs:
        node.inputs['Transmission Weight'].default_value = 1.0
    elif 'Transmission' in node.inputs:
        node.inputs['Transmission'].default_value = 1.0
    if 'Specular IOR Level' in node.inputs:
        node.inputs['Specular IOR Level'].default_value = 0.5
    if 'Subsurface Scale' in node.inputs:
        node.inputs['Subsurface Scale'].default_value = 0.005
    try:
        for ps in node.panel_states:
            ps.is_collapsed = True
    except Exception:
        pass


def _configure_principled_bsdf(node, enable_thin_wall=True, face='front'):
    """Configure a Principled BSDF for leaf translucency.

    Blender 5.2+ (native Thin Wall):
      front — Roughness 0.114, IOR 3.2,  Thin Wall ON
      back  — Roughness 0.332, IOR 1.1,  Thin Wall ON

    Blender 5.0/5.1 (fallback):
      front & back — Roughness 0.673, IOR 1.1, Thin Wall OFF
    """
    if enable_thin_wall and _has_thin_wall_support():
        if face == 'back':
            _apply_common_bsdf_settings(node,
                roughness=0.33218181133270264,
                ior=1.1,
                thin_wall=True)
        else:
            _apply_common_bsdf_settings(node,
                roughness=0.11400000005960464,
                ior=3.2,
                thin_wall=True)
    else:
        _apply_common_bsdf_settings(node,
            roughness=0.6727272868156433,
            ior=1.1,
            thin_wall=False)


class OBJECT_OT_leaf_thin_wall(bpy.types.Operator):
    """Build a Thin Wall shader for realistic leaf translucency.
    Blender 5.2+: uses native Thin Wall on Principled BSDF.
    Blender 5.0/5.1: uses a Translucent BSDF fallback for a similar look.
    Both faces receive the leaf texture with identical settings."""
    bl_idname  = 'object.leaf_thin_wall'
    bl_label   = 'Add Thin Wall'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        mat = _get_leaf_material(obj)
        return mat is not None and not _is_thin_wall_active(mat)

    def execute(self, context):
        obj = context.active_object
        mat = _get_leaf_material(obj)
        if not mat or not mat.use_nodes:
            self.report({'WARNING'}, "No material with nodes found.")
            return {'CANCELLED'}

        tree = mat.node_tree
        nodes = tree.nodes
        links = tree.links

        # --- Find existing nodes ---
        orig_principled = None
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                orig_principled = n
                break
        if orig_principled is None:
            self.report({'WARNING'}, "No Principled BSDF node found.")
            return {'CANCELLED'}

        # Find the Image Texture connected to Base Color
        tex_node = None
        if orig_principled.inputs['Base Color'].is_linked:
            tex_node = orig_principled.inputs['Base Color'].links[0].from_node

        # Find Material Output
        mat_output = None
        for n in nodes:
            if n.type == 'OUTPUT_MATERIAL' and n.is_active_output:
                mat_output = n
                break
        if mat_output is None:
            for n in nodes:
                if n.type == 'OUTPUT_MATERIAL':
                    mat_output = n
                    break
        if mat_output is None:
            self.report({'WARNING'}, "No Material Output node found.")
            return {'CANCELLED'}

        ref_x = orig_principled.location.x
        ref_y = orig_principled.location.y

        use_native = _has_thin_wall_support()

        if use_native:
            # ==========================================
            # Blender 5.2+  —  native Thin Wall path
            # ==========================================

            # Front BSDF
            front = nodes.new('ShaderNodeBsdfPrincipled')
            front.name  = 'ThinWall_Front_BSDF'
            front.label = 'Front (Thin Wall)'
            front.location = (ref_x, ref_y - 250)
            _configure_principled_bsdf(front, enable_thin_wall=True)

            # Back BSDF — Roughness 0.332, IOR 1.1, Thin Wall ON
            back = nodes.new('ShaderNodeBsdfPrincipled')
            back.name  = 'ThinWall_Back_BSDF'
            back.label = 'Back (Thin Wall)'
            back.location = (ref_x, ref_y - 660)
            _configure_principled_bsdf(back, enable_thin_wall=True, face='back')

            # Connect texture to both
            if tex_node:
                for bsdf in (front, back):
                    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                    if 'Alpha' in tex_node.outputs and 'Alpha' in bsdf.inputs:
                        links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])

            # Geometry node
            geom = nodes.new('ShaderNodeNewGeometry')
            geom.name = 'ThinWall_Geometry'
            geom.location = (ref_x - 200, ref_y + 30)

            # Mix Shader
            mix = nodes.new('ShaderNodeMixShader')
            mix.name  = 'ThinWall_MixShader'
            mix.label = 'Mix Shader'
            mix.location = (ref_x + 300, ref_y - 250)

            # Wire
            links.new(geom.outputs['Backfacing'], mix.inputs['Fac'])
            links.new(front.outputs['BSDF'], mix.inputs[1])
            links.new(back.outputs['BSDF'], mix.inputs[2])
            links.new(mix.outputs['Shader'], mat_output.inputs['Surface'])

            self.report({'INFO'}, "Thin Wall shader (native 5.2) created.")

        else:
            # Blender 5.0/5.1 — deux Principled BSDF identiques + alpha mix
            front = nodes.new('ShaderNodeBsdfPrincipled')
            front.name  = 'ThinWall_Front_BSDF'
            front.label = 'Front'
            front.location = (ref_x, ref_y - 250)
            _configure_principled_bsdf(front, enable_thin_wall=False)

            back = nodes.new('ShaderNodeBsdfPrincipled')
            back.name  = 'ThinWall_Back_BSDF'
            back.label = 'Front'
            back.location = (ref_x, ref_y - 640)
            _configure_principled_bsdf(back, enable_thin_wall=False)

            if tex_node:
                links.new(tex_node.outputs['Color'], front.inputs['Base Color'])
                if 'Alpha' in tex_node.outputs and 'Alpha' in front.inputs:
                    links.new(tex_node.outputs['Alpha'], front.inputs['Alpha'])
                links.new(tex_node.outputs['Color'], back.inputs['Base Color'])

            geom = nodes.new('ShaderNodeNewGeometry')
            geom.name = 'ThinWall_Geometry'
            geom.location = (ref_x - 180, ref_y - 70)

            mix_face = nodes.new('ShaderNodeMixShader')
            mix_face.name  = 'ThinWall_MixFace'
            mix_face.label = 'Front / Back'
            mix_face.location = (ref_x + 310, ref_y - 250)

            links.new(geom.outputs['Backfacing'], mix_face.inputs['Fac'])
            links.new(front.outputs['BSDF'],      mix_face.inputs[1])
            links.new(back.outputs['BSDF'],       mix_face.inputs[2])

            transparent = nodes.new('ShaderNodeBsdfTransparent')
            transparent.name = 'ThinWall_Transparent'
            transparent.location = (ref_x + 310, ref_y - 45)

            mix_alpha = nodes.new('ShaderNodeMixShader')
            mix_alpha.name  = 'ThinWall_MixAlpha'
            mix_alpha.label = 'Alpha Mix'
            mix_alpha.location = (ref_x + 510, ref_y - 45)

            if tex_node and 'Alpha' in tex_node.outputs:
                links.new(tex_node.outputs['Alpha'], mix_alpha.inputs['Fac'])
            links.new(transparent.outputs['BSDF'],  mix_alpha.inputs[1])
            links.new(mix_face.outputs['Shader'],   mix_alpha.inputs[2])

            links.new(mix_alpha.outputs['Shader'], mat_output.inputs['Surface'])

            mat.blend_method = 'HASHED'
            if hasattr(mat, 'surface_render_method'):
                mat.surface_render_method = 'DITHERED'

            self.report({'INFO'}, "Thin Wall 5.0 — deux Principled BSDF identiques.")

        # Move original principled aside
        orig_principled.location = (ref_x - 400, ref_y + 300)

        return {'FINISHED'}


class OBJECT_OT_leaf_thin_wall_remove(bpy.types.Operator):
    """Remove the Thin Wall shader setup and restore the original material."""
    bl_idname  = 'object.leaf_thin_wall_remove'
    bl_label   = 'Remove Thin Wall'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return False
        mat = _get_leaf_material(obj)
        return _is_thin_wall_active(mat)

    def execute(self, context):
        obj = context.active_object
        mat = _get_leaf_material(obj)
        _remove_thin_wall_setup(mat)
        self.report({'INFO'}, "Thin Wall setup removed.")
        return {'FINISHED'}


class VIEW3D_PT_leaf_retopo_panel(bpy.types.Panel):
    bl_label = "Leaf Retopo"
    bl_idname = "VIEW3D_PT_leaf_retopo"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Leaf Retopo'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Section 1: Import & Retopo
        icon = 'TRIA_DOWN' if getattr(scene, "lr_show_import_retopo", True) else 'TRIA_RIGHT'
        row = layout.row(align=True)
        row.prop(scene, "lr_show_import_retopo", text="Import & Retopo", emboss=False, icon=icon)
        if scene.lr_show_import_retopo:
            col = layout.column(align=True)

            row_imp = col.row(align=True)
            # Keep the UI label "Import Leaf PNG (Pure NGON)" but use the improved operator implementation
            row_imp.operator(ImportLeafClean.bl_idname, text="Import Leaf PNG (Pure NGON)")

            # wireframe toggle button — drawn depressed when any view has wireframe overlay active
            is_active = _any_view3d_wireframe_active()
            try:
                row_imp.operator(VIEW3D_OT_toggle_wireframe.bl_idname, text="", icon='SHADING_WIRE', depress=is_active)
            except TypeError:
                row_imp.operator(VIEW3D_OT_toggle_wireframe.bl_idname, text="", icon='SHADING_WIRE')

            row_gc = col.row(align=True)
            op_gc = row_gc.operator(MESH_OT_grid_cut.bl_idname, text="Grid Cut", icon='MOD_WIREFRAME')
            try:
                op_gc.use_density     = True
                op_gc.grid_density    = scene.gridcut_density
                op_gc.merge_threshold = scene.gridcut_merge_thresh
                op_gc.separate_parts  = scene.gridcut_separate
            except Exception:
                pass
            row_gc.operator('mesh.remove_grid_cut', text="", icon='TRASH')

        layout.separator()

        # Section 2: Curve Modifier
        icon = 'TRIA_DOWN' if getattr(scene, "lr_show_curve_modifier", True) else 'TRIA_RIGHT'
        row = layout.row(align=True)
        row.prop(scene, "lr_show_curve_modifier", text="Curve Modifier", emboss=False, icon=icon)
        if scene.lr_show_curve_modifier:
            col = layout.column(align=True)
            col.operator(OBJECT_OT_convert_to_curve.bl_idname, text="Convert to Curve")
            rowc = col.row(align=True)
            rowc.operator(OBJECT_OT_simplify_curve_points.bl_idname, text="Simplify Curve")
            rowc.operator(OBJECT_OT_curve_loft_to_quads.bl_idname, text="Loft Curve → Quad Mesh")

        layout.separator()

        # Section 4: Leaf Veins
        icon = 'TRIA_DOWN' if getattr(scene, 'lr_show_veins', True) else 'TRIA_RIGHT'
        row = layout.row(align=True)
        row.prop(scene, 'lr_show_veins', text='Leaf Veins Relief', emboss=False, icon=icon)
        if scene.lr_show_veins:
            col = layout.column(align=True)

            obj = context.active_object
            if obj and obj.type == 'MESH':
                mat = _get_leaf_material(obj)
                tex_node = _find_tex_image_node(mat) if mat else None

                if tex_node and tex_node.image:
                    box = col.box()
                    box.label(text=f"Texture: {tex_node.image.name}", icon='IMAGE_DATA')
                else:
                    col.label(text="Import a leaf PNG first", icon='INFO')

                # Live controls for Displace modifier
                disp_mod = next((m for m in obj.modifiers
                                 if m.name == 'LeafVeins_Displace'), None)
                sub_mod  = next((m for m in obj.modifiers
                                 if m.name == 'LeafVeins_Sub'), None)
                if disp_mod:
                    box = col.box()
                    box.label(text="Veins Active — Solid Viewport", icon='CHECKMARK')
                    col2 = box.column(align=True)
                    # Show strength clamped 0.000–0.010
                    row_s = col2.row(align=True)
                    row_s.prop(disp_mod, 'strength', text="Strength", slider=True)
                    # Enforce max 0.010 silently
                    if abs(disp_mod.strength) > 0.010:
                        disp_mod.strength = 0.010 * (1 if disp_mod.strength > 0 else -1)
                    col2.prop(disp_mod, 'mid_level',  text="Midlevel",  slider=True)
                    if sub_mod:
                        col2.prop(sub_mod, 'levels',  text="Subdivisions")
                    row_i = col2.row(align=True)
                    row_i.label(text="Invert")
                    inv_icon = 'CHECKBOX_HLT' if disp_mod.strength < 0 else 'CHECKBOX_DEHLT'
                    row_i.operator('object.leaf_veins_toggle_invert',
                                    text='', icon=inv_icon,
                                    depress=(disp_mod.strength < 0))
                    box.separator()
                    box.operator('object.leaf_veins_remove',
                                 text='Remove Veins', icon='X')

            row_v = col.row(align=True)
            op = row_v.operator('object.leaf_veins_bump',
                                text='Add Veins (Solid)', icon='NORMALS_FACE')
            op.method = 'DISPLACE'
            op2 = row_v.operator('object.leaf_veins_bump',
                                 text='+ Bump Shader', icon='MOD_DISPLACE')
            op2.method = 'BOTH'

        layout.separator()

        # Section: Thin Wall (Blender 5.2+)
        icon = 'TRIA_DOWN' if getattr(scene, 'lr_show_thin_wall', True) else 'TRIA_RIGHT'
        row = layout.row(align=True)
        row.prop(scene, 'lr_show_thin_wall', text='Thin Wall Translucency', emboss=False, icon=icon)
        if scene.lr_show_thin_wall:
            col = layout.column(align=True)
            obj = context.active_object
            mat = _get_leaf_material(obj) if (obj and obj.type == 'MESH') else None

            if _is_thin_wall_active(mat):
                box = col.box()
                box.label(text="Thin Wall Active", icon='CHECKMARK')
                box.operator('object.leaf_thin_wall_remove',
                             text='Remove Thin Wall', icon='X')
            else:
                col.operator('object.leaf_thin_wall',
                             text='Add Thin Wall', icon='LIGHT_SUN')

        layout.separator()

        icon = 'TRIA_DOWN' if getattr(scene, "lr_show_geometry_nodes", True) else 'TRIA_RIGHT'
        row = layout.row(align=True)
        row.prop(scene, "lr_show_geometry_nodes", text="Geometry Nodes", emboss=False, icon=icon)
        if scene.lr_show_geometry_nodes:
            col = layout.column(align=True)
            rowg = col.row(align=True)
            rowg.operator(OBJECT_OT_apply_leaf_geometry_node.bl_idname, text="Create & Apply Bend")
            rowg.operator(OBJECT_OT_apply_subdvz_geometry_node.bl_idname, text="Create & Apply Subdivision")


# ------------------ Registration ------------------

# -----------------------------------------------------------------------
# Dependency operators + AddonPreferences
# -----------------------------------------------------------------------

class LEAFRETOPO_OT_InstallDependencies(bpy.types.Operator):
    bl_idname   = "leafretopo.install_dependencies"
    bl_label    = "Install Dependencies"
    bl_description = (
        "Download and install numpy, opencv-python and Pillow into Blender's Python. "
        "Requires an internet connection. Blender may appear frozen during installation."
    )
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        import sys
        self.report({'INFO'},
            "Installing… check the system console (Window > Toggle System Console) for progress.")
        print(f"[Leaf Retopo] Python: {sys.executable}")
        print(f"[Leaf Retopo] Version: {sys.version}")
        print(f"[Leaf Retopo] Install target: {_pip_target_dir()}")

        failed = []
        for imp, pip in _LEAF_DEPS:
            try:
                importlib.import_module(imp)
                print(f"[Leaf Retopo] ✓ {pip} already installed — skipped")
                continue
            except ImportError:
                pass
            print(f"[Leaf Retopo] Installing {pip}…")
            ok = _install_dep(pip, lambda m: print(f"[Leaf Retopo] {m}"))
            if not ok:
                failed.append(pip)

        # Reload globals so the session benefits immediately
        _try_import_deps()

        if failed:
            self.report({'ERROR'},
                f"Failed: {', '.join(failed)}. "
                "Open Window > Toggle System Console for details.")
        else:
            libs = "cv2" if cv2 else "—"
            self.report({'INFO'},
                "Dependencies installed! "
                f"cv2={'✓' if cv2 else '✗'}  numpy={'✓' if np else '✗'}  "
                f"Pillow={'✓' if PIL_Image else '✗'}. "
                "Restart Blender if detection still uses the fallback.")
        return {'FINISHED'}


class LEAFRETOPO_OT_CheckDependencies(bpy.types.Operator):
    bl_idname   = "leafretopo.check_dependencies"
    bl_label    = "Refresh Status"
    bl_description = "Check which libraries are currently available"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        _try_import_deps()
        status = _dep_status()
        msg = " | ".join(f"{'✓' if ok else '✗'} {p}" for p, ok in status)
        self.report({'INFO' if _all_deps_ok() else 'WARNING'}, msg)
        return {'FINISHED'}


class LEAF_RETOPO_Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        status = _dep_status()
        all_ok = all(ok for _, ok in status)

        box = layout.box()
        box.label(text="Leaf Retopo — Optional Libraries (contour detection)", icon='SCRIPT')

        grid = box.column(align=True)
        for pip_name, ok in status:
            row = grid.row()
            row.label(text=pip_name, icon='CHECKMARK' if ok else 'X')
            row.label(text="Installed" if ok else "Not found")

        layout.separator()

        if not all_ok:
            warn = layout.box()
            warn.label(text="Missing libraries → minimal fallback (less precise contours).", icon='ERROR')
            warn.label(text="Click below to install automatically (requires internet).")
            layout.separator()
            col = layout.column()
            col.scale_y = 1.5
            col.operator("leafretopo.install_dependencies",
                         text="Install numpy + opencv-python + Pillow", icon='IMPORT')
        else:
            layout.label(text="All libraries installed — precise contour detection active.", icon='CHECKMARK')

        layout.separator()
        row = layout.row()
        row.operator("leafretopo.check_dependencies", text="Refresh Status", icon='FILE_REFRESH')


classes = (
    LEAFRETOPO_OT_InstallDependencies,
    LEAFRETOPO_OT_CheckDependencies,
    LEAF_RETOPO_Preferences,
    ImportLeafClean,
    MESH_OT_delta_quad_plus,
    MESH_OT_grid_cut,
    MESH_OT_remove_grid_cut,
    OBJECT_OT_simplify_curve_points,
    OBJECT_OT_curve_loft_to_quads,
    OBJECT_OT_convert_to_curve,
    OBJECT_OT_project_grid_retopo,
    OBJECT_OT_call_tris_convert_to_quads,
    OBJECT_OT_apply_leaf_geometry_node,
    OBJECT_OT_apply_subdvz_geometry_node,
    OBJECT_OT_leaf_veins_bump,
    OBJECT_OT_leaf_veins_toggle_invert,
    OBJECT_OT_leaf_veins_remove,
    OBJECT_OT_leaf_thin_wall,
    OBJECT_OT_leaf_thin_wall_remove,
    VIEW3D_OT_toggle_wireframe,
    VIEW3D_PT_leaf_retopo_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    # Run dep check once at startup so globals are fresh
    _try_import_deps()

    # Grid Cut scene properties (defaults)
    bpy.types.Scene.gridcut_density = IntProperty(
        name="Grid Density",
        default=10, min=1, max=100,
        description="Grid density: 1=very coarse, 100=very fine. Adapts to object size automatically"
    )
    bpy.types.Scene.gridcut_active = BoolProperty(
        name="Grid Cut Active", default=False)
    bpy.types.Scene.gridcut_div_x = IntProperty(name="Divisions X", default=4, min=1, max=4096)
    bpy.types.Scene.gridcut_div_y = IntProperty(name="Divisions Y", default=4, min=1, max=4096)
    bpy.types.Scene.gridcut_div_z = IntProperty(name="Divisions Z", default=4, min=1, max=4096)
    bpy.types.Scene.gridcut_use_cellsize = BoolProperty(name="Use Cell Size", default=False)
    bpy.types.Scene.gridcut_cell_size = FloatProperty(name="Cell Size", default=0.2, min=1e-9, soft_max=10.0)
    bpy.types.Scene.gridcut_merge_thresh = FloatProperty(name="Merge Distance", default=1e-6, min=0.0, max=1.0)
    bpy.types.Scene.gridcut_separate = BoolProperty(name="Separate Parts", default=False)
    bpy.types.Scene.gridcut_max_divisions = IntProperty(name="Max Divisions per Axis", default=256, min=4, max=4096)
    bpy.types.Scene.gridcut_cells_limit = IntProperty(name="Max Estimated Cells", default=2000000, min=1000, max=100000000)

    # Panel collapsible states (remember state in scene)
    bpy.types.Scene.lr_show_import_retopo = BoolProperty(name="Show Import & Retopo", default=True)
    bpy.types.Scene.lr_show_curve_modifier = BoolProperty(name="Show Curve Modifier", default=True)
    bpy.types.Scene.lr_show_veins = BoolProperty(name="Show Leaf Veins", default=True)
    bpy.types.Scene.lr_show_trans = BoolProperty(name="Show Leaf Translucency", default=True)
    bpy.types.Scene.lr_show_opacity = BoolProperty(name="Show Leaf Transparency", default=True)
    bpy.types.Scene.lr_show_thin_wall = BoolProperty(name="Show Thin Wall", default=True)
    bpy.types.Scene.lr_show_geometry_nodes = BoolProperty(name="Show Geometry Nodes", default=True)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    for prop in ("gridcut_density", "gridcut_active",
                 "gridcut_div_x", "gridcut_div_y", "gridcut_div_z",
                 "gridcut_use_cellsize", "gridcut_cell_size", "gridcut_merge_thresh",
                 "gridcut_separate", "gridcut_max_divisions", "gridcut_cells_limit",
                 "lr_show_import_retopo", "lr_show_curve_modifier",
                 "lr_show_veins", "lr_show_trans", "lr_show_opacity",
                 "lr_show_thin_wall", "lr_show_geometry_nodes"):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)


if __name__ == "__main__":
    register()
