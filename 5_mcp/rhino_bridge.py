#! python 3
"""Rhino side of the MCP connection.

Run inside Rhino 8: ScriptEditor -> open this file -> Run. It listens on
http://127.0.0.1:8765 and executes drawing commands sent by 5_mcp/mcp_server.py.
"""

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import System
import System.Drawing
import System.Threading
import Rhino
import Rhino.Geometry as rg
import Rhino.DocObjects as rdo

_doc = Rhino.RhinoDoc.ActiveDoc


def _run_on_ui(fn):
    """Execute fn on the Rhino UI thread and return its result."""
    result_box = [None]
    error_box = [None]
    evt = System.Threading.ManualResetEventSlim(False)

    def wrapper():
        try:
            result_box[0] = fn()
        except Exception as e:
            error_box[0] = e
        finally:
            evt.Set()

    Rhino.RhinoApp.InvokeOnUiThread(System.Action(wrapper))
    evt.Wait()
    if error_box[0]:
        raise error_box[0]
    return result_box[0]


TOOLS = [
    {
        "name": "create_sphere",
        "description": "Create a sphere at center (x,y,z) with given radius. Optional: layer (str), r/g/b (0-255).",
        "args": {"x": "float", "y": "float", "z": "float", "radius": "float",
                 "layer": "str?", "r": "int?", "g": "int?", "b": "int?"}
    },
    {
        "name": "create_box",
        "description": "Create a box from corner (x0,y0,z0) to corner (x,y,z). x0/y0/z0 default to 0. Optional: layer, r/g/b.",
        "args": {"x": "float", "y": "float", "z": "float", "x0": "float", "y0": "float", "z0": "float",
                 "layer": "str?", "r": "int?", "g": "int?", "b": "int?"}
    },
    {
        "name": "create_triangle",
        "description": "Create a triangle mesh face with three vertices. Optional: layer, r/g/b.",
        "args": {
            "x1": "float", "y1": "float", "z1": "float",
            "x2": "float", "y2": "float", "z2": "float",
            "x3": "float", "y3": "float", "z3": "float",
            "layer": "str?", "r": "int?", "g": "int?", "b": "int?"
        }
    },
    {
        "name": "create_layer",
        "description": "Create or update a layer with a given name and RGB color.",
        "args": {"name": "str", "r": "int", "g": "int", "b": "int"}
    },
    {
        "name": "set_object_color",
        "description": "Set the display color of an object by its GUID.",
        "args": {"id": "str", "r": "int", "g": "int", "b": "int"}
    },
    {
        "name": "set_layer_objects_color",
        "description": "Set the color of all objects on a given layer.",
        "args": {"layer": "str", "r": "int", "g": "int", "b": "int"}
    },
    {
        "name": "delete_object",
        "description": "Delete a specific object by its GUID string.",
        "args": {"id": "str"}
    },
    {
        "name": "delete_all_objects",
        "description": "Delete all objects in the Rhino document.",
        "args": {}
    },
    {
        "name": "create_polyline",
        "description": "Create a polyline from a list of [x,y,z] points. Set closed=true to close it. Optional: layer, r/g/b (0-255).",
        "args": {"points": "list", "closed": "bool?", "layer": "str?", "r": "int?", "g": "int?", "b": "int?"}
    },
    {
        "name": "curve_to_surface",
        "description": "Create a planar surface from a closed curve/polyline by its GUID. Optional: layer, r/g/b.",
        "args": {"id": "str", "layer": "str?", "r": "int?", "g": "int?", "b": "int?"}
    },
    {
        "name": "curve_to_extrusion",
        "description": "Extrude a closed curve/polyline by its GUID along Z by 'height' to create a solid. Optional: layer, r/g/b.",
        "args": {"id": "str", "height": "float", "layer": "str?", "r": "int?", "g": "int?", "b": "int?"}
    },
    {
        "name": "delete_empty_layers",
        "description": "Delete all layers that have no objects on them (skips Default/layer 0).",
        "args": {}
    },
    {
        "name": "zoom_extents",
        "description": "Zoom all viewports to show all objects (Zoom Extents).",
        "args": {}
    },
    {
        "name": "create_text_dot",
        "description": "Create a text dot label at (x,y,z) with given text. Always visible on screen. Optional: layer, r/g/b.",
        "args": {"x": "float", "y": "float", "z": "float", "text": "str",
                 "layer": "str?", "r": "int?", "g": "int?", "b": "int?"}
    },
    {
        "name": "create_heatmap_plane",
        "description": "Create a flat colored mesh from a grid of values (heatmap). values is a flat list of ny*nx numbers in row-major order.",
        "args": {"ox": "float", "oy": "float", "z": "float",
                 "nx": "int", "ny": "int", "dx": "float", "dy": "float",
                 "values": "list", "vmin": "float", "vmax": "float",
                 "colors": "list?", "smooth": "bool?", "layer": "str?"}
    },
    {
        "name": "set_layer_visible",
        "description": "Show or hide a layer by name. visible=true to show, visible=false to hide.",
        "args": {"name": "str", "visible": "bool"}
    },
    {
        "name": "create_planar_surface",
        "description": "Create a planar surface from a closed list of [x,y,z] points (3 or more, coplanar). Optional: holes (list of point lists), layer, r/g/b.",
        "args": {"points": "list", "holes": "list?", "layer": "str?", "r": "int?", "g": "int?", "b": "int?"}
    },
    {
        "name": "create_text",
        "description": "Create flat text in the XY plane at (x,y,z). Optional: height (model units, default 0.25), align (left/center/right), layer, r/g/b.",
        "args": {"x": "float", "y": "float", "z": "float", "text": "str", "height": "float?", "align": "str?",
                 "layer": "str?", "r": "int?", "g": "int?", "b": "int?"}
    },
    {
        "name": "delete_layer_objects",
        "description": "Delete all objects on the layer with this name.",
        "args": {"name": "str"}
    },
    {
        "name": "get_scene_info",
        "description": "Return a summary of all objects and layers in the current Rhino document: layer names, object counts per layer, object types, and bounding boxes.",
        "args": {}
    },
]


def _find_layer(name):
    for i in range(_doc.Layers.Count):
        layer = _doc.Layers[i]
        if not layer.IsDeleted and layer.Name == name:
            return i
    return -1


def _apply_attrs(obj_id, args):
    layer_name = args.get("layer")
    has_color = "r" in args and "g" in args and "b" in args
    if not layer_name and not has_color:
        return
    obj = _doc.Objects.Find(obj_id)
    if obj is None:
        return
    attr = obj.Attributes.Duplicate()
    if layer_name:
        idx = _find_layer(layer_name)
        if idx >= 0:
            attr.LayerIndex = idx
    if has_color:
        attr.ColorSource = rdo.ObjectColorSource.ColorFromObject
        attr.ObjectColor = System.Drawing.Color.FromArgb(
            int(args["r"]), int(args["g"]), int(args["b"])
        )
    _doc.Objects.ModifyAttributes(obj_id, attr, True)


def create_sphere(args):
    def action():
        center = rg.Point3d(args["x"], args["y"], args["z"])
        sphere = rg.Sphere(center, args["radius"])
        obj_id = _doc.Objects.AddSphere(sphere)
        _apply_attrs(obj_id, args)
        _doc.Views.Redraw()
        return str(obj_id)
    return {"ok": True, "id": _run_on_ui(action)}


def create_box(args):
    def action():
        x0, y0, z0 = args.get("x0", 0), args.get("y0", 0), args.get("z0", 0)
        x, y, z = args["x"], args["y"], args["z"]
        box = rg.Box(rg.BoundingBox(rg.Point3d(x0, y0, z0), rg.Point3d(x, y, z)))
        obj_id = _doc.Objects.AddBox(box)
        _apply_attrs(obj_id, args)
        _doc.Views.Redraw()
        return str(obj_id)
    return {"ok": True, "id": _run_on_ui(action)}


def create_triangle(args):
    def action():
        mesh = rg.Mesh()
        mesh.Vertices.Add(args["x1"], args["y1"], args["z1"])
        mesh.Vertices.Add(args["x2"], args["y2"], args["z2"])
        mesh.Vertices.Add(args["x3"], args["y3"], args["z3"])
        mesh.Faces.AddFace(0, 1, 2)
        mesh.Normals.ComputeNormals()
        obj_id = _doc.Objects.AddMesh(mesh)
        _apply_attrs(obj_id, args)
        _doc.Views.Redraw()
        return str(obj_id)
    return {"ok": True, "id": _run_on_ui(action)}


def create_layer(args):
    def action():
        name = args["name"]
        color = System.Drawing.Color.FromArgb(int(args["r"]), int(args["g"]), int(args["b"]))
        idx = _find_layer(name)
        if idx < 0:
            layer = rdo.Layer()
            layer.Name = name
            layer.Color = color
            idx = _doc.Layers.Add(layer)
        else:
            _doc.Layers[idx].Color = color
        _doc.Views.Redraw()
        return idx
    return {"ok": True, "layer_index": _run_on_ui(action)}


def set_object_color(args):
    def action():
        obj_id = System.Guid(args["id"])
        obj = _doc.Objects.Find(obj_id)
        if obj is None:
            return False
        attr = obj.Attributes.Duplicate()
        attr.ColorSource = rdo.ObjectColorSource.ColorFromObject
        attr.ObjectColor = System.Drawing.Color.FromArgb(
            int(args["r"]), int(args["g"]), int(args["b"])
        )
        _doc.Objects.ModifyAttributes(obj_id, attr, True)
        _doc.Views.Redraw()
        return True
    ok = _run_on_ui(action)
    return {"ok": ok}


def set_layer_objects_color(args):
    def action():
        layer_name = args["layer"]
        color = System.Drawing.Color.FromArgb(int(args["r"]), int(args["g"]), int(args["b"]))
        idx = _find_layer(layer_name)
        if idx < 0:
            raise ValueError("Layer not found: " + layer_name)
        count = 0
        for obj in _doc.Objects:
            if obj.Attributes.LayerIndex == idx:
                attr = obj.Attributes.Duplicate()
                attr.ColorSource = rdo.ObjectColorSource.ColorFromObject
                attr.ObjectColor = color
                _doc.Objects.ModifyAttributes(obj.Id, attr, True)
                count += 1
        _doc.Views.Redraw()
        return count
    return {"ok": True, "modified": _run_on_ui(action)}


def delete_object(args):
    def action():
        obj_id = System.Guid(args["id"])
        ok = _doc.Objects.Delete(obj_id, True)
        _doc.Views.Redraw()
        return ok
    return {"ok": _run_on_ui(action)}


def create_polyline(args):
    def action():
        pts = [rg.Point3d(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in args["points"]]
        pl = rg.Polyline(pts)
        if args.get("closed", False) and (pts[0].X != pts[-1].X or pts[0].Y != pts[-1].Y or pts[0].Z != pts[-1].Z):
            pl.Add(pts[0])
        obj_id = _doc.Objects.AddPolyline(pl)
        _apply_attrs(obj_id, args)
        _doc.Views.Redraw()
        return str(obj_id)
    return {"ok": True, "id": _run_on_ui(action)}


def curve_to_surface(args):
    def action():
        obj_id = System.Guid(args["id"])
        obj = _doc.Objects.Find(obj_id)
        if obj is None:
            raise ValueError("Object not found: " + args["id"])
        curve = obj.Geometry
        breps = rg.Brep.CreatePlanarBreps(curve, _doc.ModelAbsoluteTolerance)
        if not breps:
            raise ValueError("Could not create planar surface — is the curve closed and planar?")
        srf_id = _doc.Objects.AddBrep(breps[0])
        _apply_attrs(srf_id, args)
        _doc.Views.Redraw()
        return str(srf_id)
    return {"ok": True, "id": _run_on_ui(action)}


def create_planar_surface(args):
    def action():
        def closed_curve(points):
            pts = [rg.Point3d(p[0], p[1], p[2]) for p in points]
            pts.append(pts[0])
            return rg.PolylineCurve(pts)
        curves = System.Collections.Generic.List[rg.Curve]()
        curves.Add(closed_curve(args["points"]))
        for hole in args.get("holes", []):          # inner boundaries, e.g. window openings
            curves.Add(closed_curve(hole))
        breps = rg.Brep.CreatePlanarBreps(curves, _doc.ModelAbsoluteTolerance)
        if not breps:
            raise ValueError("Could not create planar surface — are the points coplanar?")
        srf_id = _doc.Objects.AddBrep(breps[0])
        _apply_attrs(srf_id, args)
        return str(srf_id)
    return {"ok": True, "id": _run_on_ui(action)}


def delete_empty_layers(args):
    def action():
        used = set()
        for obj in _doc.Objects:
            if not obj.IsDeleted:
                used.add(obj.Attributes.LayerIndex)
        deleted = []
        for i in range(_doc.Layers.Count - 1, -1, -1):
            layer = _doc.Layers[i]
            if layer.IsDeleted: continue
            if i == 0: continue  # keep Default layer
            if i not in used:
                name = str(layer.Name)
                if _doc.Layers.Delete(i, True):
                    deleted.append(name)
        _doc.Views.Redraw()
        return deleted
    return {"ok": True, "deleted": _run_on_ui(action)}


def zoom_extents(args):
    def action():
        for view in _doc.Views:
            view.ActiveViewport.ZoomExtents()
        _doc.Views.Redraw()
    _run_on_ui(action)
    return {"ok": True}


def curve_to_extrusion(args):
    def action():
        obj_id = System.Guid(args["id"])
        obj = _doc.Objects.Find(obj_id)
        if obj is None:
            raise ValueError("Object not found: " + args["id"])
        curve = obj.Geometry
        height = float(args["height"])
        extrusion = rg.Extrusion.Create(curve, height, True)
        if extrusion is None:
            raise ValueError("Could not create extrusion — is the curve closed and planar?")
        ext_id = _doc.Objects.AddExtrusion(extrusion)
        _apply_attrs(ext_id, args)
        for view in _doc.Views:
            view.ActiveViewport.ZoomExtents()
        _doc.Views.Redraw()
        return str(ext_id)
    return {"ok": True, "id": _run_on_ui(action)}


def create_heatmap_plane(args):
    def action():
        ox   = float(args["ox"])
        oy   = float(args["oy"])
        z    = float(args["z"])
        nx   = int(args["nx"])
        ny   = int(args["ny"])
        dx   = float(args["dx"])
        dy   = float(args["dy"])
        vals = args["values"]
        vmin = float(args.get("vmin", 0))
        vmax = float(args.get("vmax", 2000))
        colors = args.get("colors")       # optional [r,g,b] per cell, overrides the built-in scale

        def val_to_color(v):
            t = max(0.0, min(1.0, (float(v) - vmin) / max(vmax - vmin, 1.0)))
            if t < 0.25:
                r, g, b = 0, int(t * 4 * 255), 255
            elif t < 0.5:
                r, g, b = 0, 255, int((1 - (t - 0.25) * 4) * 255)
            elif t < 0.75:
                r, g, b = int((t - 0.5) * 4 * 255), 255, 0
            else:
                r, g, b = 255, int((1 - (t - 0.75) * 4) * 255), 0
            return System.Drawing.Color.FromArgb(r, g, b)

        if args.get("smooth"):
            # vertices at the cell centres (sensor points), colours blend between them
            mesh = rg.Mesh()
            for iy in range(ny):
                for ix in range(nx):
                    k = iy * nx + ix
                    mesh.Vertices.Add(ox + (ix + 0.5) * dx, oy + (iy + 0.5) * dy, z)
                    mesh.VertexColors.Add(System.Drawing.Color.FromArgb(*[int(c) for c in colors[k]]) if colors
                                          else val_to_color(vals[k]))
            for iy in range(ny - 1):
                for ix in range(nx - 1):
                    k = iy * nx + ix
                    mesh.Faces.AddFace(k, k + 1, k + nx + 1, k + nx)
            mesh.Normals.ComputeNormals()
            obj_id = _doc.Objects.AddMesh(mesh)
            _apply_attrs(obj_id, args)
            _doc.Views.Redraw()
            return str(obj_id)

        mesh = rg.Mesh()
        for iy in range(ny):
            for ix in range(nx):
                x0 = ox + ix * dx;  x1 = x0 + dx
                y0 = oy + iy * dy;  y1 = y0 + dy
                vi = mesh.Vertices.Count
                mesh.Vertices.Add(x0, y0, z)
                mesh.Vertices.Add(x1, y0, z)
                mesh.Vertices.Add(x1, y1, z)
                mesh.Vertices.Add(x0, y1, z)
                k = iy * nx + ix
                col = (System.Drawing.Color.FromArgb(*[int(c) for c in colors[k]]) if colors
                       else val_to_color(vals[k]))
                for _ in range(4):
                    mesh.VertexColors.Add(col)
                mesh.Faces.AddFace(vi, vi+1, vi+2, vi+3)

        mesh.Normals.ComputeNormals()
        mesh.Weld(0.01)   # merge coincident vertices -> interior edges -> no naked edge lines
        obj_id = _doc.Objects.AddMesh(mesh)

        # Hide mesh wires
        obj = _doc.Objects.Find(obj_id)
        if obj:
            attr = obj.Attributes.Duplicate()
            attr.WireDensity = -1
            _doc.Objects.ModifyAttributes(obj_id, attr, True)

        _apply_attrs(obj_id, args)
        _doc.Views.Redraw()
        return str(obj_id)
    return {"ok": True, "id": _run_on_ui(action)}


def create_text_dot(args):
    def action():
        pt = rg.Point3d(args["x"], args["y"], args["z"])
        dot = rg.TextDot(str(args["text"]), pt)
        obj_id = _doc.Objects.AddTextDot(dot)
        _apply_attrs(obj_id, args)
        _doc.Views.Redraw()
        return str(obj_id)
    return {"ok": True, "id": _run_on_ui(action)}


def create_text(args):
    """Flat annotation text lying in the XY plane at (x, y, z), height in model units."""
    def action():
        plane = rg.Plane(rg.Point3d(args["x"], args["y"], args["z"]), rg.Vector3d.ZAxis)
        te = rg.TextEntity.Create(str(args["text"]), plane, _doc.DimStyles.Current, False, 0, 0)
        te.TextHeight = float(args.get("height", 0.25))
        align = args.get("align", "left")
        te.TextHorizontalAlignment = {"left": rdo.TextHorizontalAlignment.Left,
                                      "center": rdo.TextHorizontalAlignment.Center,
                                      "right": rdo.TextHorizontalAlignment.Right}[align]
        obj_id = _doc.Objects.AddText(te)
        _apply_attrs(obj_id, args)
        _doc.Views.Redraw()
        return str(obj_id)
    return {"ok": True, "id": _run_on_ui(action)}


def set_layer_wire_density(args):
    """Set WireDensity for all objects on a layer. -1 = hide mesh edges, 0 = default."""
    def action():
        name    = args["name"]
        density = int(args.get("density", -1))
        idx = _find_layer(name)
        if idx < 0:
            return 0
        count = 0
        for obj in _doc.Objects:
            if obj.IsDeleted or obj.Attributes.LayerIndex != idx:
                continue
            attr = obj.Attributes.Duplicate()
            attr.WireDensity = density
            _doc.Objects.ModifyAttributes(obj.Id, attr, True)
            count += 1
        _doc.Views.Redraw()
        return count
    return {"ok": True, "updated": _run_on_ui(action)}


def delete_layer_objects(args):
    def action():
        name = args["name"]
        idx  = _find_layer(name)
        if idx < 0:
            return 0
        ids = [obj.Id for obj in _doc.Objects
               if not obj.IsDeleted and obj.Attributes.LayerIndex == idx]
        for oid in ids:
            _doc.Objects.Delete(oid, True)
        _doc.Views.Redraw()
        return len(ids)
    return {"ok": True, "deleted": _run_on_ui(action)}


def set_layer_visible(args):
    def action():
        name    = args["name"]
        visible = bool(args["visible"])
        idx = _find_layer(name)
        if idx < 0:
            return f"Layer not found: {name}"
        layer = _doc.Layers[idx]
        layer.IsVisible = visible
        _doc.Views.Redraw()
        return f"Layer '{name}' visible={visible}"
    return {"ok": True, "result": _run_on_ui(action)}


def get_scene_info(args):
    def action():
        layers = {}
        for i in range(_doc.Layers.Count):
            layer = _doc.Layers[i]
            if not layer.IsDeleted:
                layers[i] = {
                    "name": str(layer.FullPath),
                    "visible": bool(layer.IsVisible),
                    "color": [layer.Color.R, layer.Color.G, layer.Color.B],
                    "objects": []
                }

        for obj in _doc.Objects:
            if obj.IsDeleted:
                continue
            li = obj.Attributes.LayerIndex
            bb = obj.Geometry.GetBoundingBox(True)
            entry = {
                "id": str(obj.Id),
                "type": str(obj.ObjectType),
                "bbox_min": [round(bb.Min.X, 3), round(bb.Min.Y, 3), round(bb.Min.Z, 3)],
                "bbox_max": [round(bb.Max.X, 3), round(bb.Max.Y, 3), round(bb.Max.Z, 3)],
            }
            if li in layers:
                layers[li]["objects"].append(entry)

        layer_list = list(layers.values())
        total_objects = sum(len(l["objects"]) for l in layer_list)
        return {"layers": layer_list, "total_objects": total_objects}
    return {"ok": True, "scene": _run_on_ui(action)}


def delete_all_objects(args):
    def action():
        ids = [obj.Id for obj in _doc.Objects]
        for obj_id in ids:
            _doc.Objects.Delete(obj_id, True)
        _doc.Views.Redraw()
        return len(ids)
    return {"ok": True, "deleted": _run_on_ui(action)}


def execute(payload):
    tool = payload.get("tool")
    args = payload.get("args", {})
    if tool == "create_sphere":
        return create_sphere(args)
    if tool == "create_box":
        return create_box(args)
    if tool == "create_triangle":
        return create_triangle(args)
    if tool == "create_layer":
        return create_layer(args)
    if tool == "set_object_color":
        return set_object_color(args)
    if tool == "set_layer_objects_color":
        return set_layer_objects_color(args)
    if tool == "delete_object":
        return delete_object(args)
    if tool == "delete_all_objects":
        return delete_all_objects(args)
    if tool == "create_polyline":
        return create_polyline(args)
    if tool == "curve_to_surface":
        return curve_to_surface(args)
    if tool == "curve_to_extrusion":
        return curve_to_extrusion(args)
    if tool == "delete_empty_layers":
        return delete_empty_layers(args)
    if tool == "zoom_extents":
        return zoom_extents(args)
    if tool == "create_text_dot":
        return create_text_dot(args)
    if tool == "create_heatmap_plane":
        return create_heatmap_plane(args)
    if tool == "set_layer_visible":
        return set_layer_visible(args)
    if tool == "delete_layer_objects":
        return delete_layer_objects(args)
    if tool == "set_layer_wire_density":
        return set_layer_wire_density(args)
    if tool == "get_scene_info":
        return get_scene_info(args)
    if tool == "create_planar_surface":
        return create_planar_surface(args)
    if tool == "create_text":
        return create_text(args)
    return {"ok": False, "error": "Unknown tool: " + str(tool)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/shutdown":           # lets a newly started bridge replace this one
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"stopping")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        elif self.path == "/tools":
            body = json.dumps(TOOLS).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Rhino bridge is running")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            result = execute(payload)
        except Exception as ex:
            result = {"ok": False, "error": str(ex)}
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception:
            pass


def start_server():
    # stop a bridge started earlier in this Rhino session (versions that know /shutdown)
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/shutdown", timeout=2).read()
        time.sleep(1.0)
    except Exception:
        pass
    try:
        server = HTTPServer(("127.0.0.1", 8765), Handler)
    except OSError:
        print("Port 8765 is still used by an older bridge. Close and reopen Rhino, then run this script again.")
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Bridge v2 started. Rhino bridge running at http://127.0.0.1:8765")


start_server()
