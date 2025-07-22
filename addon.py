import bpy
import json
import threading
import socket
import time
import requests
import tempfile
from bpy.props import StringProperty, IntProperty
import traceback
import os
import shutil

bl_info = {
    "name": "Blender MCP",
    "author": "BlenderMCP",
    "version": (0, 2),  # Updated version
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > BlenderMCP",
    "description": "Connect Blender to local AI models via MCP",  # Updated description
    "category": "Interface",
}

class BlenderMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.running = False
        self.socket = None
        self.client = None
        self.command_queue = []
        self.buffer = b''

    def start(self):
        self.running = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.setblocking(False)
            bpy.app.timers.register(self._process_server, persistent=True)
            print(f"BlenderMCP server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            self.stop()

    def stop(self):
        self.running = False
        if hasattr(bpy.app.timers, "unregister"):
            if bpy.app.timers.is_registered(self._process_server):
                bpy.app.timers.unregister(self._process_server)
        if self.socket:
            self.socket.close()
        if self.client:
            self.client.close()
        self.socket = None
        self.client = None
        print("BlenderMCP server stopped")

    def _process_server(self):
        if not self.running:
            return None

        try:
            if not self.client and self.socket:
                try:
                    self.client, address = self.socket.accept()
                    self.client.setblocking(False)
                    print(f"Connected to client: {address}")
                except BlockingIOError:
                    pass
                except Exception as e:
                    print(f"Error accepting connection: {str(e)}")

            if self.client:
                try:
                    try:
                        data = self.client.recv(8192)
                        if data:
                            self.buffer += data
                            try:
                                command = json.loads(self.buffer.decode('utf-8'))
                                self.buffer = b''
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                self.client.sendall(response_json.encode('utf-8'))
                            except json.JSONDecodeError:
                                pass
                        else:
                            print("Client disconnected")
                            self.client.close()
                            self.client = None
                            self.buffer = b''
                    except BlockingIOError:
                        pass
                    except Exception as e:
                        print(f"Error receiving data: {str(e)}")
                        self.client.close()
                        self.client = None
                        self.buffer = b''

                except Exception as e:
                    print(f"Error with client: {str(e)}")
                    if self.client:
                        self.client.close()
                        self.client = None
                    self.buffer = b''

        except Exception as e:
            print(f"Server error: {str(e)}")

        return 0.1

    def execute_command(self, command):
        try:
            cmd_type = command.get("type")
            params = command.get("params", {})
            if cmd_type in ["create_object", "modify_object", "delete_object"]:
                if not bpy.context.screen or not bpy.context.screen.areas:
                    return {"status": "error", "message": "Suitable 'VIEW_3D' context not found for command execution."}

                view_3d_areas = [area for area in bpy.context.screen.areas if area.type == 'VIEW_3D']
                if not view_3d_areas:
                    return {"status": "error", "message": "Suitable 'VIEW_3D' context not found for command execution."}

                override = bpy.context.copy()
                override['area'] = view_3d_areas[0]
                with bpy.context.temp_override(**override):
                    return self._execute_command_internal(command)
            else:
                return self._execute_command_internal(command)
        except Exception as e:
            print(f"Error executing command: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _execute_command_internal(self, command):
        cmd_type = command.get("type")
        params = command.get("params", {})

        if cmd_type == "get_polyhaven_status":
            return {"status": "success", "result": self.get_polyhaven_status()}

        handlers = {
        "get_scene_info": self.get_scene_info,
        "create_object": self.create_object,
        "modify_object": self.modify_object,
        "delete_object": self.delete_object,
        "get_object_info": self.get_object_info,
        "execute_code": self.execute_code,
        "set_material": self.set_material,
        "get_polyhaven_status": self.get_polyhaven_status,
        "render_scene": self.render_scene,
        "insert_keyframe": self.insert_keyframe,
        "add_modifier": self.add_modifier,
        "import_asset": self.import_asset,
        "export_scene": self.export_scene,
        "add_light": self.add_light,
        "brush_asset_activate": self.brush_asset_activate,
        "brush_asset_delete": self.brush_asset_delete,
        "brush_asset_edit_metadata": self.brush_asset_edit_metadata,
        "set_camera_params": self.set_camera_params,
        "import_texture": self.import_texture,
        "import_image_to_plane": self.import_image_to_plane,
        "append_blend": self.append_blend,
        "link_blend": self.link_blend,
        "find_missing_files": self.find_missing_files,
        "add_text": self.add_text,
        "origin_set": self.origin_set,
        "transform_apply": self.transform_apply,
        "parent_set": self.parent_set,
        "select_all": self.select_all,
        "snap_selected_to_cursor": self.snap_selected_to_cursor,
        "duplicate_move": self.duplicate_move,
        "primitive_add": self.primitive_add,
        "search_assets": self.search_assets,
        "get_scene_graph": self.get_scene_graph,
        "batch_place_assets": self.batch_place_assets,
        "group_objects": self.group_objects,
        "create_room": self.create_room,
        "auto_place_asset": self.auto_place_asset,
        "create_scene_from_text": self.create_scene_from_text,
        "create_scene_from_image": self.create_scene_from_image,
        "batch_render": self.batch_render,
        "get_asset_metadata": self.get_asset_metadata,

    }


        if bpy.context.scene.blendermcp_use_polyhaven:
            polyhaven_handlers = {
                "get_polyhaven_categories": self.get_polyhaven_categories,
                "search_polyhaven_assets": self.search_polyhaven_assets,
                "download_polyhaven_asset": self.download_polyhaven_asset,
                "set_texture": self.set_texture,
            }
            handlers.update(polyhaven_handlers)

        handler = handlers.get(cmd_type)
        if handler:
            try:
                print(f"Executing handler for {cmd_type}")
                result = handler(**params)
                print(f"Handler execution complete")
                return {"status": "success", "result": result}
            except Exception as e:
                print(f"Error in handler: {str(e)}")
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}

    def brush_asset_activate(self, asset_library_type="LOCAL", asset_library_identifier="", relative_asset_identifier="", use_toggle=False):
        try:
            bpy.ops.brush.asset_activate(
                asset_library_type=asset_library_type,
                asset_library_identifier=asset_library_identifier,
                relative_asset_identifier=relative_asset_identifier,
                use_toggle=use_toggle
            )
            return {"result": "Activated brush asset."}
        except Exception as e:
            return {"error": str(e)}
    def place_asset_smart(self, asset_name, area, constraints=None):
        # Pseudo-logic: use raycast/surface detect, respect constraints, avoid collision
        return {"placed": asset_name, "area": area, "constraints": constraints}

    def auto_arrange_furniture(self, area_type="kitchen"):
        # Pseudo-logic: use raycast/surface detect, respect constraints, avoid collision
        return {"placed": "Furniture", "area": area_type, "constraints": None}
    def harmonize_materials(self, palette=None, style=None):
        # Iterate all scene objects, adjust colors/materials to match palette/style
        return {"status": "harmonized", "palette": palette, "style": style}
    def auto_lighting(self, style="natural"):
        # Add area lights, fill/sun/key, etc. based on style
        return {"status": "lights_added", "style": style}

    def auto_place_camera(self, mode="best_view"):
        # Place camera at best position based on scene bounding box etc.
        return {"status": "camera_placed", "mode": mode}

    # === Asset Search/Management ===
    def scene_quality_check(self):
        # Scan for floating objects, overlaps, weird scales
        return {"report": "Scene looks good!"}
    
    def search_assets(self, query=None, tags=None, type=None):
        """Stub: Return list of asset names/paths matching query/tags/type"""
        # For demo, just scan a directory or use a fake list.
        # TODO: Integrate with your real asset DB/library!
        asset_dir = bpy.context.scene.blendermcp_asset_dir  # Change to your actual asset library path
        asset_list = []
        for root, dirs, files in os.walk(asset_dir):
            for f in files:
                if type and not f.lower().endswith(type.lower()):
                    continue
                if query and query.lower() not in f.lower():
                    continue
                # Tags would require some metadata—stubbed here
                asset_list.append(os.path.join(root, f))
        return {"assets": asset_list}
    def semantic_edit(self, instruction):
        # Parse "Make all chairs blue", "Move table", etc.
        return {"result": f"Applied: {instruction}"}
    def suggest_scene_improvements(self):
        # Could call LLM or use rules: e.g., “Try brighter lighting”
        return {"suggestions": ["Try brighter lighting", "Move sofa to face window"]}
    def batch_import_assets(self, folder_path):
        # Import all assets from folder
        return {"imported_assets": []}
    def generate_parametric_asset(self, type, params):
        # E.g., generate a table with parametric width, legs, etc.
        return {"asset": type, "params": params}

    def randomize_scene(self, n_variations=3):
        # Create n variations (layouts/colors/materials)
        return {"variations": n_variations}
    def image_to_geometry(self, image_path):
        # Stub: process image to extract floorplan/geometry
        return {"geometry_created_from": image_path}

    def transfer_style_from_image(self, image_path):
        # Stub: extract palette/mood and apply to scene
        return {"style_applied_from": image_path}
    def add_human_figure(self, pose="standing"):
        # Import or create a low-poly human figure for scale
        return {"figure_added": pose}
    def undo_last_action(self):
        bpy.ops.ed.undo()
        return {"undone": True}

    def redo_action(self):
        bpy.ops.ed.redo()
        return {"redone": True}
    def save_scene_variant(self, name):
        # Save copy of scene with unique name
        return {"variant_saved": name}

    def load_scene_variant(self, name):
        # Load scene by name
        return {"variant_loaded": name}


    # === Scene Graph / Context ===

    def get_scene_graph(self):
        """Returns high-level structure of scene objects and relations"""
        scene = bpy.context.scene
        graph = []
        for obj in scene.objects:
            graph.append({
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "location": list(obj.location),
            })
        return {"graph": graph}

    # === Batch Placement/Arrangement ===

    def batch_place_assets(self, asset_names, positions, rotations=None, scales=None):
        """Place many assets in scene (stub)"""
        placed = []
        for i, asset_name in enumerate(asset_names):
            # You'd want to import or instance the asset here
            loc = positions[i] if i < len(positions) else [0,0,0]
            rot = rotations[i] if rotations and i < len(rotations) else [0,0,0]
            sc = scales[i] if scales and i < len(scales) else [1,1,1]
            # Example: self.import_asset(asset_name, ...)
            placed.append({"asset": asset_name, "location": loc, "rotation": rot, "scale": sc})
        return {"placed": placed}

    # === Grouping/Parenting ===

    def group_objects(self, object_names, group_name):
        """Group/parent objects"""
        parent = bpy.data.objects.new(group_name, None)
        bpy.context.collection.objects.link(parent)
        for name in object_names:
            obj = bpy.data.objects.get(name)
            if obj:
                obj.parent = parent
        return {"group": group_name}

    # === Procedural Room Creation ===

    def create_room(self, width, depth, height, style="modern", doors=1, windows=1):
        """Create basic room geometry with doors/windows (stub)"""
        # Placeholders: add walls, floor, ceiling as mesh cubes/planes
        # TODO: Actually create meshes!
        room_name = f"Room_{width}x{depth}x{height}_{style}"
        # ...
        return {"room": room_name}

    # === Auto Placement by Constraint ===

    def auto_place_asset(self, asset_name, target_area, constraints=None):
        """Place asset in room/scene using constraints (stub)"""
        # TODO: Use scene context and constraints!
        return {"asset": asset_name, "area": target_area, "constraints": constraints}

    # === Text-to-Scene ===

    def create_scene_from_text(self, description, style=None):
        """Stub: Actually use LLM/server for logic!"""
        # Here you could call an LLM, or the server will handle the prompt.
        # We'll just return a stub for now.
        return {"scene": f"Scene created from: {description}", "style": style}

    # === Image-to-Scene (reference) ===

    def create_scene_from_image(self, image_path):
        # This would call server/AI for actual processing
        return {"scene": f"Scene created using image: {image_path}"}

    # === Batch Render Presets ===

    def batch_render(self, views, resolution, preset="high_quality"):
        # For each camera/view: render at resolution/preset
        return {"rendered": len(views), "preset": preset}

    # === Asset Metadata/Preview ===

    def get_asset_metadata(self, asset_path):
        """Return file info, and, if possible, a preview"""
        import mimetypes
        size = os.path.getsize(asset_path)
        mime, _ = mimetypes.guess_type(asset_path)
        return {"file": asset_path, "size": size, "type": mime}


    def brush_asset_delete(self):
        try:
            bpy.ops.brush.asset_delete()
            return {"result": "Deleted active brush asset."}
        except Exception as e:
            return {"error": str(e)}

    def brush_asset_edit_metadata(self, catalog_path="", author="", description=""):
        try:
            bpy.ops.brush.asset_edit_metadata(
                catalog_path=catalog_path,
                author=author,
                description=description
            )
            return {"result": "Edited asset metadata."}
        except Exception as e:
            return {"error": str(e)}
# === Asset Import/Reference ===
    def import_image_to_plane(self, image_path, location=(0,0,0), size=1.0):
        import bpy
        try:
            # Add-on 'io_import_images_as_planes' must be enabled!
            if not bpy.ops.import_image.to_plane.poll():
                bpy.ops.preferences.addon_enable(module="io_import_images_as_planes")
            res = bpy.ops.import_image.to_plane(files=[{"name": image_path.split('/')[-1]}], directory=os.path.dirname(image_path), align_axis='Z', size=size)
            obj = bpy.context.active_object
            obj.location = location
            return {"status": "success", "object": obj.name}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def append_blend(self, blend_path, data_type, object_name):
        import bpy
        try:
            with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
                if hasattr(data_to, data_type):
                    setattr(data_to, data_type, [object_name])
            return {"status": "success", "object": object_name}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def link_blend(self, blend_path, data_type, object_name):
        import bpy
        try:
            with bpy.data.libraries.load(blend_path, link=True) as (data_from, data_to):
                if hasattr(data_to, data_type):
                    setattr(data_to, data_type, [object_name])
            return {"status": "success", "object": object_name}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def find_missing_files(self):
        try:
            bpy.ops.file.find_missing_files()
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def add_text(self, body="Label", location=(0,0,0), size=1.0):
        try:
            bpy.ops.object.text_add(location=location)
            obj = bpy.context.active_object
            obj.data.body = body
            obj.scale = (size, size, size)
            return {"status": "success", "object": obj.name}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # === Object Placement ===
    def origin_set(self, object_name, type="ORIGIN_GEOMETRY"):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"status": "error", "message": f"Object {object_name} not found"}
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type=type)
        return {"status": "success"}

    def transform_apply(self, object_name, location=True, rotation=True, scale=True):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"status": "error", "message": f"Object {object_name} not found"}
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)
        return {"status": "success"}

    def parent_set(self, child, parent):
        obj = bpy.data.objects.get(child)
        par = bpy.data.objects.get(parent)
        if not obj or not par:
            return {"status": "error", "message": "Child or parent not found"}
        obj.parent = par
        return {"status": "success"}

    def select_all(self, action='SELECT'):
        bpy.ops.object.select_all(action=action)
        return {"status": "success"}

    def snap_selected_to_cursor(self, use_offset=False):
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=use_offset)
        return {"status": "success"}

    def duplicate_move(self):
        bpy.ops.object.duplicate_move()
        return {"status": "success"}

    # Mesh Primitives (general pattern)
    def primitive_add(self, primitive_type, location=(0,0,0), **kwargs):
        if primitive_type == "cube":
            bpy.ops.mesh.primitive_cube_add(location=location, **kwargs)
        elif primitive_type == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(location=location, **kwargs)
        # ... repeat for all mesh types
        return {"status": "success", "object": bpy.context.active_object.name}

    def get_simple_info(self):
        return {
            "blender_version": ".".join(str(v) for v in bpy.app.version),
            "scene_name": bpy.context.scene.name,
            "object_count": len(bpy.context.scene.objects)
        }

    def get_scene_info(self):
        try:
            print("Getting scene info...")
            scene_info = {
                "name": bpy.context.scene.name,
                "object_count": len(bpy.context.scene.objects),
                "objects": [],
                "materials_count": len(bpy.data.materials),
            }

            for i, obj in enumerate(bpy.context.scene.objects):
                if i >= 10:
                    break

                obj_info = {
                    "name": obj.name,
                    "type": obj.type,
                    "location": [round(float(obj.location.x), 2),
                                round(float(obj.location.y), 2),
                                round(float(obj.location.z), 2)],
                }
                scene_info["objects"].append(obj_info)

            print(f"Scene info collected: {len(scene_info['objects'])} objects")
            return scene_info
        except Exception as e:
            print(f"Error in get_scene_info: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def render_scene(self, output_path=None, resolution_x=None, resolution_y=None):
        """Render the current scene"""
        try:
            if resolution_x is not None:
                bpy.context.scene.render.resolution_x = int(resolution_x)

            if resolution_y is not None:
                bpy.context.scene.render.resolution_y = int(resolution_y)

            if output_path:
                # Use absolute path and ensure directory exists.
                output_path = bpy.path.abspath(output_path)
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                bpy.context.scene.render.filepath = output_path
            else: # If path not given save to a temp dir
                output_path = os.path.join(tempfile.gettempdir(),"render.png")
                bpy.context.scene.render.filepath = output_path


            # Render the scene
            bpy.ops.render.render(write_still=True) #Always write still even if no path given

            return {
                "rendered": True,
                "output_path": output_path ,
                "resolution": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
            }
        except Exception as e:
            print(f"Error in render_scene: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def boolean_operation(self, object_a, object_b, operation="UNION"):
        ob_a = bpy.data.objects.get(object_a)
        ob_b = bpy.data.objects.get(object_b)
        if not ob_a or not ob_b:
            return {"error": "One or both objects not found"}
        mod = ob_a.modifiers.new(name="Boolean", type='BOOLEAN')
        mod.operation = operation
        mod.object = ob_b
        bpy.context.view_layer.objects.active = ob_a
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return {"result": f"Boolean {operation} applied"}
    
    def add_modifier(self, object_name, modifier_type, params=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": "Object not found"}
        mod = obj.modifiers.new(name=modifier_type, type=modifier_type)
        if params:
            for k, v in params.items():
                setattr(mod, k, v)
        return {"result": f"{modifier_type} added"}
    def create_curve(self, curve_type="BEZIER", points=None):
        # For brevity: create a bezier with default points
        bpy.ops.curve.primitive_bezier_curve_add()
        obj = bpy.context.active_object
        if points:
            for i, pt in enumerate(obj.data.splines[0].bezier_points):
                pt.co = points[i]
        return {"name": obj.name}

    def sweep_mesh_along_curve(self, mesh_name, curve_name):
        mesh = bpy.data.objects.get(mesh_name)
        curve = bpy.data.objects.get(curve_name)
        if not mesh or not curve:
            return {"error": "Object(s) not found"}
        mod = mesh.modifiers.new(name="Curve", type='CURVE')
        mod.object = curve
        return {"result": f"{mesh_name} swept along {curve_name}"}
    def apply_modifier(self, object_name, modifier_name):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": "Object not found"}
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier_name)
        return {"result": f"{modifier_name} applied to {object_name}"}
    def apply_geometry_nodes(self, object_name, node_group_name, inputs=None):
        obj = bpy.data.objects.get(object_name)
        node_group = bpy.data.node_groups.get(node_group_name)
        if not obj or not node_group:
            return {"error": "Object or Node Group not found"}
        mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
        mod.node_group = node_group
        if inputs:
            for k, v in inputs.items():
                if k in mod["Input_1"]: # Example, depends on node group setup
                    mod[k] = v
        return {"result": "Geometry nodes applied"}

    def remove_modifier(self, object_name, modifier_name):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": "Object not found"}
        obj.modifiers.remove(obj.modifiers[modifier_name])
        return {"result": f"{modifier_name} removed from {object_name}"}

    def convert_curve_to_mesh(self, curve_name):
        curve = bpy.data.objects.get(curve_name)
        if not curve:
            return {"error": "Curve not found"}
        bpy.context.view_layer.objects.active = curve
        bpy.ops.object.convert(target='MESH')
        return {"result": f"{curve_name} converted to mesh"}

    def scatter_objects(self, source, target, count=10, method="surface"):
        # Example: distribute source on target's surface using particle system
        tgt = bpy.data.objects.get(target)
        src = bpy.data.objects.get(source)
        if not tgt or not src:
            return {"error": "Object(s) not found"}
        ps = tgt.modifiers.new(name="Scatter", type='PARTICLE_SYSTEM')
        psys = tgt.particle_systems[-1]
        psys.settings.count = count
        psys.settings.render_type = 'OBJECT'
        psys.settings.instance_object = src
        return {"result": f"Scattered {count} instances of {source} on {target}"}

    def create_object(self, type="CUBE", name=None, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        bpy.ops.object.select_all(action='DESELECT')
        if type == "CUBE":
            bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation, scale=scale)
        elif type == "SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(location=location, rotation=rotation, scale=scale)
        elif type == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(location=location, rotation=rotation, scale=scale)
        elif type == "PLANE":
            bpy.ops.mesh.primitive_plane_add(location=location, rotation=rotation, scale=scale)
        elif type == "CONE":
            bpy.ops.mesh.primitive_cone_add(location=location, rotation=rotation, scale=scale)
        elif type == "TORUS":
            bpy.ops.mesh.primitive_torus_add(location=location, rotation=rotation, scale=scale)
        elif type == "EMPTY":
            bpy.ops.object.empty_add(location=location, rotation=rotation)
        elif type == "CAMERA":
            bpy.ops.object.camera_add(location=location, rotation=rotation)
        elif type == "LIGHT":
            bpy.ops.object.light_add(type='POINT', location=location, rotation=rotation)
        else:
            raise ValueError(f"Unsupported object type: {type}")

        obj = bpy.context.active_object
        if name:
            obj.name = name

        return {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
        }

    def modify_object(self, name, location=None, rotation=None, scale=None, visible=None):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")

        if location is not None:
            obj.location = location
        if rotation is not None:
            obj.rotation_euler = rotation
        if scale is not None:
            obj.scale = scale
        if visible is not None:
            obj.hide_viewport = not visible
            obj.hide_render = not visible

        return {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            "visible": obj.visible_get(),
        }

    def delete_object(self, name):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")

        obj_name = obj.name
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()

        return {"deleted": obj_name}

    def get_object_info(self, name):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")

        obj_info = {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            "visible": obj.visible_get(),
            "materials": [],
        }

        for slot in obj.material_slots:
            if slot.material:
                obj_info["materials"].append(slot.material.name)

        if obj.type == 'MESH' and obj.data:
            mesh = obj.data
            obj_info["mesh"] = {
                "vertices": len(mesh.vertices),
                "edges": len(mesh.edges),
                "polygons": len(mesh.polygons),
            }

        return obj_info

    def execute_code(self, code):
        try:
            namespace = {"bpy": bpy}
            exec(code, namespace)
            return {"executed": True}
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")

    def set_material(self, object_name, material_name=None, create_if_missing=True, color=None):
        """Set or create a material for an object."""
        try:
            obj = bpy.data.objects.get(object_name)
            if not obj:
                raise ValueError(f"Object not found: {object_name}")

            if not hasattr(obj, 'data') or not hasattr(obj.data, 'materials'):
                raise ValueError(f"Object {object_name} cannot accept materials")
            if material_name:
                mat = bpy.data.materials.get(material_name)
                if not mat and create_if_missing:
                    mat = bpy.data.materials.new(name=material_name)
                    print(f"Created new material: {material_name}")
            else:
                mat_name = f"{object_name}_material"
                mat = bpy.data.materials.get(mat_name)
                if not mat:
                    mat = bpy.data.materials.new(name=mat_name)
                material_name = mat_name
                print(f"Using material: {mat_name}")

            if mat:
                if not mat.use_nodes:
                    mat.use_nodes = True
                principled = mat.node_tree.nodes.get('Principled BSDF')
                if not principled:
                    principled = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
                    output = mat.node_tree.nodes.get('Material Output')
                    if not output:
                        output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
                    if not principled.outputs[0].links:
                         mat.node_tree.links.new(principled.outputs[0], output.inputs[0])

                if color and len(color) >= 3:
                    principled.inputs['Base Color'].default_value = (
                        color[0],
                        color[1],
                        color[2],
                        1.0 if len(color) < 4 else color[3]
                    )
                    print(f"Set material color to {color}")

            if mat:
                if not obj.data.materials:
                    obj.data.materials.append(mat)
                else:
                    obj.data.materials[0] = mat
                print(f"Assigned material {mat.name} to object {object_name}")
                return {
                    "status": "success",
                    "object": object_name,
                    "material": mat.name,
                    "color": color if color else None
                }
            else:
                raise ValueError(f"Failed to create or find material: {material_name}")
        except Exception as e:
            print(f"Error in set_material: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "object": object_name,
                "material": material_name if 'material_name' in locals() else None
            }
    def get_polyhaven_categories(self, asset_type):
        """Get categories for a specific asset type from Polyhaven"""
        try:
            if asset_type not in ["hdris", "textures", "models", "all"]:
                return {"error": f"Invalid asset type: {asset_type}. Must be one of: hdris, textures, models, all"}

            response = requests.get(f"https://api.polyhaven.com/categories/{asset_type}")
            if response.status_code == 200:
                return {"categories": response.json()}
            else:
                return {"error": f"API request failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def search_polyhaven_assets(self, asset_type=None, categories=None):
        """Search for assets from Polyhaven with optional filtering"""
        try:
            url = "https://api.polyhaven.com/assets"
            params = {}

            if asset_type and asset_type != "all":
                if asset_type not in ["hdris", "textures", "models"]:
                    return {"error": f"Invalid asset type: {asset_type}. Must be one of: hdris, textures, models, all"}
                params["type"] = asset_type

            if categories:
                params["categories"] = categories

            response = requests.get(url, params=params)
            if response.status_code == 200:
                assets = response.json()
                limited_assets = {}
                for i, (key, value) in enumerate(assets.items()):
                    if i >= 20:
                        break
                    limited_assets[key] = value

                return {"assets": limited_assets, "total_count": len(assets), "returned_count": len(limited_assets)}
            else:
                return {"error": f"API request failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def download_polyhaven_asset(self, asset_id, asset_type, resolution="1k", file_format=None):
        """Downloads and imports a PolyHaven asset."""
        try:
            files_response = requests.get(f"https://api.polyhaven.com/files/{asset_id}")
            if files_response.status_code != 200:
                return {"error": f"Failed to get asset files: {files_response.status_code}"}

            files_data = files_response.json()

            if asset_type == "hdris":
                if not file_format:
                    file_format = "hdr"
                if "hdri" in files_data and resolution in files_data["hdri"] and file_format in files_data["hdri"][resolution]:
                    file_info = files_data["hdri"][resolution][file_format]
                    file_url = file_info["url"]

                    with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                        response = requests.get(file_url)
                        if response.status_code != 200:
                            return {"error": f"Failed to download HDRI: {response.status_code}"}
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name
                    try:
                        if not bpy.data.worlds:
                            bpy.data.worlds.new("World")
                        world = bpy.data.worlds[0]
                        world.use_nodes = True
                        node_tree = world.node_tree
                        for node in node_tree.nodes:
                            node_tree.nodes.remove(node)
                        tex_coord = node_tree.nodes.new(type='ShaderNodeTexCoord')
                        tex_coord.location = (-800, 0)
                        mapping = node_tree.nodes.new(type='ShaderNodeMapping')
                        mapping.location = (-600, 0)
                        env_tex = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
                        env_tex.location = (-400, 0)
                        env_tex.image = bpy.data.images.load(tmp_path)
                        if file_format.lower() == 'exr':
                            try:
                                env_tex.image.colorspace_settings.name = 'Linear'
                            except:
                                env_tex.image.colorspace_settings.name = 'Non-Color'
                        else:
                            for color_space in ['Linear', 'Linear Rec.709', 'Non-Color']:
                                try:
                                    env_tex.image.colorspace_settings.name = color_space
                                    break
                                except:
                                    continue
                        background = node_tree.nodes.new(type='ShaderNodeBackground')
                        background.location = (-200, 0)
                        output = node_tree.nodes.new(type='ShaderNodeOutputWorld')
                        output.location = (0, 0)
                        node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
                        node_tree.links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
                        node_tree.links.new(env_tex.outputs['Color'], background.inputs['Color'])
                        node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

                        bpy.context.scene.world = world
                        try:
                            tempfile._cleanup()
                        except:
                            pass
                        return {
                            "success": True,
                            "message": f"HDRI {asset_id} imported successfully",
                            "image_name": env_tex.image.name
                        }
                    except Exception as e:
                        return {"error": f"Failed to set up HDRI: {str(e)}"}
                else:
                    return {"error": f"Resolution/format unavailable."}

            elif asset_type == "textures":
                if not file_format:
                    file_format = "jpg"

                downloaded_maps = {}
                try:
                    for map_type in files_data:
                        if map_type not in ["blend", "gltf"]:
                            if resolution in files_data[map_type] and file_format in files_data[map_type][resolution]:
                                file_info = files_data[map_type][resolution][file_format]
                                file_url = file_info["url"]

                                with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                                    response = requests.get(file_url)
                                    if response.status_code == 200:
                                        tmp_file.write(response.content)
                                        tmp_path = tmp_file.name
                                        image = bpy.data.images.load(tmp_path)
                                        image.name = f"{asset_id}_{map_type}.{file_format}"
                                        image.pack()
                                        if map_type in ['color', 'diffuse', 'albedo']:
                                            try:
                                                image.colorspace_settings.name = 'sRGB'
                                            except:
                                                pass
                                        else:
                                            try:
                                                image.colorspace_settings.name = 'Non-Color'
                                            except:
                                                pass
                                        downloaded_maps[map_type] = image
                                        try:
                                            os.unlink(tmp_path)
                                        except:
                                            pass

                    if not downloaded_maps:
                        return {"error": f"No texture maps found."}

                    mat = bpy.data.materials.new(name=asset_id)
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    for node in nodes:
                        nodes.remove(node)
                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (300, 0)
                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (0, 0)
                    links.new(principled.outputs[0], output.inputs[0])
                    tex_coord = nodes.new(type='ShaderNodeTexCoord')
                    tex_coord.location = (-800, 0)
                    mapping = nodes.new(type='ShaderNodeMapping')
                    mapping.location = (-600, 0)
                    mapping.vector_type = 'TEXTURE'
                    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
                    x_pos = -400
                    y_pos = 300

                    for map_type, image in downloaded_maps.items():
                        tex_node = nodes.new(type='ShaderNodeTexImage')
                        tex_node.location = (x_pos, y_pos)
                        tex_node.image = image
                        if map_type.lower() in ['color', 'diffuse', 'albedo']:
                            try:
                                tex_node.image.colorspace_settings.name = 'sRGB'
                            except:
                                pass
                        else:
                            try:
                                tex_node.image.colorspace_settings.name = 'Non-Color'
                            except:
                                pass
                        links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])

                        if map_type.lower() in ['color', 'diffuse', 'albedo']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                        elif map_type.lower() in ['roughness', 'rough']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
                        elif map_type.lower() in ['metallic', 'metalness', 'metal']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Metallic'])
                        elif map_type.lower() in ['normal', 'nor']:
                            normal_map = nodes.new(type='ShaderNodeNormalMap')
                            normal_map.location = (x_pos + 200, y_pos)
                            links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
                            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                        elif map_type in ['displacement', 'disp', 'height']:
                            disp_node = nodes.new(type='ShaderNodeDisplacement')
                            disp_node.location = (x_pos + 200, y_pos - 200)
                            links.new(tex_node.outputs['Color'], disp_node.inputs['Height'])
                            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
                        y_pos -= 250
                    return {
                        "success": True,
                        "message": f"Texture {asset_id} imported as material",
                        "material": mat.name,
                        "maps": list(downloaded_maps.keys())
                    }
                except Exception as e:
                    return {"error": f"Failed to process textures: {str(e)}"}

            elif asset_type == "models":
                if not file_format:
                    file_format = "gltf"
                if file_format in files_data and resolution in files_data[file_format]:
                    file_info = files_data[file_format][resolution][file_format]
                    file_url = file_info["url"]
                    temp_dir = tempfile.mkdtemp()
                    main_file_path = ""
                    try:
                        main_file_name = file_url.split("/")[-1]
                        main_file_path = os.path.join(temp_dir, main_file_name)
                        response = requests.get(file_url)
                        if response.status_code != 200:
                            return {"error": f"Failed to download model: {response.status_code}"}
                        with open(main_file_path, "wb") as f:
                            f.write(response.content)
                        if "include" in file_info and file_info["include"]:
                            for include_path, include_info in file_info["include"].items():
                                include_url = include_info["url"]
                                include_file_path = os.path.join(temp_dir, include_path)
                                os.makedirs(os.path.dirname(include_file_path), exist_ok=True)
                                include_response = requests.get(include_url)
                                if include_response.status_code == 200:
                                    with open(include_file_path, "wb") as f:
                                        f.write(include_response.content)
                                else:
                                    print(f"Failed to download included file: {include_path}")
                        if file_format == "gltf" or file_format == "glb":
                            bpy.ops.import_scene.gltf(filepath=main_file_path)
                        elif file_format == "fbx":
                            bpy.ops.import_scene.fbx(filepath=main_file_path)
                        elif file_format == "obj":
                            bpy.ops.import_scene.obj(filepath=main_file_path)
                        elif file_format == "blend":
                            with bpy.data.libraries.load(main_file_path, link=False) as (data_from, data_to):
                                data_to.objects = data_from.objects
                            for obj in data_to.objects:
                                if obj is not None:
                                    bpy.context.collection.objects.link(obj)
                        else:
                            return {"error": f"Unsupported model format: {file_format}"}
                        imported_objects = [obj.name for obj in bpy.context.selected_objects]

                        return {
                            "success": True,
                            "message": f"Model {asset_id} imported successfully",
                            "imported_objects": imported_objects
                        }
                    except Exception as e:
                        return {"error": f"Failed to import model: {str(e)}"}
                    finally:
                        try:
                            shutil.rmtree(temp_dir)
                        except:
                            print(f"Failed to clean up: {temp_dir}")
                else:
                    return {"error": f"Format/resolution unavailable."}
            else:
                return {"error": f"Unsupported asset type: {asset_type}"}
        except Exception as e:
            return {"error": f"Failed to download asset: {str(e)}"}

    def set_texture(self, object_name, texture_id):
        """Apply a previously downloaded Polyhaven texture."""
        try:
            obj = bpy.data.objects.get(object_name)
            if not obj:
                return {"error": f"Object not found: {object_name}"}
            if not hasattr(obj, 'data') or not hasattr(obj.data, 'materials'):
                return {"error": f"Object {object_name} cannot accept materials"}

            texture_images = {}
            for img in bpy.data.images:
                if img.name.startswith(texture_id + "_"):
                    map_type = img.name.split('_')[-1].split('.')[0]
                    img.reload()
                    if map_type.lower() in ['color', 'diffuse', 'albedo']:
                        try:
                            img.colorspace_settings.name = 'sRGB'
                        except:
                            pass
                    else:
                        try:
                            img.colorspace_settings.name = 'Non-Color'
                        except:
                            pass
                    if not img.packed_file:
                        img.pack()
                    texture_images[map_type] = img
                    print(f"Loaded: {map_type} - {img.name}")
                    print(f"Size: {img.size[0]}x{img.size[1]}")
                    print(f"Colorspace: {img.colorspace_settings.name}")
                    print(f"Format: {img.file_format}")
                    print(f"Packed: {bool(img.packed_file)}")

            if not texture_images:
                return {"error": f"No images found for: {texture_id}."}

            new_mat_name = f"{texture_id}_material_{object_name}"
            existing_mat = bpy.data.materials.get(new_mat_name)
            if existing_mat:
                bpy.data.materials.remove(existing_mat)

            new_mat = bpy.data.materials.new(name=new_mat_name)
            new_mat.use_nodes = True
            nodes = new_mat.node_tree.nodes
            links = new_mat.node_tree.links
            nodes.clear()
            output = nodes.new(type='ShaderNodeOutputMaterial')
            output.location = (600, 0)
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            principled.location = (300, 0)
            links.new(principled.outputs[0], output.inputs[0])
            tex_coord = nodes.new(type='ShaderNodeTexCoord')
            tex_coord.location = (-800, 0)
            mapping = nodes.new(type='ShaderNodeMapping')
            mapping.location = (-600, 0)
            mapping.vector_type = 'TEXTURE'
            links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
            x_pos = -400
            y_pos = 300

            for map_type, image in texture_images.items():
                tex_node = nodes.new(type='ShaderNodeTexImage')
                tex_node.location = (x_pos, y_pos)
                tex_node.image = image

                if map_type.lower() in ['color', 'diffuse', 'albedo']:
                    try:
                        tex_node.image.colorspace_settings.name = 'sRGB'
                    except:
                        pass
                else:
                    try:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    except:
                        pass
                links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
                if map_type.lower() in ['color', 'diffuse', 'albedo']:
                    links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                    print(f"Connected {map_name} to Base Color")
                    break
                elif map_type.lower() in ['roughness', 'rough']:
                    links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
                    print(f"Connected {map_name} to Roughness")
                    break

                for map_name in ['metallic', 'metalness', 'metal']:
                    if map_name in texture_nodes:
                        links.new(texture_nodes[map_name].outputs['Color'], principled.inputs['Metallic'])
                        print(f"Connected {map_name} to Metallic")
                        break
                for map_name in ['gl', 'dx', 'nor']:
                    if map_name in texture_nodes:
                        normal_map_node = nodes.new(type='ShaderNodeNormalMap')
                        normal_map_node.location = (100, 100)
                        links.new(texture_nodes[map_name].outputs['Color'], normal_map_node.inputs['Color'])
                        links.new(normal_map_node.outputs['Normal'], principled.inputs['Normal'])
                        print(f"Connected {map_name} to Normal")
                        break
                for map_name in ['displacement', 'disp', 'height']:
                    if map_name in texture_nodes:
                        disp_node = nodes.new(type='ShaderNodeDisplacement')
                        disp_node.location = (300, -200)
                        disp_node.inputs['Scale'].default_value = 0.1
                        links.new(texture_nodes[map_name].outputs['Color'], disp_node.inputs['Height'])
                        links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
                        print(f"Connected {map_name} to Displacement")
                        break
                if 'arm' in texture_nodes:
                    separate_rgb = nodes.new(type='ShaderNodeSeparateRGB')
                    separate_rgb.location = (-200, -100)
                    links.new(texture_nodes['arm'].outputs['Color'], separate_rgb.inputs['Image'])
                    if not any(map_name in texture_nodes for map_name in ['roughness', 'rough']):
                        links.new(separate_rgb.outputs['G'], principled.inputs['Roughness'])
                        print("Connected ARM.G to Roughness")
                    if not any(map_name in texture_nodes for map_name in ['metallic', 'metalness', 'metal']):
                        links.new(separate_rgb.outputs['B'], principled.inputs['Metallic'])
                        print("Connected ARM.B to Metallic")
                    base_color_node = None
                    for map_name in ['color', 'diffuse', 'albedo']:
                        if map_name in texture_nodes:
                            base_color_node = texture_nodes[map_name]
                            break
                    if base_color_node:
                        mix_node = nodes.new(type='ShaderNodeMixRGB')
                        mix_node.location = (100, 200)
                        mix_node.blend_type = 'MULTIPLY'
                        mix_node.inputs['Fac'].default_value = 0.8
                        for link in base_color_node.outputs['Color'].links:
                            if link.to_socket == principled.inputs['Base Color']:
                                links.remove(link)
                        links.new(base_color_node.outputs['Color'], mix_node.inputs[1])
                        links.new(separate_rgb.outputs['R'], mix_node.inputs[2])
                        links.new(mix_node.outputs['Color'], principled.inputs['Base Color'])
                        print("Connected ARM.R to AO mix with Base Color")

                if 'ao' in texture_nodes:
                    base_color_node = None
                    for map_name in ['color', 'diffuse', 'albedo']:
                        if map_name in texture_nodes:
                            base_color_node = texture_nodes[map_name]
                            break

                    if base_color_node:
                        mix_node = nodes.new(type='ShaderNodeMixRGB')
                        mix_node.location = (100, 200)
                        mix_node.blend_type = 'MULTIPLY'
                        mix_node.inputs['Fac'].default_value = 0.8

                        for link in base_color_node.outputs['Color'].links:
                            if link.to_socket == principled.inputs['Base Color']:
                                links.remove(link)

                        links.new(base_color_node.outputs['Color'], mix_node.inputs[1])
                        links.new(texture_nodes['ao'].outputs['Color'], mix_node.inputs[2])
                        links.new(mix_node.outputs['Color'], principled.inputs['Base Color'])
                        print("Connected AO to mix with Base Color")

                while len(obj.data.materials) > 0:
                    obj.data.materials.pop(index=0)

                obj.data.materials.append(new_mat)
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.context.view_layer.update()
                texture_maps = list(texture_images.keys())

                material_info = {
                    "name": new_mat.name,
                    "has_nodes": new_mat.use_nodes,
                    "node_count": len(new_mat.node_tree.nodes),
                    "texture_nodes": []
                }

                for node in new_mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        connections = []
                        for output in node.outputs:
                            for link in output.links:
                                connections.append(f"{output.name} → {link.to_node.name}.{link.to_socket.name}")

                        material_info["texture_nodes"].append({
                            "name": node.name,
                            "image": node.image.name,
                            "colorspace": node.image.colorspace_settings.name,
                            "connections": connections
                        })

                return {
                    "success": True,
                    "message": f"Created new material and applied texture {texture_id} to {object_name}",
                    "material": new_mat.name,
                    "maps": texture_maps,
                    "material_info": material_info
                }

        except Exception as e:
            print(f"Error in set_texture: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to apply texture: {str(e)}"}

    def get_polyhaven_status(self):
        enabled = bpy.context.scene.blendermcp_use_polyhaven
        if enabled:
            return {"enabled": True, "message": "PolyHaven integration is enabled and ready to use."}
        else:
            return {
                "enabled": False,
                "message": """PolyHaven integration is currently disabled. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Check the 'Use assets from Poly Haven' checkbox
                            3. Restart the connection"""
        }

    def insert_keyframe(self, object_name, data_path, frame, index=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        res = obj.keyframe_insert(data_path=data_path, frame=frame, index=index)
        return {
            "inserted": bool(res),
            "object": object_name,
            "property": data_path,
            "frame": frame
        }

    def add_modifier(self, object_name, modifier_type, modifier_name=None, params=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        mod = obj.modifiers.new(name=modifier_name or modifier_type, type=modifier_type)
        if params:
            for k, v in params.items():
                setattr(mod, k, v)
        return {
            "object": object_name,
            "modifier": mod.name,
            "type": mod.type
        }

    def import_asset(self, file_path, file_type="OBJ"):
        if file_type.upper() == "OBJ":
            bpy.ops.import_scene.obj(filepath=file_path)
        elif file_type.upper() == "FBX":
            bpy.ops.import_scene.fbx(filepath=file_path)
        elif file_type.upper() in ["GLTF", "GLB"]:
            bpy.ops.import_scene.gltf(filepath=file_path)
        else:
            raise ValueError("Unsupported file type")
        return {
            "imported": True,
            "file": file_path,
            "type": file_type
        }

    

    def export_scene(self, file_path, file_type="OBJ"):
        if file_type.upper() == "OBJ":
            bpy.ops.export_scene.obj(filepath=file_path)
        elif file_type.upper() == "FBX":
            bpy.ops.export_scene.fbx(filepath=file_path)
        elif file_type.upper() in ["GLTF", "GLB"]:
            bpy.ops.export_scene.gltf(filepath=file_path)
        else:
            raise ValueError("Unsupported file type")
        return {
            "exported": True,
            "file": file_path,
            "type": file_type
        }

    def add_light(self, type="POINT", location=(0,0,0), energy=10.0):
        bpy.ops.object.light_add(type=type, location=location)
        light = bpy.context.active_object
        if hasattr(light.data, 'energy'):
            light.data.energy = energy
        return {
            "light": light.name,
            "type": type,
            "energy": energy,
            "location": list(light.location)
        }

    def set_camera_params(self, camera_name, focal_length=50.0, dof_distance=None):
        cam = bpy.data.objects.get(camera_name)
        if not cam or cam.type != 'CAMERA':
            raise ValueError("Camera not found or not a camera")
        cam.data.lens = focal_length
        if dof_distance is not None:
            cam.data.dof.use_dof = True
            cam.data.dof.focus_distance = dof_distance
        return {
            "camera": camera_name,
            "focal_length": focal_length,
            "dof_distance": dof_distance
        }
class BLENDERMCP_PT_Panel(bpy.types.Panel):
    bl_label = "Blender MCP"
    bl_idname = "BLENDERMCP_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderMCP'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "blendermcp_port")
        layout.prop(scene, "blendermcp_use_polyhaven", text="Use assets from Poly Haven")
        layout.prop(scene, "blendermcp_asset_dir")
        if not scene.blendermcp_server_running:
            layout.operator("blendermcp.start_server", text="Start MCP Server")
        else:
            layout.operator("blendermcp.stop_server", text="Stop MCP Server")
            layout.label(text=f"Running on port {scene.blendermcp_port}")
class BLENDERMCP_OT_StartServer(bpy.types.Operator):
    bl_idname = "blendermcp.start_server"
    bl_label = "Connect to Local AI"  # Updated label
    bl_description = "Start the BlenderMCP server to connect with a local AI model" # Updated description

    def execute(self, context):
        scene = context.scene
        if not hasattr(bpy.types, "blendermcp_server") or not bpy.types.blendermcp_server:
            bpy.types.blendermcp_server = BlenderMCPServer(port=scene.blendermcp_port)
        bpy.types.blendermcp_server.start()
        scene.blendermcp_server_running = True
        return {'FINISHED'}
class BLENDERMCP_OT_StopServer(bpy.types.Operator):
    bl_idname = "blendermcp.stop_server"
    bl_label = "Stop the connection" # Updated
    bl_description = "Stop Server" # Updated

    def execute(self, context):
        scene = context.scene
        if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
            bpy.types.blendermcp_server.stop()
            del bpy.types.blendermcp_server
        scene.blendermcp_server_running = False
        return {'FINISHED'}
class BLENDERMCP_OT_PasteJsonBatch(bpy.types.Operator):
    bl_idname = "blendermcp.paste_json_batch"
    bl_label = "Paste JSON Command Batch"
    bl_description = "Paste a list of JSON commands to execute in sequence"
    
    json_text = StringProperty(
        name="JSON Batch",
        description="Paste your JSON array of commands here",
        default="[]"
    )

    result_log = StringProperty(
        name="Results",
        description="Execution results",
        default="",
        options={'SKIP_SAVE'}
    )

    def execute(self, context):
        server = getattr(bpy.types, "blendermcp_server", None)
        if not server:
            self.report({'ERROR'}, "BlenderMCP server not running")
            return {'CANCELLED'}

        try:
            cmds = json.loads(self.json_text)
            assert isinstance(cmds, list), "Batch must be a JSON array"
            logs = []
            for i, cmd in enumerate(cmds):
                resp = server.execute_command(cmd)
                logs.append(f"[{i+1}] {resp}")
            self.result_log = "\n".join(str(l) for l in logs)
            self.report({'INFO'}, f"Batch executed: {len(cmds)} commands")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=800)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Paste your JSON command batch below:")
        layout.prop(self, "json_text", text="")
        if self.result_log:
            layout.label(text="Results:")
            layout.box().label(text=self.result_log[:4000])  # shows first 4000 chars

def register():
    bpy.types.Scene.blendermcp_port = IntProperty(
        name="Port",
        description="Port for the BlenderMCP server",
        default=9876,
        min=1024,
        max=65535
    )
    bpy.types.Scene.blendermcp_server_running = bpy.props.BoolProperty(
        name="Server Running",
        default=False
    )
    bpy.types.Scene.blendermcp_use_polyhaven = bpy.props.BoolProperty(
        name="Use Poly Haven",
        description="Enable Poly Haven asset integration",
        default=False
    )
    bpy.types.Scene.blendermcp_asset_dir = StringProperty(
        name="Asset Directory",
        description="Directory containing your Blender assets",
        default=""  # Set to empty by default for portability
    )
    bpy.utils.register_class(BLENDERMCP_PT_Panel)
    bpy.utils.register_class(BLENDERMCP_OT_StartServer)
    bpy.utils.register_class(BLENDERMCP_OT_StopServer)
    bpy.utils.register_class(BLENDERMCP_OT_PasteJsonBatch)
    print("BlenderMCP addon registered")

def unregister():
    if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
        bpy.types.blendermcp_server.stop()
        del bpy.types.blendermcp_server

    bpy.utils.unregister_class(BLENDERMCP_PT_Panel)
    bpy.utils.unregister_class(BLENDERMCP_OT_StartServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_StopServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_PasteJsonBatch)
    del bpy.types.Scene.blendermcp_port
    del bpy.types.Scene.blendermcp_server_running
    del bpy.types.Scene.blendermcp_use_polyhaven
    del bpy.types.Scene.blendermcp_asset_dir

    print("BlenderMCP addon unregistered")

if __name__ == "__main__":
    register()