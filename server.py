# server.py
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Optional
import httpx
from io import BytesIO
import base64
import argparse
import os
from urllib.parse import urlparse
import os
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlenderMCPServer")

@dataclass
class BlenderConnection:
    host: str
    port: int
    sock: Optional[socket.socket] = None
    timeout: float = 15.0  # Added timeout as a property

    def __post_init__(self):
         if not isinstance(self.host, str):
             raise ValueError("Host must be a string")
         if not isinstance(self.port, int):
             raise ValueError("Port must be an int")

    def connect(self) -> bool:
        if self.sock:
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Blender at {self.host}:{self.port}")
            self.sock.settimeout(self.timeout) # Set timeout on socket
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Blender: {e!s}")
            self.sock = None
            return False

    def disconnect(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting: {e!s}")
            finally:
                self.sock = None

    def _receive_full_response(self, buffer_size: int = 8192) -> bytes:
        """Receive data with timeout using a loop."""
        chunks: List[bytes] = []
        timed_out = False
        try:
            while True:
                try:
                    chunk = self.sock.recv(buffer_size)
                    if not chunk:
                        if not chunks:
                            # Requirement 1b
                            raise Exception("Connection closed by Blender before any data was sent in this response")
                        else:
                            # Requirement 1a
                            raise Exception("Connection closed by Blender mid-stream with incomplete JSON data")
                    chunks.append(chunk)
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))  # Check if it is valid json
                        logger.debug(f"Received response ({len(data)} bytes)")
                        return data # Complete JSON received
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    logger.warning("Socket timeout during receive")
                    timed_out = True # Set flag
                    break # Stop listening to socket
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error: {e!s}")
                    self.sock = None
                    raise # re-raise to outer error handler
            
            # This part is reached if loop is broken by 'break' (only timeout case now)
            if timed_out:
                if chunks:
                    data = b''.join(chunks)
                    # Check if the partial data is valid JSON (it shouldn't be if timeout happened mid-stream)
                    try:
                        json.loads(data.decode('utf-8'))
                        # This case should ideally not be hit if JSON was incomplete,
                        # but if it's somehow valid, return it.
                        logger.warning("Timeout occurred, but received data forms valid JSON.")
                        return data
                    except json.JSONDecodeError:
                        # Requirement 2a
                        raise Exception(f"Incomplete JSON data received before timeout. Received: {data[:200]}")
                else:
                    # Requirement 2b
                    raise Exception("Timeout waiting for response, no data received.")
            
            # Fallback if loop exited for a reason not covered by explicit raises inside or by timeout logic
            # This should ideally not be reached with the current logic.
            if chunks: # Should have been handled by "Connection closed by Blender mid-stream..."
                data = b''.join(chunks)
                logger.warning(f"Exited receive loop unexpectedly with data: {data[:200]}")
                raise Exception("Receive loop ended unexpectedly with partial data.")
            else: # Should have been handled by "Connection closed by Blender before any data..." or timeout
                logger.warning("Exited receive loop unexpectedly with no data.")
                raise Exception("Receive loop ended unexpectedly with no data.")

        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            # This handles connection errors raised from within the loop or if self.sock.recv fails
            logger.error(f"Connection error during receive: {e!s}")
            self.sock = None # Ensure socket is reset
            # Re-raise with a more specific message if needed, or just re-raise
            raise Exception(f"Connection to Blender lost during receive: {e!s}")
        except Exception as e: 
            # Catch other exceptions, including our custom ones, and log them
            logger.error(f"Error during _receive_full_response: {e!s}")
            # If it's not one of the specific connection errors, it might be one of our custom messages
            # or another unexpected issue. Re-raise to be handled by send_command.
            raise


    def send_command(self, command_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
         if not self.sock and not self.connect():
            raise ConnectionError("Not connected")
         command = {"type": command_type, "params": params or {}}
         try:
              logger.info(f"Sending command: {command_type} with params: {params}")
              self.sock.sendall(json.dumps(command).encode('utf-8'))
              logger.info(f"Command sent, waiting for response...")
              response_data = self._receive_full_response()
              logger.debug(f"Received response ({len(response_data)} bytes)")
              response = json.loads(response_data.decode('utf-8'))
              logger.info(f"Response status: {response.get('status', 'unknown')}")
              if response.get("status") == "error":
                 logger.error(f"Blender error: {response.get('message')}")
                 raise Exception(response.get("message", "Unknown Blender error"))
              return response.get("result", {})

         except socket.timeout:
             logger.error("Socket timeout from Blender")
             self.sock = None # reset socket connection
             raise Exception("Timeout waiting for Blender - simplify request")
         except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
             logger.error(f"Socket connection error: {e!s}")
             self.sock = None # reset socket connection
             raise Exception(f"Connection to Blender lost: {e!s}")
         except json.JSONDecodeError as e:
             logger.error(f"Invalid JSON response: {e!s}")
             if 'response_data' in locals() and response_data:
                logger.error(f"Raw (first 200): {response_data[:200]}")
             raise Exception(f"Invalid response from Blender: {e!s}")
         except Exception as e:
              logger.error(f"Error communicating with Blender: {e!s}")
              self.sock = None # reset socket connection
              raise Exception(f"Communication error: {e!s}")


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    logger.info("BlenderMCP server starting up")
    try:
        blender = get_blender_connection()
        logger.info("Connected to Blender on startup")
    except Exception as e:
        logger.warning(f"Could not connect to Blender on startup: {e!s}")
        logger.warning("Ensure Blender addon is running before using resources")
    yield {}
    global _blender_connection
    if _blender_connection:
        logger.info("Disconnecting from Blender on shutdown")
        _blender_connection.disconnect()
        _blender_connection = None
    logger.info("BlenderMCP server shut down")

# Initialize MCP server instance globally
mcp = FastMCP(
    "BlenderOpenMCP",
    description="Blender integration with local AI models via Ollama",
    lifespan=server_lifespan
)

_blender_connection = None
_polyhaven_enabled = False
# Default values (will be overridden by command-line arguments)
_ollama_model = ""
_ollama_url = "http://localhost:11434"

def get_blender_connection() -> BlenderConnection:
    global _blender_connection, _polyhaven_enabled
    if _blender_connection:
        try:
            result = _blender_connection.send_command("get_polyhaven_status")
            _polyhaven_enabled = result.get("enabled", False)
            return _blender_connection
        except Exception as e:
            logger.warning(f"Existing connection invalid: {e!s}")
            try:
                _blender_connection.disconnect()
            except:
                pass
            _blender_connection = None
    if _blender_connection is None:
        _blender_connection = BlenderConnection(host="localhost", port=9876)
        if not _blender_connection.connect():
            logger.error("Failed to connect to Blender")
            _blender_connection = None
            raise Exception("Could not connect to Blender. Addon running?")
        logger.info("Created new persistent connection to Blender")
    return _blender_connection


async def query_ollama(prompt: str, context: Optional[List[Dict]] = None, image: Optional[Image] = None) -> str:
    global _ollama_model, _ollama_url

    payload = {"prompt": prompt, "model": _ollama_model, "format": "json", "stream": False}
    if context:
        payload["context"] = context
    if image:
        if image.data:
            payload["images"] = [image.data]
        elif image.path:
            try:
                with open(image.path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                payload["images"] = [encoded_string]
            except FileNotFoundError:
                logger.error(f"Image file not found: {image.path}")
                return "Error: Image file not found."
        else:
            logger.warning("Image without data or path. Ignoring.")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{_ollama_url}/api/generate", json=payload, timeout=60.0)
            response.raise_for_status()  # Raise HTTPStatusError for bad status
            response_data = response.json()
            logger.debug(f"Raw Ollama response: {response_data}")
            if "response" in response_data:
                return response_data["response"]
            else:
                logger.error(f"Unexpected response format: {response_data}")
                return "Error: Unexpected response format from Ollama."

    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
        return f"Error: Ollama API returned: {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error(f"Ollama API request failed: {e}")
        return "Error: Failed to connect to Ollama API."
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e!s}")
        return f"Error: An unexpected error occurred: {e!s}"
# ASSET LISTING
@mcp.tool()
def list_assets(ctx: Context, asset_dir: str, file_types: list = None, recursive: bool = True) -> str:
    """
    List all files in an asset directory. Optionally filter by file extension/type.
    """
    asset_list = []
    for root, dirs, files in os.walk(asset_dir):
        for fname in files:
            if file_types:
                ext = os.path.splitext(fname)[1].lower()
                if ext.lstrip(".") not in [ft.lower().lstrip(".") for ft in file_types]:
                    continue
            rel_path = os.path.relpath(os.path.join(root, fname), asset_dir)
            asset_list.append(rel_path)
        if not recursive:
            break
    return json.dumps({"assets": asset_list, "count": len(asset_list)}, indent=2)

# ASSET PREVIEW (just metadata, optionally first few lines of an OBJ/MTL/JPG/etc)
@mcp.tool()
def get_asset_info(ctx: Context, asset_path: str) -> str:
    """
    Return some metadata about a file (size, type, optionally preview for image/model).
    """
    if not os.path.isfile(asset_path):
        return f"Error: File not found: {asset_path}"
    size = os.path.getsize(asset_path)
    ext = os.path.splitext(asset_path)[1].lower()
    mime, _ = mimetypes.guess_type(asset_path)
    preview = None
    # For images, provide a base64 thumbnail (optional for future)
    if ext in [".jpg", ".jpeg", ".png"]:
        try:
            from PIL import Image as PILImage
            import base64, io
            with PILImage.open(asset_path) as im:
                im.thumbnail((128,128))
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                preview = base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            preview = None
    # For OBJ, show first few lines
    elif ext in [".obj", ".mtl"]:
        with open(asset_path, "r", encoding="utf-8", errors="ignore") as f:
            preview = "".join([next(f) for _ in range(10)])
    return json.dumps({
        "file": asset_path, "size": size, "type": mime, "preview": preview
    }, indent=2)
@mcp.tool()
def import_image_to_plane(ctx: Context, image_path: str, location: list = [0,0,0], size: float = 1.0) -> str:
    blender = get_blender_connection()
    res = blender.send_command("import_image_to_plane", {"image_path": image_path, "location": location, "size": size})
    return f"Image plane imported: {res.get('object', 'unknown')}" if res.get("status")=="success" else f"Error: {res.get('message', res)}"
@mcp.tool()
def search_assets(ctx: Context, query: str = None, tags: list = None, type: str = None) -> str:
    blender = get_blender_connection()
    result = blender.send_command("search_assets", {"query": query, "tags": tags, "type": type})
    return json.dumps(result, indent=2)

@mcp.tool()
def get_scene_graph(ctx: Context) -> str:
    blender = get_blender_connection()
    result = blender.send_command("get_scene_graph")
    return json.dumps(result, indent=2)

@mcp.tool()
def batch_place_assets(ctx: Context, asset_names: list, positions: list, rotations: list = None, scales: list = None) -> str:
    blender = get_blender_connection()
    result = blender.send_command("batch_place_assets", {
        "asset_names": asset_names,
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
    })
    return json.dumps(result, indent=2)

@mcp.tool()
def group_objects(ctx: Context, object_names: list, group_name: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("group_objects", {
        "object_names": object_names,
        "group_name": group_name,
    })
    return json.dumps(result, indent=2)
@mcp.tool()
def semantic_edit(ctx: Context, instruction: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("semantic_edit", {"instruction": instruction})
    return json.dumps(result, indent=2)
@mcp.tool()
def suggest_scene_improvements(ctx: Context) -> str:
    blender = get_blender_connection()
    result = blender.send_command("suggest_scene_improvements")
    return json.dumps(result, indent=2)
@mcp.tool()
def batch_import_assets(ctx: Context, folder_path: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("batch_import_assets", {"folder_path": folder_path})
    return json.dumps(result, indent=2)
@mcp.tool()
def generate_parametric_asset(ctx: Context, type: str, params: dict) -> str:
    blender = get_blender_connection()
    result = blender.send_command("generate_parametric_asset", {"type": type, "params": params})
    return json.dumps(result, indent=2)

@mcp.tool()
def randomize_scene(ctx: Context, n_variations: int = 3) -> str:
    blender = get_blender_connection()
    result = blender.send_command("randomize_scene", {"n_variations": n_variations})
    return json.dumps(result, indent=2)
@mcp.tool()
def image_to_geometry(ctx: Context, image_path: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("image_to_geometry", {"image_path": image_path})
    return json.dumps(result, indent=2)

@mcp.tool()
def transfer_style_from_image(ctx: Context, image_path: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("transfer_style_from_image", {"image_path": image_path})
    return json.dumps(result, indent=2)
@mcp.tool()
def add_human_figure(ctx: Context, pose: str = "standing") -> str:
    blender = get_blender_connection()
    result = blender.send_command("add_human_figure", {"pose": pose})
    return json.dumps(result, indent=2)

@mcp.tool()
def undo_last_action(ctx: Context) -> str:
    blender = get_blender_connection()
    result = blender.send_command("undo_last_action")
    return json.dumps(result, indent=2)

@mcp.tool()
def redo_action(ctx: Context) -> str:
    blender = get_blender_connection()
    result = blender.send_command("redo_action")
    return json.dumps(result, indent=2)

@mcp.tool()
def save_scene_variant(ctx: Context, name: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("save_scene_variant", {"name": name})
    return json.dumps(result, indent=2)

@mcp.tool()
def load_scene_variant(ctx: Context, name: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("load_scene_variant", {"name": name})
    return json.dumps(result, indent=2)

@mcp.tool()
def create_room(ctx: Context, width: float, depth: float, height: float, style: str = "modern", doors: int = 1, windows: int = 1) -> str:
    blender = get_blender_connection()
    result = blender.send_command("create_room", {
        "width": width, "depth": depth, "height": height, "style": style, "doors": doors, "windows": windows,
    })
    return json.dumps(result, indent=2)
@mcp.tool()
def auto_lighting(ctx: Context, style: str = "natural") -> str:
    blender = get_blender_connection()
    result = blender.send_command("auto_lighting", {"style": style})
    return json.dumps(result, indent=2)

@mcp.tool()
def auto_place_camera(ctx: Context, mode: str = "best_view") -> str:
    blender = get_blender_connection()
    result = blender.send_command("auto_place_camera", {"mode": mode})
    return json.dumps(result, indent=2)
@mcp.tool()
def scene_quality_check(ctx: Context) -> str:
    blender = get_blender_connection()
    result = blender.send_command("scene_quality_check")
    return json.dumps(result, indent=2)

@mcp.tool()
def auto_place_asset(ctx: Context, asset_name: str, target_area: str, constraints: dict = None) -> str:
    blender = get_blender_connection()
    result = blender.send_command("auto_place_asset", {
        "asset_name": asset_name, "target_area": target_area, "constraints": constraints,
    })
    return json.dumps(result, indent=2)
@mcp.tool()
def auto_arrange_furniture(ctx: Context, area_type: str = "kitchen") -> str:
    blender = get_blender_connection()
    result = blender.send_command("auto_arrange_furniture", {"area_type": area_type})
    return json.dumps(result, indent=2)
@mcp.tool()
def harmonize_materials(ctx: Context, palette: list = None, style: str = None) -> str:
    blender = get_blender_connection()
    result = blender.send_command("harmonize_materials", {"palette": palette, "style": style})
    return json.dumps(result, indent=2)

@mcp.tool()
def place_asset_smart(ctx: Context, asset_name: str, area: str, constraints: dict = None) -> str:
    """Place asset with collision-avoidance and surface snapping."""
    blender = get_blender_connection()
    result = blender.send_command("place_asset_smart", {
        "asset_name": asset_name,
        "area": area,
        "constraints": constraints
    })
    return json.dumps(result, indent=2)

@mcp.tool()
def create_scene_from_text(ctx: Context, description: str, style: str = None) -> str:
    blender = get_blender_connection()
    result = blender.send_command("create_scene_from_text", {
        "description": description, "style": style,
    })
    return json.dumps(result, indent=2)

@mcp.tool()
def create_scene_from_image(ctx: Context, image_path: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("create_scene_from_image", {
        "image_path": image_path,
    })
    return json.dumps(result, indent=2)

@mcp.tool()
def batch_render(ctx: Context, views: list, resolution: list = [1920,1080], preset: str = "high_quality") -> str:
    blender = get_blender_connection()
    result = blender.send_command("batch_render", {
        "views": views, "resolution": resolution, "preset": preset,
    })
    return json.dumps(result, indent=2)

@mcp.tool()
def get_asset_metadata(ctx: Context, asset_path: str) -> str:
    blender = get_blender_connection()
    result = blender.send_command("get_asset_metadata", {
        "asset_path": asset_path,
    })
    return json.dumps(result, indent=2)

@mcp.tool()
def append_blend(ctx: Context, blend_path: str, data_type: str, object_name: str) -> str:
    blender = get_blender_connection()
    res = blender.send_command("append_blend", {"blend_path": blend_path, "data_type": data_type, "object_name": object_name})
    return f"Appended: {object_name} from {blend_path}" if res.get("status")=="success" else f"Error: {res.get('message', res)}"

# Repeat for others...

# IMPORT 3D MODEL
@mcp.tool()
def import_model(ctx: Context, asset_path: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("import_asset", {"file_path": asset_path})
        return f"Imported model: {asset_path}"
    except Exception as e:
        return f"Error: {e!s}"

# IMPORT TEXTURE
@mcp.tool()
def import_texture(ctx: Context, texture_path: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("import_texture", {"file_path": texture_path})
        return f"Imported texture: {texture_path}"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.prompt()
async def base_prompt(context: Context, user_message: str) -> str:
    system_message = f"""You are a helpful assistant that controls Blender.
    You can use the following tools. Respond in well-formatted, valid JSON:
    {mcp.tools_schema()}"""
    full_prompt = f"{system_message}\n\n{user_message}"
    response = await query_ollama(full_prompt, context.history(), context.get_image())
    return response

@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")
        return json.dumps(result, indent=2)  # Return as a formatted string
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def get_object_info(ctx: Context, object_name: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_info", {"name": object_name})
        return json.dumps(result, indent=2)  # Return as a formatted string
    except Exception as e:
        return f"Error: {e!s}"
    
@mcp.tool()
def create_object(
    ctx: Context,
    type: str = "CUBE",
    name: Optional[str] = None,
    location: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    scale: Optional[List[float]] = None
) -> str:
    try:
        blender = get_blender_connection()
        loc, rot, sc = location or [0, 0, 0], rotation or [0, 0, 0], scale or [1, 1, 1]
        params = {"type": type, "location": loc, "rotation": rot, "scale": sc}
        if name: params["name"] = name
        result = blender.send_command("create_object", params)
        return f"Created {type} object: {result['name']}"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def modify_object(
    ctx: Context,
    name: str,
    location: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    scale: Optional[List[float]] = None,
    visible: Optional[bool] = None
) -> str:
    try:
        blender = get_blender_connection()
        params = {"name": name}
        if location is not None: params["location"] = location
        if rotation is not None: params["rotation"] = rotation
        if scale is not None: params["scale"] = scale
        if visible is not None: params["visible"] = visible
        result = blender.send_command("modify_object", params)
        return f"Modified object: {result['name']}"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def delete_object(ctx: Context, name: str) -> str:
    try:
        blender = get_blender_connection()
        blender.send_command("delete_object", {"name": name})
        return f"Deleted object: {name}"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def set_material(
    ctx: Context,
    object_name: str,
    material_name: Optional[str] = None,
    color: Optional[List[float]] = None
) -> str:
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name}
        if material_name: params["material_name"] = material_name
        if color: params["color"] = color
        result = blender.send_command("set_material", params)
        return f"Applied material to {object_name}: {result.get('material_name', 'unknown')}"
    except Exception as e:
        return f"Error: {e!s}"
    
@mcp.tool()
def execute_blender_code(ctx: Context, code: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": code})
        return f"Code executed: {result.get('result', '')}"
    except Exception as e:
        return f"Error: {e!s}"
    
@mcp.tool()
def get_polyhaven_categories(ctx: Context, asset_type: str = "hdris") -> str:
    try:
        blender = get_blender_connection()
        if not _polyhaven_enabled: return "PolyHaven disabled."
        result = blender.send_command("get_polyhaven_categories", {"asset_type": asset_type})
        if "error" in result: return f"Error: {result['error']}"
        categories = result["categories"]
        formatted = f"Categories for {asset_type}:\n" + \
                    "\n".join(f"- {cat}: {count}" for cat, count in
                      sorted(categories.items(), key=lambda x: x[1], reverse=True))
        return formatted
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def search_polyhaven_assets(ctx: Context, asset_type: str = "all", categories: Optional[str] = None) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("search_polyhaven_assets",
                {"asset_type": asset_type, "categories": categories})
        if "error" in result: return f"Error: {result['error']}"
        assets, total, returned = result["assets"], result["total_count"], result["returned_count"]
        formatted = f"Found {total} assets" + (f" in: {categories}" if categories else "") + \
                    f"\nShowing {returned}:\n" + "".join(
            f"- {data.get('name', asset_id)} (ID: {asset_id})\n"
            f"  Type: {['HDRI', 'Texture', 'Model'][data.get('type', 0)]}\n"
            f"  Categories: {', '.join(data.get('categories', []))}\n"
            f"  Downloads: {data.get('download_count', 'Unknown')}\n"
            for asset_id, data in sorted(assets.items(),
                                        key=lambda x: x[1].get("download_count", 0),
                                        reverse=True))
        return formatted
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def download_polyhaven_asset(ctx: Context, asset_id: str, asset_type: str,
                             resolution: str = "1k", file_format: Optional[str] = None) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("download_polyhaven_asset", {
            "asset_id": asset_id, "asset_type": asset_type,
            "resolution": resolution, "file_format": file_format})
        if "error" in result: return f"Error: {result['error']}"
        if result.get("success"):
            message = result.get("message", "Success")
            if asset_type == "hdris": return f"{message}. HDRI set as world."
            elif asset_type == "textures":
                mat_name, maps = result.get("material", ""), ", ".join(result.get("maps", []))
                return f"{message}. Material '{mat_name}' with: {maps}."
            elif asset_type == "models": return f"{message}. Model imported."
            return message
        return f"Failed: {result.get('message', 'Unknown')}"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def set_texture(ctx: Context, object_name: str, texture_id: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_texture",
                                     {"object_name": object_name, "texture_id": texture_id})
        if "error" in result: return f"Error: {result['error']}"
        if result.get("success"):
            mat_name, maps = result.get("material", ""), ", ".join(result.get("maps", []))
            info, nodes = result.get("material_info", {}), result.get("material_info", {}).get("texture_nodes", [])
            output = (f"Applied '{texture_id}' to {object_name}.\nMaterial '{mat_name}': {maps}.\n"
                      f"Nodes: {info.get('has_nodes', False)}\nCount: {info.get('node_count', 0)}\n")
            if nodes:
                output += "Texture nodes:\n" + "".join(
                    f"- {node['name']} ({node['image']})\n" +
                    ("  Connections:\n" + "".join(f"    {conn}\n" for conn in node['connections'])
                     if node['connections'] else "")
                    for node in nodes)
            return output
        return f"Failed: {result.get('message', 'Unknown')}"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def get_polyhaven_status(ctx: Context) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_polyhaven_status")
        return result.get("message", "")  # Return the message directly
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
async def set_ollama_model(ctx: Context, model_name: str) -> str:
    global _ollama_model
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{_ollama_url}/api/show",
                                         json={"name": model_name}, timeout=10.0)
            if response.status_code == 200:
                _ollama_model = model_name
                return f"Ollama model set to: {_ollama_model}"
            else: return f"Error: Could not find model '{model_name}'."
    except Exception as e:
        return f"Error: Failed to communicate: {e!s}"

@mcp.tool()
async def set_ollama_url(ctx: Context, url: str) -> str:
    global _ollama_url
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: Invalid URL format. Must start with http:// or https://."
    _ollama_url = url
    return f"Ollama URL set to: {_ollama_url}"

@mcp.tool()
async def get_ollama_models(ctx: Context) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{_ollama_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            models_data = response.json()
            if "models" in models_data:
                model_list = [model["name"] for model in models_data["models"]]
                return "Available Ollama models:\n" + "\n".join(model_list)
            else: return "Error: Unexpected response from Ollama /api/tags."
    except httpx.HTTPStatusError as e:
        return f"Error: Ollama API error: {e.response.status_code}"
    except httpx.RequestError as e:
        return "Error: Failed to connect to Ollama API."
    except Exception as e:
        return f"Error: An unexpected error: {e!s}"
@mcp.tool()
def insert_keyframe(ctx: Context, object_name: str, data_path: str, frame: int, index: int = None) -> str:
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name, "data_path": data_path, "frame": frame}
        if index is not None: params["index"] = index
        result = blender.send_command("insert_keyframe", params)
        return f"Keyframe inserted for {object_name} at frame {frame} ({data_path})"
    except Exception as e:
        return f"Error: {e!s}"
@mcp.tool()
def scatter_objects(ctx: Context, source: str, target: str, count: int = 10, method: str = "surface") -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("scatter_objects", {"source": source, "target": target, "count": count, "method": method})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"
@mcp.tool()
def apply_geometry_nodes(ctx: Context, object_name: str, node_group_name: str, inputs: dict = None) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("apply_geometry_nodes", {"object_name": object_name, "node_group_name": node_group_name, "inputs": inputs})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"
@mcp.tool()
def brush_asset_activate(ctx: Context,
    asset_library_type: str = "LOCAL",
    asset_library_identifier: str = "",
    relative_asset_identifier: str = "",
    use_toggle: bool = False
) -> str:
    try:
        blender = get_blender_connection()
        params = {
            "asset_library_type": asset_library_type,
            "asset_library_identifier": asset_library_identifier,
            "relative_asset_identifier": relative_asset_identifier,
            "use_toggle": use_toggle,
        }
        result = blender.send_command("brush_asset_activate", params)
        return result.get("result", str(result))
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def brush_asset_delete(ctx: Context) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("brush_asset_delete")
        return result.get("result", str(result))
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def brush_asset_edit_metadata(ctx: Context,
    catalog_path: str = "",
    author: str = "",
    description: str = ""
) -> str:
    try:
        blender = get_blender_connection()
        params = {
            "catalog_path": catalog_path,
            "author": author,
            "description": description
        }
        result = blender.send_command("brush_asset_edit_metadata", params)
        return result.get("result", str(result))
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def apply_modifier(ctx: Context, object_name: str, modifier_name: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("apply_modifier", {"object_name": object_name, "modifier_name": modifier_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def remove_modifier(ctx: Context, object_name: str, modifier_name: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("remove_modifier", {"object_name": object_name, "modifier_name": modifier_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def create_curve(ctx: Context, curve_type: str = "BEZIER", points: list = None) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_curve", {"curve_type": curve_type, "points": points})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def sweep_mesh_along_curve(ctx: Context, mesh_name: str, curve_name: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("sweep_mesh_along_curve", {"mesh_name": mesh_name, "curve_name": curve_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def convert_curve_to_mesh(ctx: Context, curve_name: str) -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("convert_curve_to_mesh", {"curve_name": curve_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def add_modifier(
    ctx: Context, 
    object_name: str, 
    modifier_type: str, 
    modifier_name: str = None, 
    params: dict = None
) -> str:
    try:
        blender = get_blender_connection()
        args = {
            "object_name": object_name,
            "modifier_type": modifier_type,
            "params": params or {}
        }
        if modifier_name:
            args["modifier_name"] = modifier_name
        result = blender.send_command("add_modifier", args)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"

    
@mcp.tool()
def boolean_operation(ctx: Context, object_a: str, object_b: str, operation: str = "UNION") -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("boolean_operation", {"object_a": object_a, "object_b": object_b, "operation": operation})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def import_asset(ctx: Context, file_path: str, file_type: str = "OBJ") -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("import_asset", {"file_path": file_path, "file_type": file_type})
        return f"Imported asset: {file_path} ({file_type})"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def export_scene(ctx: Context, file_path: str, file_type: str = "OBJ") -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("export_scene", {"file_path": file_path, "file_type": file_type})
        return f"Exported scene to: {file_path} ({file_type})"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def add_light(ctx: Context, type: str = "POINT", location: list = None, energy: float = 10.0) -> str:
    try:
        blender = get_blender_connection()
        params = {"type": type, "energy": energy}
        if location: params["location"] = location
        result = blender.send_command("add_light", params)
        return f"Added {type} light at {location or [0,0,0]}"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
def set_camera_params(ctx: Context, camera_name: str, focal_length: float = 50.0, dof_distance: float = None) -> str:
    try:
        blender = get_blender_connection()
        params = {"camera_name": camera_name, "focal_length": focal_length}
        if dof_distance is not None: params["dof_distance"] = dof_distance
        result = blender.send_command("set_camera_params", params)
        return f"Set camera params for {camera_name}: focal {focal_length}, dof {dof_distance}"
    except Exception as e:
        return f"Error: {e!s}"

@mcp.tool()
async def render_image(ctx: Context, file_path: str = "render.png") -> str:
    try:
        blender = get_blender_connection()
        result = blender.send_command("render_scene", {"output_path":file_path})
        if result:
            try:
                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    ctx.add_image(Image(data=encoded_string)) # Add image to the context
                    return "Image Rendered Successfully."
            except Exception as exception:
                return f"Blender rendered, however image could not be found. {exception!s}" # Use exception
    except Exception as e:
        return f"Error: {e!s}"

def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="BlenderMCP Server")
    parser.add_argument("--ollama-url", type=str, default=_ollama_url,
                        help="URL of the Ollama server")
    parser.add_argument("--ollama-model", type=str, default=_ollama_model,
                        help="Default Ollama model to use")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the MCP server to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host for the MCP server to listen on")

    args = parser.parse_args()

    # Set global variables from command-line arguments
    global _ollama_url, _ollama_model
    _ollama_url = args.ollama_url
    _ollama_model = args.ollama_model

    # MCP instance is already created globally
    mcp.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()