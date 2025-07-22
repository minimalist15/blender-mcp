# blender-open-mcp

`blender-open-mcp` is an open source project that integrates Blender with local AI models (via [Ollama](https://ollama.com/)) using the Model Context Protocol (MCP). This allows you to control Blender using natural language prompts, leveraging the power of AI to assist with 3D modeling tasks.

## Features

- **Control Blender with Natural Language:** Send prompts to a locally running Ollama model to perform actions in Blender.
- **MCP Integration:** Uses the Model Context Protocol for structured communication between the AI model and Blender.
- **Ollama Support:** Designed to work with Ollama for easy local model management.
- **Blender Add-on:** Includes a Blender add-on to provide a user interface and handle communication with the server.
- **PolyHaven Integration (Optional):** Download and use assets (HDRIs, textures, models) from [PolyHaven](https://polyhaven.com/) directly within Blender via AI prompts.
- **Basic 3D Operations:**
  - Get Scene and Object Info
  - Create Primitives
  - Modify and delete objects
  - Apply materials
- **Render Support:** Render images using the tool and retrieve information based on the output.

## Installation

### Prerequisites

1. **Blender:** Blender 3.0 or later. Download from [blender.org](https://www.blender.org/download/).
2. **Ollama:** Install from [ollama.com](https://ollama.com/), following OS-specific instructions.
3. **Python:** Python 3.10 or later.
4. **uv:** Install using `pip install uv`.
5. **Git:** Required for cloning the repository.

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/dhakalnirajan/blender-open-mcp.git
   cd blender-open-mcp
   ```

2. **Create and Activate a Virtual Environment (Recommended):**

   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/macOS
   .venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies:**

   ```bash
   uv pip install -e .
   ```

4. **Install the Blender Add-on:**

   - Open Blender.
   - Go to `Edit -> Preferences -> Add-ons`.
   - Click `Install...`.
   - Select the `addon.py` file from the `blender-open-mcp` directory.
   - Enable the "Blender MCP" add-on.

5. **Download an Ollama Model (if not already installed):**

   ```bash
   ollama run llama3.2
   ```

   *(Other models like **`Gemma3`** can also be used.)*

## Setup

1. **Start the Ollama Server:** Ensure Ollama is running in the background.

2. **Start the MCP Server:**

   ```bash
   blender-mcp
   ```

   Or,

   ```bash
   python src/blender_open_mcp/server.py
   ```

   By default, it listens on `http://0.0.0.0:8000`, but you can modify settings:

   ```bash
   blender-mcp --host 127.0.0.1 --port 8001 --ollama-url http://localhost:11434 --ollama-model llama3.2
   ```

3. **Start the Blender Add-on Server:**

   - Open Blender and the 3D Viewport.
   - Press `N` to open the sidebar.
   - Find the "Blender MCP" panel.
   - Click "Start MCP Server".

## Usage

Interact with `blender-open-mcp` using the `mcp` command-line tool:

### Example Commands

- **Basic Prompt:**

  ```bash
  mcp prompt "Hello BlenderMCP!" --host http://localhost:8000
  ```

- **Get Scene Information:**

  ```bash
  mcp tool get_scene_info --host http://localhost:8000
  ```

- **Create a Cube:**

  ```bash
  mcp prompt "Create a cube named 'my_cube'." --host http://localhost:8000
  ```

- **Render an Image:**

  ```bash
  mcp prompt "Render the image." --host http://localhost:8000
  ```

- **Using PolyHaven (if enabled):**

  ```bash
  mcp prompt "Download a texture from PolyHaven." --host http://localhost:8000
  ```

## Available Tools

| Tool Name                  | Description                            | Parameters                                            |
| -------------------------- | -------------------------------------- | ----------------------------------------------------- |
| `get_scene_info`           | Retrieves scene details.               | None                                                  |
| `get_object_info`          | Retrieves information about an object. | `object_name` (str)                                   |
| `create_object`            | Creates a 3D object.                   | `type`, `name`, `location`, `rotation`, `scale`       |
| `modify_object`            | Modifies an object’s properties.       | `name`, `location`, `rotation`, `scale`, `visible`    |
| `delete_object`            | Deletes an object.                     | `name` (str)                                          |
| `set_material`             | Assigns a material to an object.       | `object_name`, `material_name`, `color`               |
| `render_image`             | Renders an image.                      | `file_path` (str)                                     |
| `execute_blender_code`     | Executes Python code in Blender.       | `code` (str)                                          |
| `get_polyhaven_categories` | Lists PolyHaven asset categories.      | `asset_type` (str)                                    |
| `search_polyhaven_assets`  | Searches PolyHaven assets.             | `asset_type`, `categories`                            |
| `download_polyhaven_asset` | Downloads a PolyHaven asset.           | `asset_id`, `asset_type`, `resolution`, `file_format` |
| `set_texture`              | Applies a downloaded texture.          | `object_name`, `texture_id`                           |
| `set_ollama_model`         | Sets the Ollama model.                 | `model_name` (str)                                    |
| `set_ollama_url`           | Sets the Ollama server URL.            | `url` (str)                                           |
| `get_ollama_models`        | Lists available Ollama models.         | None                                                  |
| `list_assets`              | Lists all files in an asset directory. | `asset_dir`, `file_types`, `recursive`                |
| `get_asset_info`           | Returns metadata about a file.         | `asset_path`                                          |
| `import_image_to_plane`    | Imports an image as a plane.           | `image_path`, `location`, `size`                      |
| `search_assets`            | Searches local assets.                 | `query`, `tags`, `type`                               |
| `get_scene_graph`          | Returns the scene graph.               | None                                                  |
| `batch_place_assets`       | Places multiple assets in the scene.   | `asset_names`, `positions`, `rotations`, `scales`     |
| `group_objects`            | Groups objects under a parent.         | `object_names`, `group_name`                          |
| `semantic_edit`            | Applies a semantic edit instruction.   | `instruction`                                         |
| `suggest_scene_improvements`| Suggests improvements for the scene.  | None                                                  |
| `batch_import_assets`      | Imports all assets from a folder.      | `folder_path`                                         |
| `generate_parametric_asset`| Generates a parametric asset.          | `type`, `params`                                      |
| `randomize_scene`          | Creates scene variations.              | `n_variations`                                        |
| `image_to_geometry`        | Converts an image to geometry.         | `image_path`                                          |
| `transfer_style_from_image`| Transfers style from an image.         | `image_path`                                          |
| `add_human_figure`         | Adds a human figure to the scene.      | `pose`                                                |
| `create_room`              | Creates a room with given parameters.  | `width`, `depth`, `height`, `style`, `doors`, `windows`|
| `auto_lighting`            | Adds lighting to the scene.            | `style`                                               |
| `auto_place_camera`        | Places the camera automatically.       | `mode`                                                |
| `scene_quality_check`      | Checks the quality of the scene.       | None                                                  |
| `auto_place_asset`         | Places an asset automatically.         | `asset_name`, `target_area`, `constraints`            |
| `auto_arrange_furniture`   | Arranges furniture in an area.         | `area_type`                                           |
| `harmonize_materials`      | Harmonizes materials in the scene.     | `palette`, `style`                                    |
| `place_asset_smart`        | Places asset with collision avoidance. | `asset_name`, `area`, `constraints`                   |
| `create_scene_from_text`   | Creates a scene from a text description.| `description`, `style`                               |
| `create_scene_from_image`  | Creates a scene from an image.         | `image_path`                                          |
| `batch_render`             | Renders multiple views.                | `views`, `resolution`, `preset`                       |
| `get_asset_metadata`       | Gets metadata for an asset.            | `asset_path`                                          |
| `append_blend`             | Appends data from another .blend file. | `blend_path`, `data_type`, `object_name`              |
| `import_model`             | Imports a 3D model.                    | `asset_path`                                          |
| `import_texture`           | Imports a texture.                     | `texture_path`                                        |
| `insert_keyframe`          | Inserts a keyframe for an object.      | `object_name`, `data_path`, `frame`, `index`          |
| `scatter_objects`          | Scatters objects on a target.          | `source`, `target`, `count`, `method`                 |
| `apply_geometry_nodes`     | Applies geometry nodes to an object.   | `object_name`, `node_group_name`, `inputs`            |
| `brush_asset_activate`     | Activates a brush asset.               | `asset_library_type`, `asset_library_identifier`, `relative_asset_identifier`, `use_toggle` |
| `brush_asset_delete`       | Deletes the active brush asset.        | None                                                  |
| `brush_asset_edit_metadata`| Edits brush asset metadata.            | `catalog_path`, `author`, `description`               |
| `apply_modifier`           | Applies a modifier to an object.       | `object_name`, `modifier_name`                        |
| `remove_modifier`          | Removes a modifier from an object.     | `object_name`, `modifier_name`                        |
| `create_curve`             | Creates a curve object.                | `curve_type`, `points`                                |
| `sweep_mesh_along_curve`   | Sweeps a mesh along a curve.           | `mesh_name`, `curve_name`                             |
| `convert_curve_to_mesh`    | Converts a curve to a mesh.            | `curve_name`                                          |
| `add_modifier`             | Adds a modifier to an object.          | `object_name`, `modifier_type`, `modifier_name`, `params` |
| `boolean_operation`        | Performs a boolean operation.          | `object_a`, `object_b`, `operation`                   |
| `import_asset`             | Imports an asset file.                 | `file_path`, `file_type`                              |
| `export_scene`             | Exports the scene to a file.           | `file_path`, `file_type`                              |
| `add_light`                | Adds a light to the scene.             | `type`, `location`, `energy`                          |
| `set_camera_params`        | Sets camera parameters.                | `camera_name`, `focal_length`, `dof_distance`         |

## Additional Available Tools

| Tool Name                | Description                                                      |
|-------------------------|------------------------------------------------------------------|
| `undo_last_action`      | Undo the last action performed in Blender.                       |
| `redo_action`           | Redo the last undone action.                                     |
| `save_scene_variant`    | Save the current scene as a named variant.                       |
| `load_scene_variant`    | Load a previously saved scene variant by name.                   |
| `origin_set`            | Set the origin of an object.                                     |
| `transform_apply`       | Apply location, rotation, and/or scale transforms to an object.  |
| `parent_set`            | Set the parent of an object.                                     |
| `select_all`            | Select or deselect all objects.                                  |
| `snap_selected_to_cursor`| Snap selected objects to the 3D cursor.                         |
| `duplicate_move`        | Duplicate selected objects and move them.                        |
| `primitive_add`         | Add a mesh primitive (cube, sphere, etc.) to the scene.          |
| `get_simple_info`       | Get basic information about the Blender scene.                   |
| `link_blend`            | Link data from another .blend file.                              |
| `find_missing_files`    | Find and report missing files in the current Blender project.     |
| `add_text`              | Add a text object to the scene.                                  |

These tools are in addition to the core and PolyHaven tools listed above, and provide more granular control over your Blender workflow via the MCP protocol.

## Troubleshooting

If you encounter issues:

- Ensure Ollama and the `blender-open-mcp` server are running.
- Check Blender’s add-on settings.
- Verify command-line arguments.
- Refer to logs for error details.

For further assistance, visit the [GitHub Issues](https://github.com/dhakalnirajan/blender-open-mcp/issues) page.

---

Happy Blending with AI! 🚀
