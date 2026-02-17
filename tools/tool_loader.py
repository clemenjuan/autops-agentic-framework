import importlib
import os
from utils.toon_formatter import ToonFormatter

def load_tools(metadata_path='tools/tools_metadata.toon'):
    if not os.path.isabs(metadata_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metadata_path = os.path.join(base_dir, metadata_path)
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Tools metadata file not found: {metadata_path}")
    
    if not ToonFormatter.is_available():
        raise RuntimeError("TOON library not available. Install with: uv pip install toon-format or run with: uv run")
            
    with open(metadata_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        metadata = ToonFormatter.loads(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse TOON metadata file: {type(e).__name__}: {str(e)}. Ensure TOON library is installed.")
    
    if not isinstance(metadata, dict) or 'tools' not in metadata:
        raise ValueError(f"Invalid metadata structure: expected dict with 'tools' key, got {type(metadata)}")
    
    if not isinstance(metadata['tools'], list):
        raise ValueError(f"Invalid tools structure: expected list, got {type(metadata['tools'])}")
    
    tools = {}
    for tool_def in metadata['tools']:
        try:
            if not isinstance(tool_def, dict):
                print(f"Warning: Skipping invalid tool definition (not a dict): {type(tool_def)}, value: {tool_def}")
                continue
            
            module = importlib.import_module(tool_def['module'])
            func = getattr(module, tool_def['function'])
            tools[tool_def['name']] = {
                'execute': func,
                'description': tool_def['description'],
                'tags': tool_def['tags'],
                'parameters': tool_def['parameters'],
                'examples': tool_def['examples']
            }
        except Exception as e:
            import traceback
            tool_name = tool_def.get('name', 'unknown') if isinstance(tool_def, dict) else str(tool_def)
            error_msg = f"Could not load tool '{tool_name}': {type(e).__name__}: {str(e)}"
            print(f"Warning: {error_msg}")
            print(f"Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}")
            continue
    
    return tools, metadata

