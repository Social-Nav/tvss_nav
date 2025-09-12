import re
import json
from typing import List, Dict, Optional


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extract the first JSON object from the given text.
    Supports both markdown code blocks (```json ... ```) and plain JSON.

    Parameters:
        text (str): The text containing JSON, either in markdown format or plain JSON.

    Returns:
        str: The JSON string if found, else None.
    """
    try:
        # First try to find JSON in markdown code block
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json_str
        
        # If no markdown block found, try to find plain JSON object
        match = re.search(r"(\{.*?\})", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            # Validate that it's actually JSON
            json.loads(json_str)
            return json_str
            
        return None
    except json.JSONDecodeError as e:
        print(f"[extract_json_from_markdown] Failed to decode JSON: {e}")
    return None

def parse_tool_calls(response_json: str) -> Optional[List[Dict]]:
    """
    Parse the tool_calls field from a JSON string returned by the language model.

    Parameters:
        response_json (str): The full JSON response as a string.

    Returns:
        List[Dict]: A list of tool call dictionaries, each with 'tool' and 'args'.
    """
    if not response_json:
        print("Empty response JSON")
        return None
    try:
        data = json.loads(response_json)
        tool_calls = data.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise ValueError("'tool_calls' must be a list")
        return tool_calls
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
    except Exception as e:
        print(f"Unexpected error while parsing tool_calls: {e}")
    return None


def extract_scene_description(response_json: str) -> Optional[str]:
    """
    Extract the scene description from the JSON response.

    Parameters:
        response_json (str): The full JSON response as a string.

    Returns:
        str: The scene description text.
    """
    try:
        data = json.loads(response_json)
        description = data.get("description", None)
        return description
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
    return None


def extract_social_objects(response_json: str) -> Optional[List[str]]:
    """
    Extract the list of relevant social navigation objects from the JSON response.

    Parameters:
        response_json (str): The full JSON response as a string.

    Returns:
        List[str]: A list of object names.
    """
    try:
        data = json.loads(response_json)
        objects = data.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("'objects' must be a list")
        return objects
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
    except Exception as e:
        print(f"Unexpected error while parsing objects: {e}")
    return None