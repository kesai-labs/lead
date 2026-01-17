#!/usr/bin/env python3
"""Flask webapp for visualizing driving infractions from CARLA evaluations."""

import json
import subprocess
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# Default evaluation output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "local_evaluation"


def load_infractions_data(infractions_file: Path) -> dict:
    """Load infractions data from JSON file.

    Handles both old format (list) and new format (object with 'infractions' and 'video_fps').

    Args:
        infractions_file: Path to infractions.json file

    Returns:
        Dictionary with 'infractions' (list), 'video_fps' (float), and 'is_legacy_format' (bool)
    """
    with open(infractions_file) as f:
        data = json.load(f)

    # Handle legacy format (just a list)
    if isinstance(data, list):
        return {
            "infractions": data,
            "video_fps": None,  # FPS not available in legacy format
            "is_legacy_format": True,
        }

    # Handle new format (object with infractions and video_fps)
    return {
        "infractions": data.get("infractions", []),
        "video_fps": data.get("video_fps"),
        "is_legacy_format": False,
    }


@app.route("/")
@app.route("/<path:output_dir>")
def index(output_dir=None):
    """Render main dashboard page."""
    return render_template("index.html")


@app.route("/api/routes")
def list_routes():
    """List all available evaluation routes."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    output_path = Path(output_dir)

    if not output_path.exists():
        return jsonify({"error": "Directory not found", "routes": []}), 404

    routes = []
    for route_dir in sorted(output_path.iterdir()):
        if route_dir.is_dir():
            route_name = route_dir.name
            # Check if it has the expected files (with route name prefix)
            has_infractions = (route_dir / "infractions.json").exists()
            has_debug = (route_dir / f"{route_name}_debug.mp4").exists()
            has_demo = (route_dir / f"{route_name}_demo.mp4").exists()
            has_checkpoint = (route_dir / "checkpoint_endpoint.json").exists()

            if has_infractions or has_debug or has_demo:
                route_info = {
                    "name": route_name,
                    "path": str(route_dir),
                    "has_infractions": has_infractions,
                    "has_debug_video": has_debug,
                    "has_demo_video": has_demo,
                    "has_checkpoint": has_checkpoint,
                }

                # Load infraction count (excluding min speed and completion)
                if has_infractions:
                    try:
                        infraction_data = load_infractions_data(route_dir / "infractions.json")
                        infractions = infraction_data["infractions"]
                        # Filter out min speed and completion infractions
                        filtered_infractions = [
                            inf
                            for inf in infractions
                            if "minspeed" not in inf.get("infraction", "").lower()
                            and "completion" not in inf.get("infraction", "").lower()
                        ]
                        route_info["infraction_count"] = len(filtered_infractions)
                        route_info["video_fps"] = infraction_data["video_fps"]
                    except Exception:
                        route_info["infraction_count"] = 0
                        route_info["video_fps"] = None

                routes.append(route_info)

    return jsonify({"routes": routes, "output_dir": str(output_path)})


@app.route("/api/route/<path:route_name>/infractions")
def get_infractions(route_name):
    """Get infractions for a specific route."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    route_path = Path(output_dir) / route_name
    infractions_file = route_path / "infractions.json"

    if not infractions_file.exists():
        return jsonify({"error": "Infractions file not found", "infractions": [], "video_fps": None}), 404

    try:
        infraction_data = load_infractions_data(infractions_file)
        return jsonify(
            {
                "infractions": infraction_data["infractions"],
                "video_fps": infraction_data["video_fps"],
                "is_legacy_format": infraction_data["is_legacy_format"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "infractions": [], "video_fps": None}), 500


@app.route("/api/route/<path:route_name>/checkpoint")
def get_checkpoint(route_name):
    """Get checkpoint data for a specific route."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    route_path = Path(output_dir) / route_name
    checkpoint_file = route_path / "checkpoint_endpoint.json"

    if not checkpoint_file.exists():
        return jsonify({"error": "Checkpoint file not found"}), 404

    try:
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        return jsonify(checkpoint)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/route/<path:route_name>/video_info")
def get_video_info(route_name):
    """Get video metadata including FPS from infractions.json."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    route_path = Path(output_dir) / route_name
    infractions_file = route_path / "infractions.json"

    if not infractions_file.exists():
        return jsonify({"error": "Infractions file not found", "video_fps": None}), 404

    try:
        infraction_data = load_infractions_data(infractions_file)
        return jsonify(
            {
                "video_fps": infraction_data["video_fps"],
                "is_legacy_format": infraction_data["is_legacy_format"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "video_fps": None}), 500


@app.route("/video/<path:route_name>/<video_type>")
def serve_video(route_name, video_type):
    """Serve video file for a specific route with range request support."""
    output_dir = request.args.get("dir", DEFAULT_OUTPUT_DIR)
    route_path = Path(output_dir) / route_name

    # Video files are named with route prefix, e.g., 23687_debug.mp4
    video_files = {
        "debug": f"{route_name}_debug.mp4",
        "demo": f"{route_name}_demo.mp4",
    }

    if video_type not in video_files:
        return "Invalid video type", 404

    video_file = video_files[video_type]
    video_path = route_path / video_file

    if not video_path.exists():
        return f"Video file {video_file} not found", 404

    # Support range requests for video seeking

    from flask import Response

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range", None)

    if range_header:
        byte_start, byte_end = 0, None
        match = range_header.replace("bytes=", "").split("-")
        byte_start = int(match[0])
        byte_end = int(match[1]) if match[1] else file_size - 1

        length = byte_end - byte_start + 1

        with open(video_path, "rb") as f:
            f.seek(byte_start)
            data = f.read(length)

        response = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
        response.headers.add("Content-Range", f"bytes {byte_start}-{byte_end}/{file_size}")
        response.headers.add("Accept-Ranges", "bytes")
        response.headers.add("Content-Length", str(length))
        return response

    # Return full video if no range requested
    return send_from_directory(route_path, video_file, mimetype="video/mp4")


@app.route("/api/open_directory", methods=["POST"])
def open_directory():
    """Open the directory containing the route's videos in the file manager."""
    data = request.json
    route_name = data.get("route_name")
    output_dir = data.get("output_dir", DEFAULT_OUTPUT_DIR)

    if not route_name:
        return jsonify({"error": "Missing route_name parameter"}), 400

    route_path = Path(output_dir) / route_name

    if not route_path.exists():
        return jsonify({"error": f"Directory not found: {route_path}"}), 404

    try:
        # Open directory in file manager (works on Linux, Mac, Windows)
        import platform

        system = platform.system()

        if system == "Linux":
            subprocess.run(["xdg-open", str(route_path)], check=True)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(route_path)], check=True)
        elif system == "Windows":
            subprocess.run(["explorer", str(route_path)], check=True)
        else:
            return jsonify({"error": f"Unsupported operating system: {system}"}), 400

        return jsonify({"success": True, "path": str(route_path)})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to open directory: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cut_video", methods=["POST"])
def cut_video():
    """Cut a video segment around an infraction timestamp."""
    data = request.json
    route_name = data.get("route_name")
    video_type = data.get("video_type")
    timestamp = data.get("timestamp")
    buffer_seconds = data.get("buffer", 3)
    infraction_number = data.get("infraction_number")
    output_dir = data.get("output_dir", DEFAULT_OUTPUT_DIR)

    if not all([route_name, video_type, timestamp is not None, infraction_number]):
        return jsonify({"error": "Missing required parameters"}), 400

    route_path = Path(output_dir) / route_name

    # Get input video file
    video_files = {
        "debug": f"{route_name}_debug.mp4",
        "demo": f"{route_name}_demo.mp4",
    }

    if video_type not in video_files:
        return jsonify({"error": "Invalid video type"}), 400

    input_video = route_path / video_files[video_type]

    if not input_video.exists():
        return jsonify({"error": f"Video file {video_files[video_type]} not found"}), 404

    # Calculate start and duration
    start_time = max(0, timestamp - buffer_seconds)
    duration = buffer_seconds * 2  # buffer before + buffer after

    # Output file name
    output_filename = f"{route_name}_{video_type}_infraction_{infraction_number}.mp4"
    output_path = route_path / output_filename

    try:
        # Use ffmpeg to cut the video
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if exists
            "-ss",
            str(start_time),  # Start time
            "-i",
            str(input_video),  # Input file
            "-t",
            str(duration),  # Duration
            "-c",
            "copy",  # Copy codec (fast, no re-encoding)
            str(output_path),  # Output file
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        return jsonify(
            {
                "success": True,
                "output_path": str(output_path),
                "start_time": start_time,
                "duration": duration,
                "buffer": buffer_seconds,
            }
        )

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg error: {e.stderr}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Infraction Dashboard...")
    print(f"Default output directory: {DEFAULT_OUTPUT_DIR}")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, host="0.0.0.0")
