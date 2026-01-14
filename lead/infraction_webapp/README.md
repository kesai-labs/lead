# LEAD Infraction Dashboard

A web-based viewer for analyzing driving infractions from CARLA evaluations.

## Features

- **Video Player** - Watch debug/demo videos with enhanced controls
- **Infraction List** - Scrollable list of all detected infractions
- **Jump to Timestamp** - Click any infraction to jump to that moment in the video
- **Playback Controls** - Speed control (0.5x, 1x, 2x) and frame-by-frame navigation
- **Filter** - Search and filter infractions by type
- **Route Stats** - View driving scores and metrics
- **Keyboard Shortcuts** - Space to play/pause, arrows to seek

## Quick Start

### 1. Install Flask

```bash
pip install flask
```

### 2. Run the Dashboard

```bash
python lead/infraction_webapp/app.py
```

### 3. Open in Browser

Navigate to `http://localhost:5000`

## Usage

1. **Click "Load Routes"** to scan the default output directory (`outputs/local_evaluation/`)
2. **Select a route** from the sidebar to view its infractions
3. **Click any infraction** in the list to jump to that timestamp in the video
4. Use **video controls** for playback speed and frame-by-frame navigation

## Keyboard Shortcuts

- `Space` - Play/Pause
- `←` - Back 1 second
- `→` - Forward 1 second
- `Shift + ←` - Back 5 seconds
- `Shift + →` - Forward 5 seconds

## Custom Output Directory

To use a different evaluation directory, enter the path in the header input field and click "Load Routes".

## Requirements

- Flask
- Browser with HTML5 video support
- Evaluation data with `infractions.json` and video files
