# AirScribe

AirScribe is an innovative application that transforms your webcam into a virtual drawing board. Using hand gestures, you can draw, sketch, and create shapes in real-time, all while enjoying a sci-fi-inspired interface.

## Features

- **Hand Gesture Recognition**: Powered by MediaPipe, the app detects hand landmarks to enable intuitive drawing.
- **Drawing Modes**: Choose between freehand drawing, straight lines, and rectangles.
- **Sci-Fi Background**: A futuristic grid background enhances the drawing experience.
- **Customizable Tools**: Adjust pen thickness and color on the fly.
- **Undo and Clear**: Easily undo the last shape or clear the entire canvas.
- **Save Your Work**: Save your creations as PNG files with a single key press.
- **High Performance**: Supports 720p resolution at 120 FPS for a smooth drawing experience.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ArshaFrn/AirScribe.git
   cd AirScribe
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Usage

- **Start Drawing**: Launch the app and use your index finger to draw on the screen.
- **Modes**:
  - Press `1` for freehand drawing.
  - Press `2` for straight lines.
  - Press `3` for rectangles.
- **Change Color**:
  - Press `r` for red.
  - Press `g` for green.
  - Press `b` for blue.
- **Adjust Thickness**:
  - Press `+` to increase pen thickness.
  - Press `-` to decrease pen thickness.
- **Undo and Clear**:
  - Press `u` to undo the last shape.
  - Press `c` to clear the canvas.
- **Save Drawing**: Press `s` to save your drawing as a PNG file.
- **Exit**: Press `q` to quit the application.

## Important Note

- **Drawing Speed**: For the best experience, draw slowly and steadily. Drawing too fast may result in gaps or cut-off lines due to the frame processing limitations.

## Requirements

- Python 3.7 or higher
- OpenCV
- MediaPipe

## How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks and tracks finger movements.
2. **Drawing Logic**: The app determines whether the user is pinching (drawing) or not.
3. **Rendering**: OpenCV is used to render the drawings and the sci-fi background.

## Technical Details

### Mathematical Concepts

1. **Catmull-Rom Splines**:
   - The application uses Catmull-Rom splines to create smooth curves from a series of points. This is achieved by interpolating between control points using a parametric equation.
   - The formula calculates intermediate points based on the positions of four control points, ensuring smooth transitions.

2. **3D Distance Calculation**:
   - The distance between the index finger tip and thumb tip is calculated in 3D space using the formula:
     \[
     \text{distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}
     \]
   - This distance determines whether the user is "pinching" (drawing) or not.

3. **Smoothing Algorithm**:
   - A simple exponential smoothing technique is applied to stabilize the finger position:
     \[
     \text{smoothed\_pos} = \alpha \cdot \text{raw\_pos} + (1 - \alpha) \cdot \text{previous\_smoothed\_pos}
     \]
   - This reduces jitter and ensures a more stable drawing experience.

### Code Architecture

1. **Main Loop**:
   - The `main()` function contains the primary loop that:
     - Captures frames from the webcam.
     - Processes hand landmarks using MediaPipe.
     - Updates the drawing canvas based on user interactions.

2. **Drawing Modes**:
   - The application supports three modes:
     - **Freehand Drawing**: Points are connected using Catmull-Rom splines.
     - **Straight Lines**: A line is drawn between two points.
     - **Rectangles**: A rectangle is drawn using the start and end points.

3. **State Management**:
   - The app uses counters and thresholds to manage drawing states:
     - `pinch_counter` and `unpinch_counter` ensure stable detection of drawing actions.
     - `hand_lost_counter` handles cases where the hand is temporarily out of view.

4. **Sci-Fi Background**:
   - The `draw_sci_fi_background()` function creates a futuristic grid by drawing vertical and horizontal lines at regular intervals.

5. **Live Feedback**:
   - The app provides real-time feedback by:
     - Highlighting the current finger position.
     - Displaying the current mode and controls on the screen.

6. **Saving Drawings**:
   - Drawings are saved as PNG files with a timestamped filename. Only the drawing canvas (without the live video feed) is saved.

### Libraries Used

- **OpenCV**:
  - Captures webcam input and handles image processing.
  - Provides drawing functions like `line`, `rectangle`, and `polylines`.

- **MediaPipe**:
  - Detects and tracks hand landmarks in real-time.
  - Provides 3D coordinates for each landmark, enabling precise gesture recognition.

- **NumPy**:
  - Used for efficient numerical computations, such as generating evenly spaced points for splines.

### Performance Considerations

- The app processes frames in real-time, ensuring minimal latency.
- Computationally intensive tasks, like spline interpolation, are optimized to maintain smooth performance.

### Future Enhancements

- **Multi-Hand Support**: Extend functionality to support multiple hands.
- **Custom Shapes**: Add more drawing tools, such as circles and polygons.
- **Gesture-Based Controls**: Use additional gestures to switch modes or adjust settings.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking.
- [OpenCV](https://opencv.org/) for image processing.