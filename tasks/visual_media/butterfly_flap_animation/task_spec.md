# Butterfly Flap Animation Task

## Goal

Create a seamless animation of the provided butterfly image flapping its wings while
moving along a smooth curved path in a 2.5D-style composition. You may use any software.

## Input Asset

- `input/butterfly.jpeg`: the source butterfly artwork. Reuse this visual texture in the
  final animation.

## Required Animation

1. Split or mask the butterfly into a central body and left/right wings.
2. Set wing anchor points near the wing-body joints.
3. Animate the wings with repeated Y-rotation or an equivalent perspective deformation.
4. Produce exactly 4 complete wing flap cycles. One complete cycle is open -> closed -> open.
5. Make the butterfly move along a smooth curved path through the frame.
6. The trajectory should include at least 2 Y-axis turning points.
7. Avoid abrupt jumps in position or velocity.
8. Keep the butterfly texture recognizably identical to the source image, especially on open-wing frames.
9. Use a plain dark or black background so the butterfly motion remains clear.

## Export Requirements

- Export one H.264 MP4.
- Frame rate: 30 fps.
- Duration: 4.0 to 5.5 seconds.
- Output path: `output/output.mp4`.
