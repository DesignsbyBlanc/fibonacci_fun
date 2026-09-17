#!/usr/bin/env python3
"""Detect a petal center and overlay a golden Fibonacci spiral."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_flower_center(image: np.ndarray) -> tuple[tuple[float, float], float]:
    """Find the green/yellow flower disk and return its center and radius."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 45, 35], dtype=np.uint8)
    upper = np.array([95, 255, 230], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(
            "Could not detect the flower center; use --center to specify it.")

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < image.shape[0] * image.shape[1] * 0.01:
        raise ValueError(
            "Detected center is too small; use --center to specify it.")

    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    return (center_x, center_y), radius


def detect_wooden_petal_center(image: np.ndarray) -> tuple[tuple[float, float], float]:
    """Estimate the wooden petal center from its warm brown/orange surface."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 45, 20], dtype=np.uint8)
    upper = np.array([28, 255, 235], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))

    height, width = image.shape[:2]
    roi = np.zeros_like(mask)
    cv2.ellipse(
        roi,
        (width // 2, height // 2),
        (int(width * 0.38), int(height * 0.43)),
        0,
        0,
        360,
        255,
        -1,
    )
    mask = cv2.bitwise_and(mask, roi)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(
            "Could not detect the wooden petal; use --center to specify it.")

    image_center = np.array([width / 2, height / 2])

    def contour_score(contour: np.ndarray) -> float:
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return 0
        centroid = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        distance = np.linalg.norm(centroid - image_center)
        return area / (1 + distance)

    contour = max(contours, key=contour_score)
    moments = cv2.moments(contour)
    center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    _, radius = cv2.minEnclosingCircle(contour)
    object_radius = max(radius * 1.7, min(width, height) * 0.25)
    object_radius = min(object_radius, min(width, height) * 0.38)
    return center, object_radius


def fibonacci_points(
    center: tuple[float, float],
    radius: float,
    turns: float = 2.5,
    samples: int = 900,
) -> np.ndarray:
    """Build points for a logarithmic spiral whose growth ratio is phi."""
    phi = (1 + np.sqrt(5)) / 2
    angles = np.linspace(0, turns * 2 * np.pi, samples)
    growth = np.log(phi) / (np.pi / 2)
    spiral_radius = radius * 0.035 * np.exp(growth * angles)
    spiral_radius = np.minimum(spiral_radius, radius * 1.45)
    points = np.column_stack(
        (
            center[0] + spiral_radius * np.cos(angles),
            center[1] + spiral_radius * np.sin(angles),
        )
    )
    return np.round(points).astype(np.int32).reshape((-1, 1, 2))


def draw_spiral(
    image: np.ndarray,
    center: tuple[float, float],
    radius: float,
    turns: float,
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    overlay = image.copy()
    points = fibonacci_points(center, radius, turns)
    cv2.polylines(overlay, [points], False, color, thickness, cv2.LINE_AA)
    cv2.circle(overlay, np.round(center).astype(
        int), 5, color, -1, cv2.LINE_AA)
    return overlay


def parse_center(value: str) -> tuple[float, float]:
    try:
        x, y = (float(part.strip()) for part in value.split(","))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "center must be written as x,y") from error
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input flower image")
    parser.add_argument("-o", "--output", type=Path, help="Output image path")
    parser.add_argument("--center", type=parse_center,
                        help="Manual center as x,y")
    parser.add_argument("--mode", choices=("flower", "wooden"), default="wooden",
                        help="Object color model to use for automatic detection")
    parser.add_argument("--turns", type=float, default=2.5,
                        help="Number of spiral turns")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Spiral line thickness")
    args = parser.parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise SystemExit(f"Could not read image: {args.input}")

    if args.center is None:
        try:
            detector = (detect_wooden_petal_center
                        if args.mode == "wooden" else detect_flower_center)
            center, radius = detector(image)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        center = args.center
        radius = min(image.shape[:2]) * 0.2

    result = draw_spiral(image, center, radius, args.turns,
                         (40, 35, 220), args.thickness)
    output = args.output or args.input.with_name(
        f"{args.input.stem}_spiral{args.input.suffix}")
    if not cv2.imwrite(str(output), result):
        raise SystemExit(f"Could not write output: {output}")
    print(
        f"Detected center: ({center[0]:.1f}, {center[1]:.1f}), radius: {radius:.1f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
