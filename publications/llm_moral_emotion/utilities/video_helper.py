# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Video helper for compressing recordings before upload or model submission."""
import os
import subprocess

class VideoHelper:

    """Compress video using FFmpeg with optional scaling and frame rate adjustment, creating the output directory if needed and applying HEVC encoding with configurable quality.
    """
    def __init__(self):
        pass

    # Utility function: compress video

    def compress_video(self, input_path, output_path, path_ffmpeg_bin,  scale_width=None, fps=None, crf=23):
        """
        VSCode Notebook compatible FFmpeg compression with full debug output.

        crf:  range: 0 - 51,     default: 23,     lower value = better quality, larger file,  higher value = worse quality, smaller file
        """

        # Ensure output folder exists
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        filters = []
        if scale_width is not None:
            filters.append(f"scale={scale_width}:-1")
        if fps is not None:
            filters.append(f"fps={fps}")

        filter_str = ",".join(filters) if filters else "null"
        
        cmd = [
            path_ffmpeg_bin,
            "-y",
            "-i", input_path,
            "-vf", filter_str,
            "-vcodec", "libx265",
            "-crf", str(crf),
            "-preset", "faster",
            "-c:a", "copy",    # ← Keep audio
            output_path
        ]

        print("Running FFmpeg command:")
        print(" ".join(cmd))

        # Run and capture output in Jupyter
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Raise error if FFmpeg failed
        if process.returncode != 0:
            raise RuntimeError("FFmpeg failed. See above logs.")

        # Verify output file
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not created: {output_path}")

        print(f"Compression complete → {output_path}")
