# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Video helper for compressing recordings before upload or model submission."""
import os
import subprocess


class VideoHelper:
    """Compress video using FFmpeg with optional scaling and frame rate adjustment.
    Automatically detects and caches the best available H.265 / H.264 / MPEG-4 encoder
    across system FFmpeg (/usr/bin/ffmpeg) and Conda FFmpeg builds (libx265, libx264,
    libopenh264, or mpeg4).
    """

    def __init__(self):
        self._working_encoder_idx = None

    def _build_encoder_candidates(self, path_ffmpeg_bin: str, crf: int):
        """Return ordered list of (label, ffmpeg_bin, video_codec_args, audio_codec_args)."""
        bins = []
        if os.name != "nt" and os.path.exists("/usr/bin/ffmpeg"):
            bins.append("/usr/bin/ffmpeg")
        if path_ffmpeg_bin not in bins:
            bins.append(path_ffmpeg_bin)

        candidates = []
        for b in bins:
            candidates.extend(
                [
                    (
                        f"{b} [libx265 -crf {crf}]",
                        b,
                        ["-c:v", "libx265", "-crf", str(crf), "-preset", "faster"],
                        ["-c:a", "aac", "-b:a", "96k"],
                    ),
                    (
                        f"{b} [libx264 -crf {crf}]",
                        b,
                        ["-c:v", "libx264", "-crf", str(crf), "-preset", "fast"],
                        ["-c:a", "aac", "-b:a", "96k"],
                    ),
                    (
                        f"{b} [libopenh264 500k]",
                        b,
                        ["-c:v", "libopenh264", "-b:v", "500k"],
                        ["-c:a", "aac", "-b:a", "96k"],
                    ),
                    (
                        f"{b} [mpeg4 -q:v 8]",
                        b,
                        ["-c:v", "mpeg4", "-q:v", "8"],
                        ["-c:a", "aac", "-b:a", "96k"],
                    ),
                ]
            )
        return candidates

    def compress_video(
        self,
        input_path,
        output_path,
        path_ffmpeg_bin="ffmpeg",
        scale_width=None,
        fps=None,
        crf=28,
    ):
        """
        VSCode / Jupyter Notebook compatible FFmpeg compression with automatic codec fallback
        for Conda FFmpeg builds compiled with libopenh264 instead of libx265/libx264.
        """
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        filters = []
        if scale_width is not None:
            filters.append(f"scale={scale_width}:-2")
        if fps is not None:
            filters.append(f"fps={fps}")

        filter_str = ",".join(filters) if filters else "null"
        candidates = self._build_encoder_candidates(path_ffmpeg_bin, crf)

        # Try the cached working encoder first if already discovered
        order = list(range(len(candidates)))
        if self._working_encoder_idx is not None and 0 <= self._working_encoder_idx < len(candidates):
            order.remove(self._working_encoder_idx)
            order.insert(0, self._working_encoder_idx)

        last_err = ""
        for idx in order:
            label, bin_path, v_args, a_args = candidates[idx]
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

            cmd = (
                [bin_path, "-y", "-i", str(input_path), "-vf", filter_str]
                + v_args
                + a_args
                + [str(output_path)]
            )

            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if (
                process.returncode == 0
                and os.path.exists(output_path)
                and os.path.getsize(output_path) > 1024
            ):
                if self._working_encoder_idx != idx:
                    self._working_encoder_idx = idx
                    print(f"[FFmpeg] Selected working encoder: {label}")
                print(f"Compression complete ({label}) -> {output_path}")
                return

            last_err = f"[{label}] {process.stderr}"

        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise RuntimeError(f"All FFmpeg encoder candidates failed. Last error:\n{last_err}")
