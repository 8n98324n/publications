"""Silero v6 probabilities on 512-sample blocks with 64-sample context."""
from pathlib import Path
import numpy as np
RATE=16000
BLOCK=512
CONTEXT=64
INFERENCE_BATCH=1024

def make_session(ort,model_path):
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"], sess_options=options)
    got = session.get_session_options()
    assert got.inter_op_num_threads == got.intra_op_num_threads == 1
    assert got.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert [entry.name for entry in session.get_inputs()] == ["input", "h", "c"]
    return session

def probabilities(audio, session, np, batch=INFERENCE_BATCH):
    if audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Decoded audio is empty, non-mono, or non-finite")
    padding = (-len(audio)) % BLOCK
    padded = np.pad(audio, (0, padding)) if padding else audio
    windows = padded.reshape(-1, BLOCK)
    # Separate context storage avoids mutating the last 64 real audio samples.
    # Adapted from faster-whisper's Silero inference loop (MIT; see notices).
    # Separate context preserves the real samples at the terminal boundary.
    context = np.zeros((len(windows), CONTEXT), dtype=np.float32)
    context[1:] = windows[:-1, -CONTEXT:]
    inputs = np.concatenate((context, windows), axis=1)
    h = np.zeros((1, 1, 128), dtype=np.float32)
    c = np.zeros((1, 1, 128), dtype=np.float32)
    outputs = []
    for start in range(0, len(windows), batch):
        output, h, c = session.run(None, {"input": inputs[start:start + batch], "h": h, "c": c})
        outputs.append(output.reshape(-1))
    result = np.concatenate(outputs)
    if len(result) != len(windows) or not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError("VAD probability range, shape, or finiteness failed")
    return result, padding
