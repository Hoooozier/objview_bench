"""Protocol example only, not a competitive reconstruction algorithm."""
import json
import os
from pathlib import Path
import time

import httpx

from submission_client import SubmissionClient, SubmissionError


def submit_with_retry(client, action):
    # Keep the SAME request_id and payload after an ambiguous network failure.
    for attempt in range(3):
        try:
            return client.submit(action)
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(.25)


def main():
    client = SubmissionClient(os.environ['BENCHMARK_URL'], os.environ['OBJVIEW_EPISODE_TOKEN'])
    observations = Path(os.environ.get('OBSERVATION_ROOT', '/observations')).resolve()
    try:
        # Load model/weights BEFORE ready(). This interval is separately limited.
        client.ready()
        config = client.episode()
        after = -1
        while True:
            step = client.next_step(after)
            if step is None:
                continue
            if step.get('status') == 'done':
                return 0
            # RGB: uint8 PNG; depth: NPZ key 'depth', float32 camera-Z;
            # mask: PNG 0/255. A real algorithm loads these for inference.
            metadata_path = observations / step['observation']['frame_meta_path']
            metadata = json.loads(metadata_path.read_text())
            assert metadata['source'] == 'released_observation'
            candidates = [[3., 0., 0., 0., 0., 0., 0.], [0., 3., 0., 0., 0., 0., 0.]]
            candidates = [pose for pose in candidates if pose != step['current_pose']]
            feasible = client.feasibility(candidates) if candidates else []
            pose = next((p for p, ok in zip(candidates, feasible) if ok), None)
            if step['step_index'] == 0 and config['max_visited_view_num'] > 1 and pose:
                action = client.make_action(step, 'move', pose=pose)
            else:
                action = client.make_action(step, 'stop', stop_reason='plan_end')
            submit_with_retry(client, action)
            after = step['step_index']
    finally:
        client.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (SubmissionError, httpx.TransportError):
        # Never export raw benchmark responses, tokens or observations through logs.
        raise SystemExit(2)
