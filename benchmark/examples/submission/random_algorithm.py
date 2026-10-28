"""Trusted development dummy that chooses a deterministic random viewpoint.

The seed is operator-controlled through ``OBJVIEW_RANDOM_SEED``.  The
benchmark still measures and owns the episode result; this process never
reports a score or timing.
"""
import json
import os
import random
import time

import httpx

from submission_client import SubmissionClient, SubmissionError


def submit_with_retry(client, action):
    for attempt in range(3):
        try:
            return client.submit(action)
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(.1)


def main():
    client = SubmissionClient(os.environ['BENCHMARK_URL'], os.environ['OBJVIEW_EPISODE_TOKEN'])
    rng = random.Random(os.environ.get('OBJVIEW_RANDOM_SEED', 'trusted-smoke'))
    try:
        client.ready()
        config = client.episode()
        after = -1
        while True:
            step = client.next_step(after)
            if step is None:
                continue
            if step.get('status') == 'done':
                return 0
            metadata = json.loads(client.observation(step, 'frame_meta'))
            if metadata.get('source') != 'released_observation':
                return 2
            candidates = [[3., 0., 0., 0., 0., 0., 0.],
                          [0., 3., 0., 0., 0., 0., 0.]]
            rng.shuffle(candidates)
            candidates = [pose for pose in candidates if pose != step['current_pose']]
            feasible = client.feasibility(candidates) if candidates else []
            pose = next((candidate for candidate, ok in zip(candidates, feasible) if ok), None)
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
        raise SystemExit(2)
