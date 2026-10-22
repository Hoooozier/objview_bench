"""Run only explicitly trusted smoke images. Untrusted submission execution is disabled."""
import argparse
import json
import math
import os
from pathlib import Path
import secrets
import time

from .docker_runtime import DockerRuntime, LABEL, RuntimeFailure, container_args


class JobFailure(RuntimeError):
    pass


def validate_summary(summary, job_id):
    if (summary.get('mode') != 'smoke_test' or summary.get('status') != 'completed'
            or summary.get('job_id') != job_id or summary.get('visited_view_num') != 2
            or any(summary.get(key) is not False for key in
                   ('official_result', 'leaderboard_eligible', 'algorithm_correctness_checked'))):
        raise JobFailure('INVALID_SUMMARY')
    timing = summary.get('official_timing', {})
    steps = timing.get('steps')
    total = timing.get('total_decision_wall_time_ns')
    if (timing.get('timing_source') != 'benchmark_monotonic_clock' or type(total) is not int
            or total < 0 or not isinstance(steps, list) or len(steps) != 2):
        raise JobFailure('INVALID_SUMMARY')
    for index, step in enumerate(steps):
        if (not isinstance(step, dict) or step.get('step_index') != index
                or type(step.get('decision_wall_time_ns')) is not int or step['decision_wall_time_ns'] < 0):
            raise JobFailure('INVALID_SUMMARY')
    if sum(step['decision_wall_time_ns'] for step in steps) != total:
        raise JobFailure('INVALID_SUMMARY')
    return {'timing_source': 'benchmark_monotonic_clock', 'total_decision_wall_time_ns': total,
            'steps': [{'step_index': index, 'decision_wall_time_ns': step['decision_wall_time_ns']}
                      for index, step in enumerate(steps)]}


def run_smoke(runtime, benchmark_image, algorithm_image, output_dir, *, trusted_smoke=False,
              timeout_sec=30, poll_sec=.1):
    if trusted_smoke is not True:
        raise ValueError('Only explicit trusted smoke runs are enabled')
    if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 3600 or not 0 < poll_sec <= 1:
        raise ValueError('Invalid timeout')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    job_id = secrets.token_hex(16)
    prefix = f'ovb-{job_id}'
    names = {role: f'{prefix}-{role}' for role in ('init', 'benchmark', 'algorithm', 'net', 'data')}
    env_file = output_dir / 'episode.env'
    summary_file = output_dir / 'summary.json'
    report_file = output_dir / 'report.json'
    started = time.perf_counter_ns()
    deadline = time.monotonic() + timeout_sec
    resources = []
    report = {'job_id': job_id, 'schema_version': 'objview-worker-report/v1', 'mode': 'smoke_test',
              'official_result': False, 'leaderboard_eligible': False, 'algorithm_correctness_checked': False,
              'status': 'PREPARING', 'states': ['PREPARING'], 'failure_code': None, 'timing': {},
              'cleanup': {'complete': False, 'unconfirmed_resources': []},
              'not_verified': ['untrusted_code_sandbox', 'egress_canaries', 'disk_quota', 'gpu',
                               'algorithm_quality', 'website', 'queue', 'leaderboard']}

    def save():
        # Controller-owned directory. Atomic replacement prevents partial success JSON.
        temporary = report_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(report_file)

    def transition(state):
        report['status'] = state
        report['states'].append(state)
        save()

    def remaining_budget():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise JobFailure('JOB_TIMEOUT')
        return min(10, remaining)

    def call(args):
        return runtime.call(args, timeout=remaining_budget())

    def state(role):
        data = json.loads(call(['container', 'inspect', names[role]]))
        return data[0]['State']

    def await_exit(role, failure_code):
        while True:
            current = state(role)
            if not current['Running']:
                if current.get('OOMKilled'):
                    raise JobFailure('ALGORITHM_OUT_OF_MEMORY' if role == 'algorithm' else failure_code)
                if current['ExitCode'] != 0:
                    raise JobFailure(failure_code)
                return
            if role == 'algorithm':
                benchmark_state = state('benchmark')
                if not benchmark_state['Running'] and benchmark_state['ExitCode'] != 0:
                    raise JobFailure('BENCHMARK_FAILED')
            time.sleep(min(poll_sec, max(0, deadline - time.monotonic())))

    def create(role, image):
        # Record intent first: a lost CLI response must not orphan a created resource.
        resources.append(('container', names[role]))
        call(container_args(name=names[role], image_id=image, network=names['net'], volume=names['data'],
                            role=role, job_id=job_id, token_file=env_file))

    save()
    try:
        report['docker_host'] = runtime.preflight(timeout=remaining_budget())
        images = {'benchmark': runtime.resolve_image(benchmark_image, 'benchmark', timeout=remaining_budget()),
                  'algorithm': runtime.resolve_image(algorithm_image, 'algorithm', timeout=remaining_budget())}
        report['image_ids'] = images
        fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(f'OBJVIEW_EPISODE_TOKEN={secrets.token_hex(32)}\nOBJVIEW_JOB_ID={job_id}\n')
        resources.append(('network', names['net']))
        call(['network', 'create', '--internal', '--label', f'{LABEL}={job_id}', names['net']])
        resources.append(('volume', names['data']))
        call(['volume', 'create', '--label', f'{LABEL}={job_id}', names['data']])
        create('init', images['benchmark'])
        call(['start', names['init']])
        await_exit('init', 'INFRASTRUCTURE_FAILED')
        create('benchmark', images['benchmark'])
        transition('STARTING')
        call(['start', names['benchmark']])
        # Ready file is created only after Uvicorn listens, never before app setup.
        while True:
            current = state('benchmark')
            if not current['Running']:
                raise JobFailure('BENCHMARK_FAILED')
            ready = call(['exec', names['benchmark'], 'python', '-c',
                          "from pathlib import Path; print('ready' if Path('/episode/private/gateway.ready').is_file() else 'waiting')"])
            if ready == 'ready':
                break
            time.sleep(poll_sec)
        create('algorithm', images['algorithm'])
        startup = time.perf_counter_ns()
        call(['start', names['algorithm']])
        report['timing']['algorithm_start_call_ns'] = time.perf_counter_ns() - startup
        # docker start return is NOT readiness/model loading time.
        transition('RUNNING')
        await_exit('algorithm', 'ALGORITHM_CRASHED')
        report['timing']['algorithm_container_wall_time_ns'] = time.perf_counter_ns() - startup
        await_exit('benchmark', 'BENCHMARK_FAILED')
        call(['cp', f"{names['benchmark']}:/episode/private/summary.json", str(summary_file)])
        if summary_file.stat().st_size > 1024 * 1024:
            raise JobFailure('INVALID_SUMMARY')
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        report['benchmark_timing'] = validate_summary(summary, job_id)
    except JobFailure as exc:
        report['failure_code'] = str(exc)
    except RuntimeFailure as exc:
        report['failure_code'] = 'JOB_TIMEOUT' if time.monotonic() >= deadline else 'INFRASTRUCTURE_FAILED'
        report['runtime_code'] = exc.code
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, OSError):
        report['failure_code'] = 'INVALID_SUMMARY' if summary_file.exists() else 'INFRASTRUCTURE_FAILED'
    except BaseException:
        report['failure_code'] = 'INTERRUPTED'
        raise
    finally:
        unconfirmed = []
        # Containers must go before network/volume, irrespective of intent order.
        ordered = [r for r in reversed(resources) if r[0] == 'container']
        ordered += [r for r in resources if r[0] == 'network']
        ordered += [r for r in resources if r[0] == 'volume']
        for kind, name in ordered:
            try:
                record = runtime.lookup(kind, name, timeout=5)
                if record is None:
                    continue
                labels = (record.get('Config') or {}).get('Labels') if kind == 'container' else record.get('Labels')
                if (labels or {}).get(LABEL) != job_id:
                    unconfirmed.append(name)
                    continue
                args = ['rm', '--force', '--volumes', name] if kind == 'container' else [kind, 'rm', name]
                runtime.call(args, timeout=5)
                if runtime.lookup(kind, name, timeout=5) is not None:
                    unconfirmed.append(name)
            except (RuntimeFailure, ValueError, TypeError, OSError):
                unconfirmed.append(name)
        for path in (env_file, summary_file):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                unconfirmed.append(path.name)
        report['cleanup'] = {'complete': not unconfirmed, 'unconfirmed_resources': unconfirmed}
        if unconfirmed and report['failure_code'] is None:
            report['failure_code'] = 'CLEANUP_FAILED'
        report['timing']['total_job_wall_time_ns'] = time.perf_counter_ns() - started
        transition('FAILED' if report['failure_code'] else 'SUCCEEDED')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trusted-smoke', action='store_true', required=True,
                        help='Only repository-controlled development images; not uploaded code')
    parser.add_argument('--benchmark-image', required=True)
    parser.add_argument('--algorithm-image', required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New private job directory')
    parser.add_argument('--docker-host', required=True, help='Explicit local unix:///... Docker socket')
    parser.add_argument('--timeout-sec', type=float, default=30)
    args = parser.parse_args()
    runtime = DockerRuntime(args.docker_host)
    result = run_smoke(runtime, args.benchmark_image, args.algorithm_image, args.output_dir,
                       trusted_smoke=args.trusted_smoke, timeout_sec=args.timeout_sec)
    print(json.dumps({'status': result['status'], 'failure_code': result['failure_code'],
                      'report': str(args.output_dir / 'report.json')}))
    return 0 if result['status'] == 'SUCCEEDED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
