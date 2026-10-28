import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from submission_worker.docker_runtime import DockerRuntime, RuntimeFailure, container_args

IMAGE = 'sha256:' + 'a' * 64
JOB = 'b' * 32


class RuntimeTests(unittest.TestCase):
    def arguments(self, role='algorithm', **changes):
        values = dict(name=f'ovb-{JOB}-{role}', image_id=IMAGE, network=f'ovb-{JOB}-net',
                      volume=f'ovb-{JOB}-data', role=role, job_id=JOB, token_file=Path('/tmp/job.env'))
        values.update(changes)
        return container_args(**values)

    def test_algorithm_has_no_benchmark_or_observation_mount(self):
        args = self.arguments()
        self.assertIn('--read-only', args)
        self.assertIn('65532:65532', args)
        self.assertIn('ALL', args)
        self.assertIn('no-new-privileges', args)
        self.assertNotIn('--mount', args)
        self.assertNotIn('--volume', args)
        self.assertNotIn('OBSERVATION_ROOT=', ' '.join(args))
        for forbidden in ('--privileged', '--publish', '-p', '--device', '--gpus', '--pid', '--ipc'):
            self.assertNotIn(forbidden, args)
        self.assertEqual(args[-1], IMAGE)
        for flag in ('--cpus', '--memory', '--memory-swap', '--pids-limit', '--tmpfs', '--log-driver'):
            self.assertIn(flag, args)

    def test_benchmark_has_private_volume_but_no_host_mount(self):
        args = self.arguments('benchmark')
        mount = args[args.index('--mount') + 1]
        self.assertIn('target=/episode', mount)
        self.assertNotIn('volume-subpath', mount)
        self.assertNotIn('docker.sock', ' '.join(args))

    def test_malformed_arguments_fail_before_docker(self):
        for changes in ({'image_id': 'ubuntu:latest'}, {'name': '--privileged'},
                        {'network': 'host'}, {'volume': '/home/huhao'},
                        {'role': 'uploaded'}, {'job_id': 'x\nTOKEN=secret'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.arguments(**changes)

    def test_remote_daemon_is_rejected(self):
        for endpoint in ('tcp://127.0.0.1:2375', 'ssh://host', 'unix://relative'):
            with self.assertRaises(ValueError):
                DockerRuntime(endpoint)

    def test_runtime_uses_argv_and_sanitizes_error(self):
        calls = []
        def execute(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 1, '', 'SECRET raw docker diagnostic')
        runtime = DockerRuntime('unix:///tmp/docker.sock', execute=execute)
        with self.assertRaises(RuntimeFailure) as caught:
            runtime.call(['version'])
        self.assertNotIn('SECRET', str(caught.exception))
        self.assertEqual(calls[0][0], ['docker', '--host', 'unix:///tmp/docker.sock', 'version'])
        self.assertFalse(calls[0][1]['shell'])
        self.assertNotIn('DOCKER_CONTEXT', calls[0][1]['env'])

    def test_image_is_resolved_to_immutable_id_and_role_checked(self):
        def execute(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, json.dumps([{
                'Id': IMAGE, 'Config': {'Labels': {'org.objview.smoke.role': 'algorithm'}}}]), '')
        runtime = DockerRuntime('unix:///tmp/docker.sock', execute=execute)
        self.assertEqual(runtime.resolve_image('objview-dummy:dev', 'algorithm'), IMAGE)
        with self.assertRaises(RuntimeFailure):
            runtime.resolve_image('objview-dummy:dev', 'benchmark')

    def test_command_timeout_and_resource_warning_fail_closed(self):
        def timeout(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 1)
        runtime = DockerRuntime('unix:///tmp/docker.sock', execute=timeout)
        with self.assertRaises(RuntimeFailure) as caught:
            runtime.call(['info'])
        self.assertEqual(caught.exception.code, 'DOCKER_COMMAND_TIMEOUT')
        runtime.execute = lambda argv, **kw: subprocess.CompletedProcess(argv, 0, '', 'WARNING: memory limit ignored')
        with self.assertRaises(RuntimeFailure):
            runtime.call(['info'])

    def test_preflight_rejects_missing_resource_controls(self):
        info = {'OSType': 'linux', 'CgroupDriver': 'systemd', 'CgroupVersion': '2',
                'MemoryLimit': True, 'PidsLimit': True, 'ServerVersion': 'test',
                'SecurityOptions': ['name=rootless']}
        def execute(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, json.dumps(info), '')
        runtime = DockerRuntime('unix:///tmp/docker.sock', execute=execute)
        self.assertEqual(runtime.preflight()['cgroup_version'], '2')
        for key, value in [('MemoryLimit', False), ('PidsLimit', False), ('CgroupDriver', 'none'),
                           ('CgroupVersion', '1')]:
            original = info[key]
            info[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeFailure):
                runtime.preflight()
            info[key] = original

    def test_lookup_does_not_treat_similar_resource_name_as_owned_target(self):
        def execute(argv, **kwargs):
            self.assertIn('ls', argv)
            return subprocess.CompletedProcess(argv, 0, f'ovb-{JOB}-algorithm-other\n', '')
        runtime = DockerRuntime('unix:///tmp/docker.sock', execute=execute)
        self.assertIsNone(runtime.lookup('container', f'ovb-{JOB}-algorithm'))

    def test_declared_image_volumes_are_rejected(self):
        def execute(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, json.dumps([{'Id': IMAGE, 'Config': {
                'Labels': {'org.objview.smoke.role': 'algorithm'}, 'Volumes': {'/extra': {}}}}]), '')
        runtime = DockerRuntime('unix:///tmp/docker.sock', execute=execute)
        with self.assertRaises(RuntimeFailure) as caught:
            runtime.resolve_image('dummy:dev', 'algorithm')
        self.assertEqual(caught.exception.code, 'IMAGE_VOLUMES_FORBIDDEN')


if __name__ == '__main__':
    unittest.main()
