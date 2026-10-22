"""Constrained local Docker CLI boundary; never evaluates a shell command."""
import json
import os
from pathlib import Path
import re
import subprocess

LABEL = 'org.objview.job'
IMAGE_ID = re.compile(r'sha256:[0-9a-f]{64}')
RESOURCE = re.compile(r'ovb-[0-9a-f]{32}-(?:algorithm|benchmark|init|net|data)')


class RuntimeFailure(RuntimeError):
    def __init__(self, code='DOCKER_COMMAND_FAILED'):
        self.code = code
        super().__init__(code)


class DockerRuntime:
    def __init__(self, endpoint, *, execute=subprocess.run):
        if not isinstance(endpoint, str) or not endpoint.startswith('unix:///') or '\n' in endpoint:
            raise ValueError('Only an explicit local Unix Docker socket is allowed')
        self.endpoint = endpoint
        self.execute = execute

    def call(self, args, *, timeout=30):
        env = {key: value for key, value in os.environ.items() if not key.startswith('DOCKER_')}
        try:
            result = self.execute(['docker', '--host', self.endpoint, *args], shell=False,
                                  env=env, capture_output=True, text=True, timeout=max(.001, timeout), check=False)
        except subprocess.TimeoutExpired:
            raise RuntimeFailure('DOCKER_COMMAND_TIMEOUT') from None
        except OSError:
            raise RuntimeFailure('DOCKER_UNAVAILABLE') from None
        # Never propagate raw output (can contain token/env/paths). Warnings about
        # ignored rootless resource controls must not silently turn into success.
        if result.returncode != 0 or 'warning' in result.stderr.lower():
            raise RuntimeFailure()
        if len(result.stdout) > 1024 * 1024:
            raise RuntimeFailure('DOCKER_OUTPUT_TOO_LARGE')
        return result.stdout.strip()

    def preflight(self, *, timeout=30):
        try:
            info = json.loads(self.call(['info', '--format', '{{json .}}'], timeout=timeout))
            if info.get('OSType') != 'linux' or info.get('CgroupDriver') not in {'systemd', 'cgroupfs'}:
                raise RuntimeFailure('UNSUPPORTED_DOCKER_HOST')
            if 'rootless' in ' '.join(info.get('SecurityOptions', [])):
                if info.get('CgroupDriver') != 'systemd' or str(info.get('CgroupVersion')) != '2':
                    raise RuntimeFailure('ROOTLESS_RESOURCE_LIMITS_UNAVAILABLE')
            if not info.get('MemoryLimit') or not info.get('PidsLimit'):
                raise RuntimeFailure('RESOURCE_LIMITS_UNAVAILABLE')
            return {'server_version': info['ServerVersion'], 'cgroup_driver': info['CgroupDriver'],
                    'cgroup_version': str(info.get('CgroupVersion')), 'security_options': info.get('SecurityOptions', [])}
        except (ValueError, KeyError, TypeError):
            raise RuntimeFailure('INVALID_DOCKER_RESPONSE') from None

    def resolve_image(self, reference, role, *, timeout=30):
        if not isinstance(reference, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}', reference):
            raise ValueError('Invalid local image reference')
        try:
            record = json.loads(self.call(['image', 'inspect', reference], timeout=timeout))[0]
            image_id = record['Id']
            labels = record['Config'].get('Labels') or {}
            if not IMAGE_ID.fullmatch(image_id) or labels.get('org.objview.smoke.role') != role:
                raise RuntimeFailure('SMOKE_IMAGE_REQUIRED')
            # VOLUME declarations create implicit mounts outside our contract.
            if record['Config'].get('Volumes'):
                raise RuntimeFailure('IMAGE_VOLUMES_FORBIDDEN')
            return image_id
        except (ValueError, KeyError, TypeError, IndexError):
            raise RuntimeFailure('INVALID_IMAGE_RESPONSE') from None

    def lookup(self, kind, name, *, timeout=10):
        if kind not in {'container', 'network', 'volume'} or not RESOURCE.fullmatch(name):
            raise ValueError('Invalid cleanup resource')
        listing = [kind, 'ls'] + (['--all'] if kind == 'container' else [])
        field = '{{.Names}}' if kind == 'container' else '{{.Name}}'
        names = self.call(listing + ['--format', field], timeout=timeout).splitlines()
        if name not in names:
            return None
        try:
            return json.loads(self.call([kind, 'inspect', name], timeout=timeout))[0]
        except (ValueError, TypeError, IndexError):
            raise RuntimeFailure('INVALID_DOCKER_RESPONSE') from None


def container_args(*, name, image_id, network, volume, role, job_id, token_file):
    if (not IMAGE_ID.fullmatch(image_id) or not re.fullmatch('[0-9a-f]{32}', job_id)
            or role not in {'algorithm', 'benchmark', 'init'}):
        raise ValueError('Invalid container contract')
    for resource in (name, network, volume):
        if not RESOURCE.fullmatch(resource) or not resource.startswith(f'ovb-{job_id}-'):
            raise ValueError('Invalid resource identity')
    if name != f'ovb-{job_id}-{role}' or network != f'ovb-{job_id}-net' or volume != f'ovb-{job_id}-data':
        raise ValueError('Resource role mismatch')
    token_file = Path(token_file)
    if not token_file.is_absolute() or any(c in str(token_file) for c in '\r\n'):
        raise ValueError('Invalid env file')
    args = ['create', '--name', name, '--label', f'{LABEL}={job_id}', '--pull', 'never',
            '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--cpus', '1', '--memory', '512m', '--memory-swap', '512m', '--pids-limit', '128',
            '--ulimit', 'nofile=1024:1024', '--log-driver', 'none',
            '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=67108864,mode=1777',
            '--env', 'PYTHONDONTWRITEBYTECODE=1']
    if role == 'init':
        # Trusted fixed initializer, not an algorithm. Only capability needed to
        # transfer newly created volume directories to the non-root runtime UID.
        args += ['--network', 'none', '--user', '0:0', '--cap-add', 'CHOWN',
                 '--mount', f'type=volume,source={volume},target=/episode,volume-nocopy',
                 '--entrypoint', 'python', image_id, '-c',
                 "import os; from pathlib import Path; "
                 "[(p.mkdir(mode=0o755), os.chown(p,65532,65532)) for p in "
                 "[Path('/episode/private'),Path('/episode/released')]]"]
        return args
    args += ['--network', network, '--user', '65532:65532', '--env-file', str(token_file)]
    if role == 'algorithm':
        args += ['--mount', f'type=volume,source={volume},target=/observations,volume-subpath=released,readonly,volume-nocopy',
                 '--env', 'OBSERVATION_ROOT=/observations', '--env', 'BENCHMARK_URL=http://benchmark:8081']
    else:
        args += ['--network-alias', 'benchmark', '--mount',
                 f'type=volume,source={volume},target=/episode,volume-nocopy']
    return args + [image_id]
