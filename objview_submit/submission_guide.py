"""Public integration guide; never imports the worker or benchmark implementation."""
from html import escape
from pathlib import Path


EXAMPLE_FILES = ('Dockerfile', 'run.sh', 'algorithm.py', 'submission_client.py')


def example_file(name):
    if name not in EXAMPLE_FILES:
        return None
    return (Path(__file__).parent / 'examples' / name).read_text()


def dockerfile_example():
    return ('<section class="panel"><h2>Example Dockerfile</h2>'
            '<p>Download the four files below into one directory. Replace the dummy planning logic in '
            '<code>algorithm.py</code> with your method; keep the API interaction and launch contract.</p>'
            '<pre><code>' + escape(example_file('Dockerfile')) + '</code></pre><ul>' + ''.join(
                '<li><a href="/ui/examples/' + name + '" download="' + name + '">' + name + '</a></li>'
                for name in EXAMPLE_FILES) + '</ul>'
            '<p>The example base image is Linux/AMD64. Use an approved ARM64 image on an ARM64 worker. '
            'An image download does not make it architecture-compatible. Python and pip must be available; '
            'configure an internal package mirror or provide offline dependencies during the build.</p>'
            '<p>The worker injects <code>BENCHMARK_URL</code> and a fresh <code>OBJVIEW_EPISODE_TOKEN</code> '
            'at runtime. Never bake a token, a dataset, or a benchmark service into your algorithm image.</p>'
            '<p><a href="/ui/guide">Read the API workflow and resource limits</a></p></section>')


def render_guide():
    return '''<section class="hero"><p class="eyebrow">SUBMISSION GUIDE</p>
<h1>Consume observations. Produce view actions.</h1>
<p class="lead">Your algorithm runs separately from the trusted benchmark. Exchange observations and actions through the episode API; the benchmark owns rendering, scoring and timing.</p></section>
<section class="panel"><h2>Producer–consumer workflow</h2>
<ol class="workflow"><li><strong>Prepare.</strong> Load your model, then call <code>POST /v1/ready</code>. Read the public episode configuration.</li>
<li><strong>Consume.</strong> Long-poll for the next step using <code>after</code> and <code>wait=10</code>. Download its released assets through the observation API.</li>
<li><strong>Plan.</strong> Compute a viewpoint. Optionally check candidate feasibility in a bounded batch.</li>
<li><strong>Produce.</strong> Submit one <code>move</code> pose or <code>stop</code> action for that episode and step. The benchmark validates it, renders the next observation and computes metrics privately.</li>
<li><strong>Advance.</strong> Update <code>after</code> to the consumed step. Exit when the response reports <code>status=done</code>.</li></ol>
<p>Only an accepted action advances the episode. Scores, runtimes, arbitrary commands and filesystem paths are not accepted as action payloads.
Decision time starts when the benchmark releases the step and includes observation transfer, API calls, inference and backoff.</p></section>
<section class="panel"><h2>Allowed episode API</h2>
<p>Every endpoint requires <code>Authorization: Bearer &lt;episode token&gt;</code>. This is a runtime credential, separate from the website submission invitation.</p>
<p><strong>Required for an algorithm:</strong> get the episode configuration, signal readiness, consume steps, download the current observations and submit one action per step. The required calls are <code>GET /v1/episode</code>, <code>POST /v1/ready</code>, <code>GET /v1/steps/next</code>, <code>GET /v1/observations/{step_index}/{asset}</code> and <code>POST /v1/actions</code>.</p>
<p><strong>Optional:</strong> <code>GET /v1/interface</code> is capability discovery, <code>GET /v1/status</code> is a lightweight diagnostic endpoint, and <code>POST /v1/feasibility/check</code> is a convenience filter. The benchmark still validates every submitted action, so an algorithm may choose not to call the optional endpoints.</p>
<div class="table-scroll" tabindex="0"><table><thead><tr><th>Endpoint</th><th>Purpose</th></tr></thead><tbody>
<tr><th>GET /v1/interface</th><td>Protocol capabilities and enforced limits.</td></tr>
<tr><th>GET /v1/episode</th><td>Anonymous episode ID, camera convention and view cap.</td></tr>
<tr><th>POST /v1/ready</th><td>Signal model readiness with an empty JSON object.</td></tr>
<tr><th>GET /v1/steps/next?after=-1&amp;wait=10</th><td>One pending step, HTTP 204 when no step is ready, or terminal status.</td></tr>
<tr><th>GET /v1/observations/{step_index}/{asset}</th><td>Current step only: rgb, depth, mask, frame_meta; points only if published.</td></tr>
<tr><th>POST /v1/feasibility/check</th><td>Check candidate poses. Returns feasibility, never an oracle score.</td></tr>
<tr><th>POST /v1/actions</th><td>Submit a move or stop action with episode_id, step_index and request_id.</td></tr>
<tr><th>GET /v1/status</th><td>Minimal lifecycle status, without scores or hidden object identity.</td></tr>
</tbody></table></div>
<p>RGB and mask use PNG; depth uses NPZ key <code>depth</code> with float32 camera-Z values in normalized object units. Frame metadata uses JSON.
The algorithm receives no benchmark filesystem mount. There are no dataset, ground-truth, score, log, directory-listing or arbitrary-file APIs.</p>
<p>A pose is <code>[cam_x, cam_y, cam_z, lookat_x, lookat_y, lookat_z, roll_rad]</code> in the normalized object frame.
For a move, add <code>action="move"</code> and <code>pose</code>; for a stop, add <code>action="stop"</code> and
one of <code>plan_end</code>, <code>candidate_exhausted</code>, <code>next_view_visited</code> or <code>next_view_infeasible</code> as <code>stop_reason</code>.</p></section>''' + dockerfile_example() + '''
<section class="panel"><h2>API-only algorithm example</h2>
<p>The downloadable example exercises a move followed by a stop. It checks delivery of RGB, depth, mask and metadata; replace candidate selection with your own model.
It is a protocol demonstration, not a competitive algorithm.</p>
<pre><code>client.ready()
config = client.episode()
after = -1
while True:
    step = client.next_step(after, wait=10)
    if step is None:
        continue
    if step.get("status") == "done":
        break
    rgb = client.observation(step, "rgb")
    depth = client.observation(step, "depth")
    # Select a pose using these observations, then submit exactly one action.
    # See algorithm.py for a complete runnable move/stop example.
    action = client.make_action(step, "stop", stop_reason="plan_end")
    client.submit(action)
    after = step["step_index"]</code></pre>
<p>Use one sequential client, cache downloaded assets locally, and avoid busy polling. The SDK makes at most three attempts for temporary HTTP 429 responses,
waiting one second between attempts. Transport retries reuse the same action and request_id. Hard quota errors are not retried.</p></section>
<section class="panel"><h2>Enforced request and resource limits</h2>
<ul><li>Per episode gateway: 20 requests/second with a burst of 40, including rejected and unauthenticated requests.</li>
<li>Total request budget: <code>128 + 64 × max_visited_view_num</code>. Exceeding it fails the episode.</li>
<li>At most 8 in-flight requests, including response delivery; one long poll and two observation downloads at a time.</li>
<li>JSON request bodies: 64 KiB, with a 5-second body-read timeout.</li>
<li>Observation assets: 8 MiB per file, 10 downloads and 32 MiB per step, 512 MiB per episode. Only published assets from the current pending step are available.</li>
<li>Feasibility: at most 512 poses per request and 4,096 queried poses per step. All actions must use finite, validated poses and the current episode/step identity.</li>
<li>Temporary pressure returns HTTP 429 with Retry-After. Changing a token, connection or forwarded IP does not reset the gateway budget.</li>
<li>The trusted smoke worker sets non-root execution, a read-only root filesystem, no benchmark/host/socket mounts, dropped capabilities, no-new-privileges,
1 CPU, 512 MiB memory including swap, 128 PIDs and a 64 MiB temporary filesystem. Job deadlines and targeted cleanup remain worker responsibilities.</li></ul>
<p><strong>Execution availability:</strong> This development service collects source references and schedules trusted smoke jobs only.
Arbitrary submitted Dockerfiles are not built or executed. Request limits do not prove protection against every malicious container:
network/DNS isolation, kernel escape resistance, disk quotas and recovery still need dedicated deployment validation before untrusted execution is enabled.</p></section>'''
