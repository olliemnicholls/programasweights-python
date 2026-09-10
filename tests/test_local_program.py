"""Local archives remain local, isolated by their bytes, and safe to load."""

import builtins
import ctypes
import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import programasweights as paw
from programasweights import cache, config
from programasweights.client import PAWClient, Program


PID = "a" * 20
MODEL = b"GGUF" + b"M" * 4092
ADAPTER = b"GGUF" + b"A" * (cache.MIN_ADAPTER_GGUF_SIZE - 4)
TEMPLATE = "Classify:{INPUT_PLACEHOLDER}:Label"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PAW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PAW_API_URL", "https://api.invalid.test")
    monkeypatch.delenv("PAW_OFFLINE", raising=False)
    for runtime in cache.LEGACY_RUNTIME_MANIFESTS.values():
        model = runtime["local_sdk"]["base_model"]
        monkeypatch.setitem(model, "size_bytes", len(MODEL))
        monkeypatch.setitem(model, "sha256", hashlib.sha256(MODEL).hexdigest())

    def forbidden(*args, **kwargs):
        pytest.fail("Local-program handling must not issue network requests")

    for name in ("get", "post", "delete", "stream", "request"):
        monkeypatch.setattr(httpx, name, forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.fixture
def native(monkeypatch):
    state = SimpleNamespace(instances=[], adapters=[], freed=[])
    module = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = object()
            self.ctx = object()
            self.n_tokens = 0
            self.input_ids = [0] * 4096
            self.closed = False
            self.tokenized = []
            self._sample_index = 0
            state.instances.append(self)

        def tokenize(self, data, *, add_bos, special):
            self.tokenized.append(data)
            return list(data)

        def eval(self, tokens):
            self.n_tokens += len(tokens)

        def sample(self, *, temp):
            token = ord("A") if self._sample_index == 0 else 0
            self._sample_index += 1
            return token

        def token_eos(self):
            return 0

        def detokenize(self, tokens):
            return bytes(tokens)

        def reset(self):
            self.n_tokens = 0
            self._sample_index = 0

        def close(self):
            self.closed = True

    def load_adapter(model, path):
        state.adapters.append(Path(path.decode()))
        return object()

    module.Llama = FakeLlama
    module.llama_adapter_lora_init = load_adapter
    module.llama_set_adapter_lora = lambda *args: None
    module.llama_adapter_lora_free = state.freed.append
    module.llama_token = ctypes.c_int
    module.llama_state_seq_load_file = lambda *args: 0
    module.llama_state_seq_save_file = lambda *args: 0
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    sys.modules.pop("programasweights.runtime_llamacpp", None)
    runtime = importlib.import_module("programasweights.runtime_llamacpp")
    yield runtime, state
    sys.modules.pop("programasweights.runtime_llamacpp", None)


def entries(**overrides):
    runtime = cache.get_base_runtime_manifest("gpt2")
    metadata = {
        "version": 4, "program_id": PID, "spec": "Classify input.",
        "interpreter": "gpt2", "runtime_id": runtime["runtime_id"],
        "runtime_manifest_version": runtime["manifest_version"], "runtime": runtime,
    }
    result = {
        "meta.json": json.dumps(metadata).encode(),
        "adapter.gguf": ADAPTER,
        "prompt_template.txt": TEMPLATE.encode(),
        "pseudo_program.txt": b"Task: Classify input as A.",
    }
    result.update(overrides)
    return result


def archive(path, members=None, *, extras=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as output:
        for name, value in (members if members is not None else entries()).items():
            output.writestr(name, value)
        for name, value in extras:
            output.writestr(name, value)
    return path


def base_model():
    runtime = cache.get_base_runtime_manifest("gpt2")
    path = config.get_base_models_dir() / runtime["local_sdk"]["base_model"]["file"]
    path.write_bytes(MODEL)
    return path


def import_local(path):
    from programasweights.local_program import import_local_program
    return import_local_program(path)


def test_local_bundle_loads_and_infers_offline_through_actual_runtime(tmp_path, native):
    source = archive(tmp_path / "example.paw")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    before = source.read_bytes()
    base_model()
    runtime, state = native
    fn = paw.function(source, offline=True, n_gpu_layers=0, verbose=True)
    try:
        assert isinstance(fn, runtime.PawFunction)
        assert fn("hello", max_tokens=2) == "A"
        assert fn.interpreter == "gpt2"
        assert state.instances[0].kwargs["n_gpu_layers"] == 0
        assert state.adapters == [config.get_cache_dir() / "local_programs" / digest / "adapter.gguf"]
        assert TEMPLATE.split(cache.INPUT_PLACEHOLDER)[0].encode() in state.instances[0].tokenized
    finally:
        fn.close()
    assert state.instances[0].closed
    assert source.read_bytes() == before


@pytest.mark.parametrize("offline_mode", ["argument", "environment"])
def test_local_missing_base_offline_fails_before_native_load(tmp_path, native, monkeypatch, offline_mode):
    source = archive(tmp_path / "example.paw")
    _, state = native
    if offline_mode == "environment":
        monkeypatch.setenv("PAW_OFFLINE", "1")
    with pytest.raises((RuntimeError, FileNotFoundError), match="[Oo]ffline|[Cc]ached|[Mm]issing"):
        paw.function(source, offline=offline_mode == "argument", verbose=True)
    assert state.instances == []
    assert state.adapters == []


def test_local_default_mode_can_fetch_only_missing_base_model(tmp_path, native, monkeypatch):
    source = archive(tmp_path / "example.paw")
    downloads = []

    def download(url, dest, *args, **kwargs):
        downloads.append((url, dest))
        Path(dest).write_bytes(MODEL)

    monkeypatch.setattr(cache, "_download_file", download)
    fn = paw.function(source, verbose=True)
    try:
        assert fn("hello", max_tokens=1) == "A"
    finally:
        fn.close()
    assert len(downloads) == 1
    assert downloads[0][0] == cache.BASE_MODEL_URLS["gpt2-q8_0"]


@pytest.mark.parametrize("representation", ["path", "relative_paw", "uppercase_paw", "dot_prefix", "parent_prefix", "absolute", "tilde", "custom_pathlike"])
def test_explicit_local_references_are_imported(tmp_path, native, monkeypatch, representation):
    monkeypatch.chdir(tmp_path)
    filename = "EXAMPLE.PAW" if representation == "uppercase_paw" else "example.paw"
    source = archive(tmp_path / filename)
    if representation in ("dot_prefix", "parent_prefix", "absolute", "custom_pathlike"):
        source = source.rename(tmp_path / "extensionless")
    if representation == "path":
        reference = source
    elif representation in ("relative_paw", "uppercase_paw"):
        reference = source.name
    elif representation == "dot_prefix":
        reference = "./extensionless"
    elif representation == "parent_prefix":
        (tmp_path / "subdir").mkdir()
        monkeypatch.chdir(tmp_path / "subdir")
        reference = "../extensionless"
    elif representation == "absolute":
        reference = str(source)
    elif representation == "tilde":
        # Python 3.9/3.10 cache os.path.expanduser in pathlib's accessor;
        # patch the public Path method so this fixture is version-independent.
        original_expanduser = Path.expanduser

        def expand_test_home(value):
            if value.parts and value.parts[0] == "~":
                return tmp_path.joinpath(*value.parts[1:])
            return original_expanduser(value)

        monkeypatch.setattr(Path, "expanduser", expand_test_home)
        reference = "~/example.paw"
    else:
        class LocalPath:
            def __fspath__(self):
                return str(source)
        reference = LocalPath()
    base_model()
    fn = paw.function(reference, offline=True, verbose=True)
    fn.close()


@pytest.mark.parametrize("reference", ["missing.paw", "missing.PAW", "./missing", "../missing", "/unlikely-paw-test-missing", "~/missing-local-paw-test", r"C:\missing\file.paw", r"\\host\share\missing.paw"])
def test_missing_local_reference_never_looks_up_hub_or_loads_native(reference, native, monkeypatch):
    _, state = native
    local_paths = []
    if os.name == "nt" and (reference.startswith("\\\\") or reference.startswith("C:")):
        from programasweights import local_program

        def missing_windows_path(path):
            # Test Windows/UNC classification without attempting a real SMB
            # connection or touching an arbitrary drive outside the fixture.
            local_paths.append(path)
            raise FileNotFoundError(path)

        monkeypatch.setattr(local_program, "import_local_program", missing_windows_path)
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError, OSError)):
        paw.function(reference, verbose=True)
    if os.name == "nt" and (reference.startswith("\\\\") or reference.startswith("C:")):
        assert local_paths == [Path(reference)]
    assert not state.instances
    assert not state.adapters


@pytest.mark.parametrize("reference", ["https://example.com/a.paw", "http://example.com/a.paw", "file:///tmp/a.paw", "s3://bucket/a.paw", "ftp://example.com/a", "https://example.com/slug"])
def test_unsupported_uri_rejected_before_lookup(reference, native):
    _, state = native
    with pytest.raises(ValueError):
        paw.function(reference)
    assert not state.instances


@pytest.mark.parametrize("reference", [b"file.paw", [], ["file.paw"], {}, 123])
def test_invalid_reference_types_rejected(reference, native):
    with pytest.raises(TypeError):
        paw.function(reference)
    assert not native[1].instances


def test_remote_reference_checks_native_dependency_before_hub_lookup(monkeypatch):
    original_import = builtins.__import__

    def missing_runtime(name, *args, **kwargs):
        if name == "runtime_llamacpp":
            raise ModuleNotFoundError("Native runtime unavailable")
        return original_import(name, *args, **kwargs)

    def forbidden_lookup(*args, **kwargs):
        pytest.fail("Hub lookup must not run without the native runtime")

    monkeypatch.setattr(builtins, "__import__", missing_runtime)
    monkeypatch.setattr(paw, "_resolve_program_id", forbidden_lookup)
    with pytest.raises(ModuleNotFoundError, match="Native runtime unavailable"):
        paw.function("owner/slug")


def test_missing_local_file_is_reported_before_native_dependency(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_runtime_import(name, *args, **kwargs):
        if name == "runtime_llamacpp":
            pytest.fail("Missing local files must fail before native import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_runtime_import)
    with pytest.raises(FileNotFoundError):
        paw.function(tmp_path / "missing.paw")


def test_pathlike_returning_bytes_rejected(native):
    class BadPath:
        def __fspath__(self):
            return b"file.paw"
    with pytest.raises(TypeError):
        paw.function(BadPath())
    assert not native[1].instances


@pytest.mark.parametrize("reference", [PID, "owner/slug", "ordinary-slug"])
def test_remote_reference_stays_remote_even_when_same_named_file_exists(tmp_path, native, monkeypatch, reference):
    monkeypatch.chdir(tmp_path)
    archive(tmp_path / reference)
    program_dir = config.get_programs_dir() / PID
    program_dir.mkdir()
    for name, value in entries().items():
        (program_dir / name).write_bytes(value)
    base_model()
    resolutions = []

    def resolve(value, *, offline, **kwargs):
        resolutions.append(value)
        return PID

    monkeypatch.setattr(paw, "_resolve_program_id", resolve)
    fn = paw.function(reference, offline=True, verbose=True)
    fn.close()
    assert resolutions == [reference]
    assert native[1].adapters == [program_dir / "adapter.gguf"]
    assert not (config.get_cache_dir() / "local_programs").exists()


def test_program_object_and_explicit_base_mode_remain_supported(tmp_path, native, monkeypatch):
    program_dir = config.get_programs_dir() / PID
    program_dir.mkdir()
    for name, value in entries().items():
        (program_dir / name).write_bytes(value)
    base_model()
    seen = []
    monkeypatch.setattr(paw, "_resolve_program_id", lambda value, **kwargs: seen.append(value) or PID)
    fn = paw.function(Program(id=PID, status="ready"), offline=True, verbose=True)
    fn.close()
    assert seen == [PID]
    base = paw.function(None, interpreter="gpt2", offline=True, verbose=True)
    assert base("hello", max_tokens=1) == "A"
    base.close()


def test_content_addressed_import_does_not_clobber_hub_or_alias_cache(tmp_path):
    first = archive(tmp_path / "first.paw")
    second = archive(tmp_path / "second.paw", entries(**{"adapter.gguf": b"GGUF" + b"B" * (len(ADAPTER) - 4)}))
    hub_dir = config.get_programs_dir() / PID
    hub_dir.mkdir()
    for name, value in entries().items():
        (hub_dir / name).write_bytes(value)
    sentinel = hub_dir / "keep.txt"
    sentinel.write_bytes(b"existing Hub program")
    cache.save_slug_mapping("owner/slug", PID)
    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    second_hash = hashlib.sha256(second.read_bytes()).hexdigest()
    first_dir, second_dir = import_local(first), import_local(second)
    assert first_dir == config.get_cache_dir() / "local_programs" / first_hash
    assert second_dir == config.get_cache_dir() / "local_programs" / second_hash
    assert first_dir != second_dir
    assert json.loads((first_dir / "meta.json").read_text())["program_id"] == PID
    assert json.loads((second_dir / "meta.json").read_text())["program_id"] == PID
    assert sentinel.read_bytes() == b"existing Hub program"
    assert cache.get_cached_slug("owner/slug") == PID
    assert hashlib.sha256(first.read_bytes()).hexdigest() == first_hash
    assert hashlib.sha256(second.read_bytes()).hexdigest() == second_hash


def test_concurrent_identical_imports_publish_one_complete_directory(tmp_path):
    source = archive(tmp_path / "example.paw")
    before = source.read_bytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        dirs = list(pool.map(import_local, [source] * 8))
    assert len(set(dirs)) == 1
    imported = dirs[0]
    assert (imported / "adapter.gguf").read_bytes() == ADAPTER
    assert (imported / "prompt_template.txt").read_text() == TEMPLATE
    assert source.read_bytes() == before
    assert not list(imported.parent.glob(".import-*"))


def test_separate_processes_import_same_archive_atomically(tmp_path):
    # Fresh child interpreters do not inherit the in-process fake model hash.
    # An older current-format bundle may omit its optional embedded runtime.
    members = entries()
    metadata = json.loads(members["meta.json"])
    metadata.pop("runtime")
    members["meta.json"] = json.dumps(metadata).encode()
    source = archive(tmp_path / "example.paw", members)
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from programasweights.local_program import import_local_program\n"
        "print(import_local_program(Path(sys.argv[1])))\n"
    )
    expected = config.get_cache_dir() / "local_programs" / hashlib.sha256(source.read_bytes()).hexdigest()
    processes = [subprocess.Popen([sys.executable, "-c", script, str(source)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    try:
        results = [process.communicate(timeout=30) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    for process, (stdout, stderr) in zip(processes, results):
        assert process.returncode == 0, stderr
        assert stdout.strip() == str(expected)
    assert (expected / "adapter.gguf").read_bytes() == ADAPTER
    assert not list(expected.parent.glob(".import-*"))


def test_original_changed_after_snapshot_does_not_change_imported_bytes(tmp_path, monkeypatch):
    source = archive(tmp_path / "example.paw")
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original_extract = PAWClient._safe_extract_paw

    def extract_after_change(snapshot, destination):
        archive(source, entries(**{"adapter.gguf": b"GGUF" + b"Z" * (len(ADAPTER) - 4)}))
        return original_extract(snapshot, destination)

    monkeypatch.setattr(PAWClient, "_safe_extract_paw", extract_after_change)
    imported = import_local(source)
    assert imported.name == original_hash
    assert (imported / "adapter.gguf").read_bytes() == ADAPTER
    assert hashlib.sha256(source.read_bytes()).hexdigest() != original_hash


def test_reimport_preserves_generated_prefix_state_and_rejects_corrupted_cache(tmp_path):
    source = archive(tmp_path / "example.paw")
    imported = import_local(source)
    prefix = imported / "prefix_kv_cache.bin"
    prefix.write_bytes(b"runtime-generated state")
    assert import_local(source) == imported
    assert prefix.read_bytes() == b"runtime-generated state"
    (imported / "adapter.gguf").write_bytes(b"GGUF" + b"Z" * (len(ADAPTER) - 4))
    with pytest.raises(ValueError):
        import_local(source)
    assert (imported / "adapter.gguf").read_bytes() == b"GGUF" + b"Z" * (len(ADAPTER) - 4)
    assert prefix.read_bytes() == b"runtime-generated state"


def test_changed_source_gets_new_cache_directory_preserving_old_import(tmp_path):
    source = archive(tmp_path / "example.paw")
    old_dir = import_local(source)
    archive(source, entries(**{"prompt_template.txt": b"Changed:{INPUT_PLACEHOLDER}"}))
    new_dir = import_local(source)
    assert old_dir != new_dir
    assert (old_dir / "prompt_template.txt").read_text() == TEMPLATE
    assert (new_dir / "prompt_template.txt").read_text() == "Changed:{INPUT_PLACEHOLDER}"


@pytest.mark.parametrize("problem", ["missing_meta", "missing_adapter", "missing_template", "meta_list", "invalid_json", "bad_id", "no_interpreter", "bad_template", "multiple_placeholders", "bad_gguf", "tiny_gguf", "legacy", "non_zip", "traversal", "absolute_member", "symlink_member", "duplicate_member", "native_prefix_state"])
def test_invalid_local_archives_fail_before_native_loading(tmp_path, native, problem):
    members = entries()
    source = tmp_path / "invalid.paw"
    extras = []
    if problem.startswith("missing_"):
        del members[{"missing_meta": "meta.json", "missing_adapter": "adapter.gguf", "missing_template": "prompt_template.txt"}[problem]]
    elif problem == "meta_list":
        members["meta.json"] = b"[]"
    elif problem == "invalid_json":
        members["meta.json"] = b"{invalid"
    elif problem in ("bad_id", "no_interpreter"):
        meta = json.loads(members["meta.json"])
        meta["program_id" if problem == "bad_id" else "interpreter"] = "../bad" if problem == "bad_id" else ""
        members["meta.json"] = json.dumps(meta).encode()
    elif problem == "bad_template":
        members["prompt_template.txt"] = b"No placeholder"
    elif problem == "multiple_placeholders":
        members["prompt_template.txt"] = (cache.INPUT_PLACEHOLDER * 2).encode()
    elif problem == "bad_gguf":
        members["adapter.gguf"] = b"WRNG" + ADAPTER[4:]
    elif problem == "tiny_gguf":
        members["adapter.gguf"] = b"GGUF"
    elif problem in ("traversal", "absolute_member"):
        extras = [("../outside" if problem == "traversal" else "/outside", b"bad")]
    elif problem == "symlink_member":
        info = zipfile.ZipInfo("dangerous-link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        extras = [(info, b"../outside")]
    elif problem == "duplicate_member":
        extras = [("adapter.gguf", ADAPTER)]
    elif problem == "native_prefix_state":
        extras = [("prefix_kv_cache.bin", b"untrusted native serialized state")]
    if problem in ("legacy", "non_zip"):
        source.write_bytes(b"PAW\x00old-format" if problem == "legacy" else b"not a zip archive")
    elif problem == "duplicate_member":
        with pytest.warns(UserWarning, match="Duplicate"):
            archive(source, members, extras=extras)
    else:
        archive(source, members, extras=extras)
    before = source.read_bytes()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        paw.function(source, verbose=True)
    assert source.read_bytes() == before
    assert not native[1].instances
    assert not (tmp_path / "outside").exists()


def test_directory_source_is_rejected_without_native_loading(tmp_path, native):
    source = tmp_path / "directory.paw"
    source.mkdir()
    with pytest.raises((ValueError, OSError, RuntimeError)):
        paw.function(source, verbose=True)
    assert source.is_dir()
    assert not native[1].instances


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO test requires POSIX")
def test_fifo_source_is_rejected_without_blocking(tmp_path, native):
    source = tmp_path / "fifo.paw"
    os.mkfifo(source)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        paw.function(source, verbose=True)
    assert stat.S_ISFIFO(source.stat().st_mode)
    assert not native[1].instances


def test_source_symlink_to_regular_archive_is_supported_readonly(tmp_path, native):
    target = archive(tmp_path / "target.paw")
    source = tmp_path / "link.paw"
    try:
        source.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows runner lacks the privilege to create symlinks")
        raise
    before = target.read_bytes()
    base_model()
    fn = paw.function(source, offline=True, verbose=True)
    assert fn("hello", max_tokens=1) == "A"
    fn.close()
    assert source.is_symlink()
    assert target.read_bytes() == before
    assert native[1].instances[0].closed


@pytest.mark.parametrize("bound", ["archive_size", "member_count", "expanded_size"])
def test_local_archive_resource_limits_are_enforced(tmp_path, native, monkeypatch, bound):
    import programasweights.client as client_module
    source = archive(tmp_path / "example.paw")
    name = {"archive_size": "MAX_PAW_ARCHIVE_BYTES", "member_count": "MAX_PAW_ARCHIVE_MEMBERS", "expanded_size": "MAX_PAW_EXPANDED_BYTES"}[bound]
    monkeypatch.setattr(client_module, name, 1)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        paw.function(source, verbose=True)
    assert not native[1].instances
