"""Long sync response waits must not weaken other timeouts or retry work."""

import time
from unittest.mock import Mock

import httpx
import pytest

import programasweights as paw
from programasweights.client import PAWClient, Program


BASE_URL = "https://api.example.test"
SPEC = "Classify the input as FORMAT or OTHER."


@pytest.fixture(autouse=True)
def no_network_or_retry(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected HTTP request or retry sleep")

    for method in ("post", "get", "delete", "stream"):
        monkeypatch.setattr(httpx, method, forbidden)
    monkeypatch.setattr(time, "sleep", forbidden)
    monkeypatch.setattr(paw, "get_api_url", lambda: BASE_URL)
    monkeypatch.setattr(paw, "get_api_key", lambda: "test-api-key")


@pytest.mark.parametrize("method,verb,path,args,kwargs,read,other", [
    ("compile", "post", "/api/v1/compile", (SPEC,), {}, 2400.0, 120.0),
    ("compile_async", "post", "/api/v1/compile/async", (SPEC,), {"compiler": "paw-ft-bs48"}, 30.0, 30.0),
    ("precheck_compile", "post", "/api/v1/compile/precheck", (SPEC,), {}, 10.0, 10.0),
    ("get_compile_status", "get", "/api/v1/compile/job123", ("job123",), {}, 10.0, 10.0),
    ("cancel_compile", "delete", "/api/v1/compile/job123", ("job123",), {}, 10.0, 10.0),
])
def test_compile_endpoint_timeout_components_are_explicit_and_scoped(
    monkeypatch, method, verb, path, args, kwargs, read, other,
):
    response = httpx.Response(200, request=httpx.Request(verb.upper(), BASE_URL + path),
                              json={"status": "ready", "program_id": "a" * 20})
    request = Mock(return_value=response)
    monkeypatch.setattr(httpx, verb, request)

    getattr(PAWClient(api_url=BASE_URL, api_key="test-api-key"), method)(*args, **kwargs)

    request.assert_called_once()
    assert request.call_args.args == (BASE_URL + path,)
    timeout = httpx.Timeout(request.call_args.kwargs["timeout"])
    assert timeout.read == read
    assert timeout.connect == other
    assert timeout.write == other
    assert timeout.pool == other


def test_sync_compile_request_and_program_result_are_unchanged(monkeypatch):
    payload = {
        "program_id": "b" * 20, "status": "ready", "slug": "user/classifier",
        "compiler_snapshot": "paw-ft-snapshot", "compiler_kind": "finetune_lora",
        "pseudo_program_strategy": "none", "runtime_id": "runtime-test",
        "runtime_manifest_version": 1, "timings": {"total_s": 350.0},
        "error": None, "version": 3, "version_action": "new",
    }
    response = httpx.Response(200, request=httpx.Request("POST", BASE_URL + "/api/v1/compile"), json=payload)
    post = Mock(return_value=response)
    monkeypatch.setattr(httpx, "post", post)

    result = PAWClient(api_url=BASE_URL, api_key="test-api-key").compile(
        SPEC, compiler="paw-ft-bs48", name="Classifier", tags=["test"],
        public=False, slug="classifier", ephemeral=True,
    )

    post.assert_called_once()
    assert post.call_args.args == (BASE_URL + "/api/v1/compile",)
    assert post.call_args.kwargs["json"] == {
        "spec": SPEC, "compiler": "paw-ft-bs48", "name": "Classifier",
        "tags": ["test"], "public": False, "slug": "classifier", "ephemeral": True,
    }
    assert post.call_args.kwargs["headers"] == {
        "Content-Type": "application/json", "X-API-Key": "test-api-key",
    }
    assert result == Program(id=payload["program_id"], **{
        key: value for key, value in payload.items() if key != "program_id"
    })


def test_sync_read_timeout_propagates_same_exception_after_one_request(monkeypatch):
    expected = httpx.ReadTimeout(
        "mock response wait expired", request=httpx.Request("POST", BASE_URL + "/api/v1/compile"),
    )
    post = Mock(side_effect=expected)
    monkeypatch.setattr(httpx, "post", post)

    with pytest.raises(httpx.ReadTimeout) as caught:
        PAWClient(api_url=BASE_URL, api_key="test-api-key").compile(SPEC)

    assert caught.value is expected
    post.assert_called_once()


@pytest.mark.parametrize("entrypoint", ["compile", "compile_and_load"])
@pytest.mark.parametrize("failure", ["api_error", "read_timeout"])
def test_public_compile_helpers_propagate_failures_without_loading_or_retry(
    monkeypatch, entrypoint, failure,
):
    request = httpx.Request("POST", BASE_URL + "/api/v1/compile")
    response = httpx.Response(503, request=request, json={"detail": {
        "error": "durable_queue_unavailable", "message": "Try after recovery.",
        "request_id": "compile-request-123",
    }})
    timeout = httpx.ReadTimeout("mock read timeout", request=request)
    post = Mock(side_effect=timeout) if failure == "read_timeout" else Mock(return_value=response)
    load = Mock(side_effect=AssertionError("failed compilation must not load a function"))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(paw, "function", load)

    with pytest.raises(httpx.ReadTimeout if failure == "read_timeout" else paw.APIError) as caught:
        getattr(paw, entrypoint)(SPEC, compiler="paw-ft-bs48")

    if failure == "read_timeout":
        assert caught.value is timeout
    else:
        assert isinstance(caught.value, httpx.HTTPStatusError)
        assert caught.value.response is response
        assert caught.value.request is request
        assert caught.value.code == "durable_queue_unavailable"
        assert caught.value.message == "Try after recovery."
        assert caught.value.request_id == "compile-request-123"
    post.assert_called_once()
    load.assert_not_called()
