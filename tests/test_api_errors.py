"""Compile API failures expose server context without retrying submissions."""

import time

import httpx
import pytest

import programasweights as paw
from programasweights.client import PAWClient, Program
from programasweights.errors import APIError, raise_for_api_status


BASE_URL = "https://api.example.test"
SPEC = "Classify the input as FORMAT or OTHER."
REQUEST_SECRET = "SECRET-COMPILE-SPEC-DO-NOT-DUMP"
BODY_SECRET = "SECRET-RAW-BODY-DO-NOT-DUMP"


METHODS = [
    ("compile", "post", "/api/v1/compile", (SPEC,), {}),
    ("precheck_compile", "post", "/api/v1/compile/precheck", (SPEC,), {}),
    ("compile_async", "post", "/api/v1/compile/async", (SPEC,), {"compiler": "paw-ft-bs48"}),
    ("get_compile_status", "get", "/api/v1/compile/job123", ("job123",), {}),
    ("cancel_compile", "delete", "/api/v1/compile/job123", ("job123",), {}),
]


def make_response(status=503, *, payload=None, body=None, headers=None):
    request = httpx.Request(
        "POST", BASE_URL + "/api/v1/compile/async",
        json={"spec": REQUEST_SECRET},
        headers={"X-API-Key": "SECRET-API-KEY"},
    )
    options = {"content": body} if body is not None else {"json": payload}
    return httpx.Response(status, request=request, headers=headers, **options)


def capture(response):
    with pytest.raises(APIError) as caught:
        raise_for_api_status(response)
    return caught.value


def test_api_error_is_public_and_httpx_compatible():
    assert paw.APIError is APIError
    assert issubclass(APIError, httpx.HTTPStatusError)


@pytest.mark.parametrize("method,http_method,path,args,kwargs", METHODS)
@pytest.mark.parametrize("code,message", [
    ("durable_queue_unavailable", "Async compile is unavailable until durable Redis is healthy."),
    ("dispatch_failed", "Compile was recorded durably; recovery will reconcile it."),
])
def test_each_compile_api_surfaces_context_once_without_retry(
    monkeypatch, method, http_method, path, args, kwargs, code, message,
):
    calls = []
    responses = []

    def fake_request(url, **request_kwargs):
        calls.append((url, request_kwargs))
        request = httpx.Request(http_method.upper(), url, headers=request_kwargs["headers"])
        response = httpx.Response(503, request=request,
            headers={"Retry-After": "60", "X-Request-ID": "header-request-id"},
            json={"detail": {"error": code, "message": message, "request_id": "body-request-id"}},
        )
        responses.append(response)
        return response

    def forbidden(*args, **kwargs):
        pytest.fail("API error handling must not retry, sleep, or call another endpoint")

    for verb in ("post", "get", "delete"):
        monkeypatch.setattr(httpx, verb, fake_request if verb == http_method else forbidden)
    monkeypatch.setattr(time, "sleep", forbidden)
    client = PAWClient(api_url=BASE_URL, api_key="test-api-key")
    with pytest.raises(httpx.HTTPStatusError) as caught:
        getattr(client, method)(*args, **kwargs)
    error = caught.value
    assert isinstance(error, APIError)
    assert len(calls) == 1
    assert calls[0][0] == BASE_URL + path
    assert error.code == code
    assert error.message == message
    assert error.request_id == "body-request-id"
    assert code in str(error)
    assert message in str(error)
    assert "body-request-id" in str(error)
    assert "503" in str(error)
    assert error.request is responses[0].request
    assert error.response is responses[0]
    assert error.response.status_code == 503
    assert error.response.headers["retry-after"] == "60"


@pytest.mark.parametrize("method,http_method,path,args,kwargs", METHODS)
def test_successful_compile_api_results_are_unchanged(monkeypatch, method, http_method, path, args, kwargs):
    payload = {
        "program_id": "a" * 20, "status": "ready", "job_id": "job123",
        "cached": True, "compiler_snapshot": "snapshot123", "error": None,
        "runtime_id": "qwen-runtime", "runtime_manifest_version": 1,
        "compiler_kind": "finetune_lora", "pseudo_program_strategy": "none",
        "timings": {"total_s": 3.5}, "slug": "user/classifier", "version": 2,
        "version_action": "new_version", "server_extension": {"keep": True},
    }
    calls = []

    def request(url, **request_kwargs):
        calls.append(url)
        return httpx.Response(200, request=httpx.Request(http_method.upper(), url), json=payload)

    monkeypatch.setattr(httpx, http_method, request)
    client = PAWClient(api_url=BASE_URL)
    result = getattr(client, method)(*args, **kwargs)
    assert calls == [BASE_URL + path]
    if method == "compile":
        assert isinstance(result, Program)
        assert result.id == payload["program_id"]
        assert result.status == payload["status"]
        assert result.slug == payload["slug"]
        assert result.timings == payload["timings"]
        assert result.version == payload["version"]
        assert result.runtime_id == payload["runtime_id"]
    else:
        assert result == payload


@pytest.mark.parametrize("payload,code,message,request_id", [
    ({"detail": {"error": "temporarily_unavailable", "message": "Try later.", "request_id": "req-nested"}}, "temporarily_unavailable", "Try later.", "req-nested"),
    ({"error": "temporarily_unavailable", "message": "Try later.", "request_id": "req-top"}, "temporarily_unavailable", "Try later.", "req-top"),
    ({"detail": "Compiler is unavailable."}, None, "Compiler is unavailable.", None),
    ({"detail": {"error": "nested-code"}, "message": "Top-level message.", "request_id": "req-top"}, "nested-code", "Top-level message.", "req-top"),
])
def test_supported_error_body_shapes(payload, code, message, request_id):
    response = make_response(payload=payload)
    error = capture(response)
    assert (error.code, error.message, error.request_id) == (code, message, request_id)
    assert error.response is response
    assert error.request is response.request


@pytest.mark.parametrize("payload", [
    {}, None, [], [BODY_SECRET], BODY_SECRET, 123, True,
    {"detail": []},
    {"detail": [{"loc": ["body", "spec"], "msg": "too short", "input": BODY_SECRET}]},
    {"detail": {"unknown": BODY_SECRET}},
    {"detail": {"error": [BODY_SECRET], "message": {"input": BODY_SECRET}, "request_id": 123}},
    {"error": {"input": BODY_SECRET}, "message": False, "request_id": [BODY_SECRET]},
    {"error": "", "message": "  ", "request_id": ""},
])
def test_malformed_or_validation_body_keeps_safe_httpx_message(payload):
    response = make_response(422, payload=payload)
    with pytest.raises(httpx.HTTPStatusError) as original:
        response.raise_for_status()
    error = capture(response)
    assert error.code is None
    assert error.message is None
    assert error.request_id is None
    assert str(error) == str(original.value)
    assert BODY_SECRET not in str(error)
    assert REQUEST_SECRET not in str(error)
    assert "SECRET-API-KEY" not in str(error)


@pytest.mark.parametrize("body", [b"", b"{broken json", ("<html>" + BODY_SECRET + "</html>").encode(), b"\xff\xfe\xff"])
def test_non_json_body_keeps_safe_httpx_message(body):
    response = make_response(500, body=body)
    with pytest.raises(httpx.HTTPStatusError) as original:
        response.raise_for_status()
    error = capture(response)
    assert str(error) == str(original.value)
    assert error.message is None
    assert error.code is None
    assert BODY_SECRET not in str(error)


def test_request_id_header_fallback_and_original_headers_preserved():
    response = make_response(400, payload={"detail": "Invalid compiler."}, headers={"X-ReQuEsT-Id": "header-id", "Retry-After": "15"})
    error = capture(response)
    assert error.request_id == "header-id"
    assert "header-id" in str(error)
    assert error.message == "Invalid compiler."
    assert error.response.headers["retry-after"] == "15"


def test_only_known_text_fields_are_added_not_other_payload_or_input():
    response = make_response(payload={
        "detail": {"error": "unavailable", "message": "Try later.", "request_id": "req123", "input": BODY_SECRET},
        "spec": REQUEST_SECRET, "traceback": BODY_SECRET,
    })
    error = capture(response)
    assert "Try later." in str(error)
    assert BODY_SECRET not in str(error)
    assert REQUEST_SECRET not in str(error)
    assert error.response.json()["spec"] == REQUEST_SECRET  # Original response remains available explicitly.


def test_redirect_keeps_existing_httpx_behavior():
    response = make_response(302, payload={"detail": "not an API rejection"})
    with pytest.raises(httpx.HTTPStatusError) as caught:
        raise_for_api_status(response)
    assert type(caught.value) is httpx.HTTPStatusError


def test_success_helper_does_not_require_json():
    assert raise_for_api_status(make_response(204, body=b"")) is None


def test_transport_errors_are_not_wrapped_or_retried(monkeypatch):
    request = httpx.Request("POST", BASE_URL + "/api/v1/compile/async")
    expected = httpx.ConnectError("connection failed", request=request)
    calls = []

    def failing_post(*args, **kwargs):
        calls.append(args)
        raise expected

    monkeypatch.setattr(httpx, "post", failing_post)
    with pytest.raises(httpx.ConnectError) as caught:
        PAWClient(api_url=BASE_URL).compile_async(SPEC, compiler="paw-ft-bs48")
    assert caught.value is expected
    assert len(calls) == 1
