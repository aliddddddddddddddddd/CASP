"""Model backends. Each returns a dict:
   {text, input_tokens, output_tokens, cache_write_tokens, cache_read_tokens,
    latency_s, stop_reason, error}
Raw HTTP is used (requests) so results do not depend on SDK versions.
"""
import copy
import json
import os
import random
import re
import time

import requests

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _post(url, headers, payload, timeout=180, max_tries=6):
    delay = 2.0
    last = None
    for attempt in range(max_tries):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last = f"network: {e}"
            time.sleep(delay); delay *= 2
            continue
        dt = time.time() - t0
        if r.status_code == 200:
            return r.json(), dt, None
        last = f"HTTP {r.status_code}: {r.text[:500]}"
        if r.status_code in RETRY_STATUS:
            time.sleep(delay + random.random()); delay = min(delay * 2, 60)
            continue
        return None, dt, last
    return None, 0.0, last


def _empty(err=None):
    return dict(text="", input_tokens=0, output_tokens=0, cache_write_tokens=0,
                cache_read_tokens=0, latency_s=0.0, stop_reason=None, error=err)


# ------------------------------------------------------------------ Anthropic (Claude)
def call_anthropic(cfg, prompt, use_cache=False, temperature=1.0, max_tokens=1024):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _empty("ANTHROPIC_API_KEY not set")
    user_content = []
    if prompt.cache_prefix:
        block = {"type": "text", "text": prompt.cache_prefix}
        if use_cache:
            block["cache_control"] = {"type": "ephemeral"}
        user_content.append(block)
    user_content.append({"type": "text", "text": prompt.user})
    messages = [{"role": "user", "content": user_content}]
    if prompt.prefill:
        messages.append({"role": "assistant", "content": prompt.prefill})
    payload = {"model": cfg["model"], "max_tokens": max_tokens, "messages": messages}
    if temperature is not None and cfg.get("send_temperature", True):
        payload["temperature"] = temperature
    if prompt.system:
        payload["system"] = prompt.system
    if cfg.get("thinking") is not None:
        payload["thinking"] = cfg["thinking"]
    if prompt.schema is not None:
        payload["output_config"] = {"format": {"type": "json_schema", "schema": prompt.schema}}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    data, dt, err = _post("https://api.anthropic.com/v1/messages", headers, payload)
    if err:
        out = _empty(err); out["latency_s"] = dt
        return out
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage", {})
    return dict(text=(prompt.prefill + text) if prompt.prefill else text,
                input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                cache_write_tokens=u.get("cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=u.get("cache_read_input_tokens", 0) or 0,
                latency_s=dt, stop_reason=data.get("stop_reason"), error=None)


# ------------------------------------------------------------------ OpenAI (optional)
def call_openai(cfg, prompt, use_cache=False, temperature=1.0, max_tokens=1024):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return _empty("OPENAI_API_KEY not set")
    user = (prompt.cache_prefix + "\n\n" if prompt.cache_prefix else "") + prompt.user
    messages = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": user})
    payload = {"model": cfg["model"], "messages": messages,
               cfg.get("max_tokens_field", "max_completion_tokens"): max_tokens}
    if temperature is not None and cfg.get("send_temperature", True):
        payload["temperature"] = temperature
    if prompt.schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "result", "strict": True, "schema": prompt.schema}}
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    data, dt, err = _post("https://api.openai.com/v1/chat/completions", headers, payload)
    if err:
        out = _empty(err); out["latency_s"] = dt
        return out
    ch = data["choices"][0]
    u = data.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return dict(text=ch["message"].get("content") or "", input_tokens=u.get("prompt_tokens", 0),
                output_tokens=u.get("completion_tokens", 0), cache_write_tokens=0,
                cache_read_tokens=cached, latency_s=dt, stop_reason=ch.get("finish_reason"),
                error=None)


# ------------------------------------------------------------------ Gemini (optional)
def _gemini_schema(s):
    s = copy.deepcopy(s)
    def strip(o):
        if isinstance(o, dict):
            o.pop("additionalProperties", None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for v in o:
                strip(v)
    strip(s)
    return s


def call_gemini(cfg, prompt, use_cache=False, temperature=1.0, max_tokens=1024):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return _empty("GEMINI_API_KEY not set")
    user = (prompt.cache_prefix + "\n\n" if prompt.cache_prefix else "") + prompt.user
    gen = {"maxOutputTokens": max_tokens}
    if temperature is not None:
        gen["temperature"] = temperature
    if prompt.schema is not None:
        gen["responseMimeType"] = "application/json"
        gen["responseSchema"] = _gemini_schema(prompt.schema)
    if cfg.get("thinking_budget") is not None:
        gen["thinkingConfig"] = {"thinkingBudget": cfg["thinking_budget"]}
    payload = {"contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen}
    if prompt.system:
        payload["systemInstruction"] = {"parts": [{"text": prompt.system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent?key={key}"
    data, dt, err = _post(url, {"content-type": "application/json"}, payload)
    if err:
        out = _empty(err.replace(key, "***")); out["latency_s"] = dt
        return out
    try:
        cand = data["candidates"][0]
        text = "".join(p.get("text", "") for p in cand["content"]["parts"])
    except (KeyError, IndexError):
        cand, text = {}, ""
    u = data.get("usageMetadata", {})
    return dict(text=text, input_tokens=u.get("promptTokenCount", 0),
                output_tokens=u.get("candidatesTokenCount", 0), cache_write_tokens=0,
                cache_read_tokens=u.get("cachedContentTokenCount", 0) or 0, latency_s=dt,
                stop_reason=cand.get("finishReason"), error=None)


# ------------------------------------------------------------------ Mock (pipeline test ONLY)
def call_mock(cfg, prompt, use_cache=False, temperature=1.0, max_tokens=1024, item=None):
    """Synthetic responses used only to verify that the pipeline runs end to end.
    Mock output is NEVER to be reported as a result."""
    rng = random.Random(hash((prompt.user, prompt.system, prompt.prefill, str(prompt.schema),
                              time.time_ns())))
    body = prompt.user
    if item is not None and isinstance(item["gold"], (int, float)):
        gold = item["gold"] if item else 0
        ans = gold if rng.random() < 0.8 else gold + rng.choice([-2, 1, 10])
        if prompt.schema:
            text = json.dumps({"reasoning": "mock", "final_answer": ans}
                              if "reasoning" in prompt.schema["properties"] else {"final_answer": ans})
        elif "<answer>" in body:
            text = f"<thinking>mock</thinking><answer>{ans:g}</answer>"
        else:
            text = f"## Answer\n{ans:g}"
    else:
        gold = item["gold"] if item else {"persons": [], "organizations": [], "locations": []}
        pred = {k: [e for e in v if rng.random() < 0.85] for k, v in gold.items()}
        if "BANANA42" in body and rng.random() < 0.1:
            text = "BANANA42"
        elif prompt.schema or prompt.prefill:
            text = json.dumps(pred)
        else:
            text = ("Here is the JSON:\n" if rng.random() < 0.4 else "") + json.dumps(pred)
    return dict(text=text, input_tokens=len(body) // 4, output_tokens=len(text) // 4,
                cache_write_tokens=0, cache_read_tokens=(len(prompt.cache_prefix) // 4) if use_cache else 0,
                latency_s=rng.uniform(0.5, 2.0), stop_reason="end_turn", error=None)


PROVIDERS = {"anthropic": call_anthropic, "openai": call_openai, "gemini": call_gemini,
             "mock": call_mock}
