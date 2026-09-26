"""Generate strict, auditable JevEmbed training records from a fixed task."""

import argparse
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import tempfile

from ..errors import ValidationError
from .config import load_config, validate_config
from .provider import ChatTeacher, TeacherError
from .quotas import quota_answers
from .validation import parse_json, validate_answer, validate_generation


class _ReferenceValidatorFailure(Exception):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _atomic(path, content):
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _lock(handle):
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise ValidationError("Synthesis output is locked by another process") from exc


def _answer_format(question):
    if question["type"] == "choice":
        return ("The value of answer MUST be a JSON object with exactly one field, "
                f"choice, whose string value is one exact key from {list(question['criteria'])}. "
                'Example shape: {"answer":{"choice":"<candidate key>"}}. '
                "Replace the placeholder with the selected key; never return a scalar answer.")
    if question["type"] == "score":
        return ("The value of answer MUST be a JSON object with exactly one field, "
                f"level, whose value is an integer from 0 to {len(question['criteria']) - 1}. "
                'Example shape: {"answer":{"level":0}}. '
                "The shown zero illustrates syntax only; compute the actual level. "
                "Never return a scalar answer.")
    return ("The value of answer MUST be a JSON object with exactly one field, "
            'noul, whose value is a JSON boolean. Example shapes: {"answer":{"noul":true}} '
            'or {"answer":{"noul":false}}. Compute the actual value; '
            "never return a scalar answer.")


def _answer_schema(question):
    if question["type"] == "choice":
        field = "choice"
        value = {"type": "string", "enum": list(question["criteria"])}
    elif question["type"] == "score":
        field = "level"
        value = {"type": "integer", "minimum": 0, "maximum": len(question["criteria"]) - 1}
    else:
        field = "noul"
        value = {"type": "boolean"}
    return {"type": "object", "additionalProperties": False,
            "required": [field], "properties": {field: value}}


def _response_shape(question, *, verification):
    if question["type"] == "choice":
        answer = '{"choice":"<one exact candidate key>"}'
    elif question["type"] == "score":
        answer = '{"level":<integer in allowed range>}'
    else:
        answer = '{"noul":<true or false>}'
    if verification:
        return '{"answer":' + answer + '}'
    return '{"state":<state matching state_schema>,"answer":' + answer + '}'


def _messages(config, case_type, index, *, state=None, target_answer=None, attempt=0):
    if state is not None and target_answer is not None:
        raise ValueError("Verifier must not receive a target answer")
    task = config["task"]
    question = task["question"]
    common = {"domain": task["domain"], "question": question,
              "required_answer_format": _answer_format(question),
              "response_shape_replace_placeholders": _response_shape(question, verification=state is not None),
              "required_response_schema": {"type": "object", "additionalProperties": False,
                   "required": ["answer"] if state is not None else ["state", "answer"],
                   "properties": ({"answer": _answer_schema(question)} if state is not None else
                                  {"state": {"type": task["state_schema"]["type"],
                                             "description": "Must also satisfy the separate state_schema"},
                                   "answer": _answer_schema(question)})},
              "state_description": task["state_description"], "state_schema": task["state_schema"],
              "labeling_guidance": task["labeling_guidance"]}
    if state is None:
        instruction = ("Create one realistic, distinct state and its hard answer for the fixed question. "
                       "Return only valid JSON with exactly two top-level fields: state and answer. "
                       "The answer field MUST contain the nested hard-answer object specified in "
                       "required_answer_format, never a string, number, or boolean directly. "
                       "If the question cannot be answered unambiguously from the state and guidance, "
                       "return only {\"reject\":\"ambiguous\"}. Do not change the question. "
                       "Use the requested case type as a diversity cue, never as a reason to invent a label. "
                       "Do not put a label, answer, rationale, or target hint into the state. "
                       "Calculate any arithmetic and apply the fixed question's rule before labeling. "
                       "Check the response against required_response_schema.")
        common.update({"case_type": case_type, "sample_number": index,
                       "attempt_number": attempt + 1})
        if target_answer is not None:
            instruction += (" Generate a state that genuinely satisfies required_target_answer "
                            "under the fixed question. Compute the answer from the state; do not "
                            "merely echo the requested label. If no unambiguous matching state "
                            "can be produced, return {\"reject\":\"ambiguous\"}.")
            if question["type"] == "noul":
                if target_answer["noul"]:
                    instruction += (" For this sample, the correct answer must be true. "
                                    "Construct a clear positive example of the fixed question, "
                                    "not a negative example labeled true. The state must provide "
                                    "enough information to decide without inventing context.")
                else:
                    instruction += (" For this sample, the correct answer must be false. "
                                    "Construct a clear negative example of the fixed question, "
                                    "not a positive example labeled false. The state must provide "
                                    "enough information to decide without inventing context.")
            elif question["type"] == "choice":
                label = json.dumps(target_answer["choice"], ensure_ascii=False)
                instruction += (f" For this sample, the correct answer must be the exact candidate key {label}. "
                                "Construct a state that makes this key correct under the fixed question.")
            else:
                instruction += (f" For this sample, the correct answer must be zero-based level "
                                f"{target_answer['level']} in the ordered criteria. Construct a state "
                                "that warrants this level under the fixed question.")
            common["required_target_answer"] = target_answer
    else:
        instruction = ("Independently label this state for the fixed question. Return only valid "
                       "JSON with exactly one top-level field, answer, whose value MUST be the "
                       "nested hard-answer object specified in required_answer_format, never a "
                       "string, number, or boolean directly. If ambiguous, return only "
                       "{\"reject\":\"ambiguous\"}. No rationale. "
                       "Calculate any arithmetic and apply the fixed question's rule before labeling. "
                       "Check the response against required_response_schema. "
                       "The generated answer is intentionally withheld.")
        common["state"] = state
    return [{"role": "system", "content": instruction},
            {"role": "user", "content": _json(common)}]


def _prompt_hash(config):
    case = config["generation"]["case_types"][0]
    quotas = quota_answers(config)
    if quotas is None:
        question = config["task"]["question"]
        if question["type"] == "choice":
            targets = [{"choice": next(iter(question["criteria"]))}]
        elif question["type"] == "score":
            targets = [{"level": 0}]
        else:
            targets = [{"noul": True}]
    else:
        targets = [answer for answer, _ in quotas]
    # The config hash already covers the full label vocabulary. Prompt identity
    # needs only representative target branches; expanding every label would
    # repeat a large Choice question quadratically.
    targets = [targets[0]] + ([targets[-1]] if targets[-1] != targets[0] else [])
    return _hash({"untargeted": [_messages(config, case, 0, attempt=i) for i in (0, 1)],
                  "targeted": [[_messages(config, case, 0, target_answer=answer, attempt=i)
                                for i in (0, 1)] for answer in targets],
                  "verification": _messages(config, case, 0, state={"sample_state": True})})


def _allocation(config):
    """Map each deterministic slot to (case type, optional target answer)."""
    generation = config["generation"]
    cases = generation["case_types"]
    order = list(range(generation["count"]))
    random.Random(generation["seed"]).shuffle(order)
    quotas = quota_answers(config)
    if quotas is None:
        return {slot: (cases[position % len(cases)], None)
                for position, slot in enumerate(order)}
    allocation = {}
    cursor = 0
    for label_index, (answer, count) in enumerate(quotas):
        for within_label, slot in enumerate(order[cursor:cursor + count]):
            allocation[slot] = (cases[(within_label + label_index) % len(cases)], answer)
        cursor += count
    assert cursor == generation["count"]
    return allocation


def _check_reference(sample_validator, state, answer, question):
    if sample_validator is None:
        return True
    try:
        result = sample_validator(deepcopy(state), deepcopy(answer), deepcopy(question))
    except Exception:
        raise _ReferenceValidatorFailure("Trusted sample validator failed") from None
    if type(result) is not bool:
        raise _ReferenceValidatorFailure("Trusted sample validator returned a non-boolean value")
    return result


def _teacher_reply(teacher, messages, tally):
    tally["calls"] += 1
    tally["calls_without_usage"] += 1
    try:
        text, usage = teacher.complete(messages)
    finally:
        if hasattr(teacher, "last_http_requests"):
            tally["http_requests"] += teacher.last_http_requests()
    if usage is not None and all(type(usage.get(key)) is int and usage[key] >= 0
                                 for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        tally["calls_without_usage"] -= 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                tally[key] += value
    return parse_json(text)


def _sample(index, case_type, target_answer, config, teacher, sample_validator):
    tally = {"calls": 0, "calls_without_usage": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0, "http_requests": 0}
    reasons = {}
    for attempt in range(config["generation"]["max_attempts_per_sample"]):
        try:
            payload = _teacher_reply(teacher, _messages(config, case_type, index,
                                                        target_answer=target_answer, attempt=attempt), tally)
            if isinstance(payload, dict) and set(payload) == {"reject"}:
                reason = "teacher_rejected"
            else:
                state, answer = validate_generation(payload, config)
                if target_answer is not None and answer != target_answer:
                    reason = "target_label_mismatch"
                elif config["generation"]["verify_labels"]:
                    verification = _teacher_reply(teacher, _messages(config, case_type, index, state=state), tally)
                    if isinstance(verification, dict) and set(verification) == {"reject"}:
                        reason = "verifier_rejected"
                    elif not isinstance(verification, dict) or set(verification) != {"answer"}:
                        reason = "invalid_verification"
                    else:
                        validate_answer(verification["answer"], config)
                        if verification["answer"] != answer:
                            reason = "label_disagreement"
                        elif not _check_reference(sample_validator, state, answer,
                                                  config["task"]["question"]):
                            reason = "reference_rejected"
                        else:
                            return {"slot": index, "case_type": case_type, "state": state,
                                    "answer": answer, "tally": tally, "reasons": reasons}
                else:
                    if not _check_reference(sample_validator, state, answer,
                                            config["task"]["question"]):
                        reason = "reference_rejected"
                    else:
                        return {"slot": index, "case_type": case_type, "state": state,
                                "answer": answer, "tally": tally, "reasons": reasons}
        except (ValidationError, ValueError, TypeError, KeyError):
            reason = "invalid_generation"
        except TeacherError as exc:
            if exc.fatal:
                raise
            reason = "teacher_error"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"slot": index, "case_type": case_type, "tally": tally, "reasons": reasons}


def _record(event, config):
    task = config["task"]
    return {"id": f"{task['id']}-{event['slot']:06d}", "group": task["id"],
            "request": {"state": event["state"],
                        "questions": {task["question_id"]: task["question"]}},
            "answers": {task["question_id"]: event["answer"]}}


def _results(events, config, validator_identity=None):
    """Recompute dedup/quarantine from the journal, independent of completion order."""
    by_fingerprint = {}
    conflicts = set()
    reasons = {}
    tally = {"calls": 0, "calls_without_usage": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0, "http_requests": 0}
    for event in events.values():
        for key, value in event["tally"].items():
            tally[key] += value
        for key, value in event["reasons"].items():
            reasons[key] = reasons.get(key, 0) + value
    for event in sorted(events.values(), key=lambda item: item["slot"]):
        if "state" not in event:
            continue
        fingerprint = _hash({"state": event["state"], "question": config["task"]["question"]})
        if fingerprint in conflicts:
            reasons["conflicting_duplicate"] = reasons.get("conflicting_duplicate", 0) + 1
            continue
        previous = by_fingerprint.get(fingerprint)
        if previous is None:
            by_fingerprint[fingerprint] = event
        elif previous["answer"] == event["answer"]:
            reasons["duplicate"] = reasons.get("duplicate", 0) + 1
        else:
            conflicts.add(fingerprint)
            del by_fingerprint[fingerprint]
            reasons["conflicting_duplicate"] = reasons.get("conflicting_duplicate", 0) + 2
    accepted = sorted(by_fingerprint.values(), key=lambda item: item["slot"])
    labels, cases = {}, {}
    attempted_cases = {}
    for event in events.values():
        key = event["case_type"]
        attempted_cases[key] = attempted_cases.get(key, 0) + 1
    for event in accepted:
        label = _json(event["answer"])
        labels[label] = labels.get(label, 0) + 1
        cases[event["case_type"]] = cases.get(event["case_type"], 0) + 1
    quotas = quota_answers(config)
    quota_labels = None if quotas is None else [
        {"answer": answer, "requested": count,
         "accepted": labels.get(_json(answer), 0),
         "shortfall": count - labels.get(_json(answer), 0)}
        for answer, count in quotas]
    report = {"task_id": config["task"]["id"], "domain": config["task"]["domain"],
              "task_type": config["task"]["question"]["type"],
              "state_kind": config["task"]["state_schema"]["type"],
              "requested": config["generation"]["count"], "processed": len(events),
              "complete": len(events) == config["generation"]["count"],
              "accepted": len(accepted), "shortfall": config["generation"]["count"] - len(accepted),
              "case_types": cases, "case_types_attempted": attempted_cases,
              "labels": labels, "rejections": reasons,
              "quota_labels": quota_labels,
              "quota_satisfied": None if quota_labels is None else all(
                  item["shortfall"] == 0 for item in quota_labels),
              "teacher_usage": tally, "verification_enabled": config["generation"]["verify_labels"],
              "validator_enabled": validator_identity is not None,
              "validator_identity": validator_identity,
              "usage_scope": "Journaled slots only; in-flight or fatal requests before journaling are excluded.",
              "verification_note": "Same-teacher agreement does not establish ground truth."}
    return [_record(event, config) for event in accepted], report


def _load_journal(path, config, allocation, sample_validator):
    events = {}
    if not path.exists():
        return events
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[:raw.rfind(b"\n") + 1]
        path.write_bytes(raw)
    for line in raw.splitlines():
        try:
            event = parse_json(line.decode("utf-8"))
            slot = event["slot"]
            if type(slot) is not int or not 0 <= slot < config["generation"]["count"] or slot in events:
                raise ValueError("invalid slot")
            keys = {"slot", "case_type", "tally", "reasons"}
            if set(event) not in (keys, keys | {"state", "answer"}):
                raise ValueError("invalid event fields")
            case_type, target_answer = allocation[slot]
            if event["case_type"] != case_type:
                raise ValueError("invalid case type")
            tally_keys = {"calls", "calls_without_usage", "prompt_tokens", "completion_tokens",
                          "total_tokens", "http_requests"}
            if type(event["tally"]) is not dict or set(event["tally"]) != tally_keys or any(
                type(value) is not int or value < 0 for value in event["tally"].values()
            ):
                raise ValueError("invalid usage tally")
            if type(event["reasons"]) is not dict or any(
                type(key) is not str or type(value) is not int or value < 0
                for key, value in event["reasons"].items()
            ):
                raise ValueError("invalid rejection counts")
            if "state" in event:
                validate_generation({"state": event["state"], "answer": event["answer"]}, config)
                if target_answer is not None and event["answer"] != target_answer:
                    raise ValueError("accepted answer differs from assigned quota label")
                if not _check_reference(sample_validator, event["state"], event["answer"],
                                        config["task"]["question"]):
                    raise ValueError("accepted event fails trusted validator")
            events[slot] = event
        except _ReferenceValidatorFailure:
            raise ValidationError("Trusted sample validator failed while checking journal") from None
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise ValidationError("Invalid synthesis journal; resume aborted") from exc
    return events


def _write_outputs(output, events, config, validator_identity=None):
    records, report = _results(events, config, validator_identity)
    _atomic(output / "data.jsonl", "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records))
    _atomic(output / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def run(config, output=None, *, resume=False, validate_only=False, teacher=None,
        sample_validator=None, validator_id=None):
    """Run or validate a synthesis job. Inject ``teacher.complete`` for offline tests."""
    config = load_config(config) if isinstance(config, (str, Path)) else validate_config(config)
    if sample_validator is None:
        if validator_id is not None:
            raise ValidationError("validator_id requires sample_validator")
        validator_identity = None
    else:
        if not callable(sample_validator) or type(validator_id) is not str or not validator_id.strip():
            raise ValidationError("sample_validator requires a callable and stable nonempty validator_id")
        validator_identity = hashlib.sha256(validator_id.encode("utf-8")).hexdigest()
    if validate_only:
        maximum_calls = (config["generation"]["count"]
                         * config["generation"]["max_attempts_per_sample"]
                         * (2 if config["generation"]["verify_labels"] else 1))
        maximum_requests = maximum_calls * (config["teacher"]["max_retries"] + 1)
        quotas = quota_answers(config)
        return {"valid": True, "task_id": config["task"]["id"],
                "requested": config["generation"]["count"],
                "quota_labels": None if quotas is None else [
                    {"answer": answer, "requested": count} for answer, count in quotas],
                "validator_enabled": validator_identity is not None,
                "validator_identity": validator_identity,
                "maximum_teacher_calls": maximum_calls,
                "maximum_http_requests": maximum_requests,
                "maximum_completion_tokens_requested": maximum_requests * config["teacher"]["max_tokens"]}
    if output is None:
        raise ValidationError("--output is required unless --validate-only is set")
    output = Path(output)
    if resume and not output.exists():
        raise ValidationError("Cannot resume: output directory does not exist")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+b") as lock:
        _lock(lock)
        manifest_path = output / "manifest.json"
        fingerprint = _hash(config)
        prompt_fingerprint = _prompt_hash(config)
        if resume:
            if not manifest_path.exists():
                raise ValidationError("Cannot resume: manifest is missing")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValidationError("Cannot resume: manifest is invalid") from exc
            if (manifest.get("config_sha256") != fingerprint or manifest.get("format_version") != 3
                    or manifest.get("prompt_sha256") != prompt_fingerprint
                    or manifest.get("validator_identity") != validator_identity):
                raise ValidationError("Cannot resume: configuration or format differs")
        else:
            if any(path.name != ".lock" for path in output.iterdir()):
                raise ValidationError("Output is not empty; use --resume for the same configuration")
            manifest = {"format_version": 3, "config_sha256": fingerprint,
                        "prompt_sha256": prompt_fingerprint,
                        "validator_identity": validator_identity,
                        "teacher_model": config["teacher"]["model"],
                        "task_id": config["task"]["id"], "question_id": config["task"]["question_id"],
                        "question_type": config["task"]["question"]["type"],
                        "count": config["generation"]["count"]}
            _atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        allocation = _allocation(config)
        journal_path = output / "journal.jsonl"
        if resume and not journal_path.exists() and any(
            (output / name).exists() and (output / name).stat().st_size
            for name in ("data.jsonl", "report.json")
        ):
            raise ValidationError("Cannot resume: progress journal is missing")
        events = _load_journal(journal_path, config, allocation, sample_validator)
        pending = [slot for slot in range(config["generation"]["count"]) if slot not in events]
        if not pending:
            return _write_outputs(output, events, config, validator_identity)
        own_teacher = teacher is None
        teacher = teacher or ChatTeacher(config["teacher"])
        try:
            parallelism = config["generation"]["concurrency"]
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                active = {}
                remaining = iter(pending)
                while True:
                    while len(active) < parallelism:
                        slot = next(remaining, None)
                        if slot is None:
                            break
                        case_type, target_answer = allocation[slot]
                        active[executor.submit(_sample, slot, case_type, target_answer, config, teacher,
                                               sample_validator)] = slot
                    if not active:
                        break
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        slot = active.pop(future)
                        event = future.result()
                        with (output / "journal.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(_json(event) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        events[slot] = event
                        print(f"Synthesis {len(events)}/{config['generation']['count']} slots complete",
                              file=sys.stderr, flush=True)
            return _write_outputs(output, events, config, validator_identity)
        except _ReferenceValidatorFailure:
            _write_outputs(output, events, config, validator_identity)
            raise ValidationError("Trusted sample validator failed or returned a non-boolean value") from None
        except BaseException:
            _write_outputs(output, events, config, validator_identity)
            raise
        finally:
            if own_teacher:
                teacher.close()


def _load_cli_validator(spec):
    if ":" not in spec:
        raise ValidationError("--validator must be module:function")
    module_name, function_name = spec.rsplit(":", 1)
    if not module_name or not function_name.isidentifier():
        raise ValidationError("--validator must be module:function")
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        source = inspect.getsourcefile(function)
        if not inspect.isfunction(function) or source is None:
            raise ValueError("not a Python source function")
        file_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    except Exception:
        raise ValidationError("Cannot load trusted --validator function") from None
    return function, "cli-v1:" + file_hash + ":" + function.__qualname__


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate JevEmbed supervision for one fixed question")
    parser.add_argument("--config", required=True, help="Synthesis YAML configuration")
    parser.add_argument("--output", help="Directory for data.jsonl, report, and resume journal")
    parser.add_argument("--validate-only", action="store_true", help="Validate config without contacting teacher")
    parser.add_argument("--resume", action="store_true", help="Resume matching output directory")
    parser.add_argument("--validator", help="Trusted Python module:function for reference label checks")
    args = parser.parse_args(argv)
    try:
        sample_validator, validator_id = _load_cli_validator(args.validator) if args.validator else (None, None)
        report = run(args.config, args.output, resume=args.resume, validate_only=args.validate_only,
                     sample_validator=sample_validator, validator_id=validator_id)
    except (ValidationError, TeacherError, OSError) as exc:
        parser.exit(2, f"synthesis: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
