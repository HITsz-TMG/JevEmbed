import argparse
import json
from pathlib import Path
import sys

from .client import JevEmbed
from .config import ModelConfig
from .errors import JevEmbedError


def _default_playground_models():
    return (
        ModelConfig(model_id="qwen3-embedding-0.6b", model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
                    aliases=("Qwen/Qwen3-Embedding-0.6B",), expected_dimension=1024,
                    attention_implementation="sdpa"),
        ModelConfig(model_id="jevembed-qwen3-embedding-0.6b",
                    model_name_or_path="HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B",
                    aliases=("HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B", "JevEmbed-Qwen3-Embedding-0.6B"),
                    expected_dimension=1024, attention_implementation="sdpa",
                    max_input_tokens=1024, overflow_policy="truncate"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="JevEmbed local embedding decisions")
    parser.add_argument("--config", action="append", help="YAML config; repeat to register multiple models; --playground has defaults")
    parser.add_argument("--input", default="-", help="Request JSON or - for stdin")
    parser.add_argument("--output", default="-", help="Result JSON or - for stdout")
    parser.add_argument("--explain", action="store_true", help="Compile only, without loading weights")
    parser.add_argument("--trace", action="store_true", help="Return response and diagnostic trace")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--playground", action="store_true", help="Serve the interactive demo at /playground/ (implies --serve)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-request-bytes", type=int, default=2 * 1024 * 1024,
                        help="Maximum HTTP JSON body size (default: 2097152)")
    parser.add_argument("--max-questions", type=int, default=64,
                        help="Maximum questions per HTTP request (default: 64)")
    parser.add_argument("--max-embedding-inputs", type=int, default=4096,
                        help="Maximum compiled embedding inputs per HTTP request (default: 4096)")
    parser.add_argument("--max-concurrent-requests", type=int, default=4,
                        help="Maximum active HTTP inference requests per process (default: 4)")
    parser.add_argument("--body-read-timeout-seconds", type=float, default=30.0,
                        help="Overall HTTP request body read timeout in seconds (default: 30)")
    args = parser.parse_args(argv)
    if not args.config and not args.playground:
        parser.error("--config is required unless --playground is used")
    try:
        client = JevEmbed()
        configs = [ModelConfig.load(path) for path in args.config] if args.config else _default_playground_models()
        for config in configs:
            client.register(config)
        if args.serve or args.playground:
            import uvicorn
            from .server import HTTPServiceLimits, create_app
            limits = HTTPServiceLimits(max_body_bytes=args.max_request_bytes,
                                       max_questions=args.max_questions,
                                       max_embedding_inputs=args.max_embedding_inputs,
                                       max_concurrent_requests=args.max_concurrent_requests,
                                       body_read_timeout_seconds=args.body_read_timeout_seconds)
            options = {"enable_playground": True} if args.playground else {}
            uvicorn.run(create_app(client, limits=limits, **options), host=args.host, port=args.port)
            return 0
        request = json.loads(sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8"))
        result = client.explain(request) if args.explain else (
            client.evaluate_with_trace(request) if args.trace else client.evaluate(request))
        output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output == "-":
            sys.stdout.write(output)
        else:
            Path(args.output).write_text(output, encoding="utf-8")
        return 0
    except (JevEmbedError, ValueError, OSError, ImportError) as exc:
        print(f"jevembed: {exc}", file=sys.stderr)
        return 2
