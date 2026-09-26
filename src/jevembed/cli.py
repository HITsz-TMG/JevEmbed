import argparse
import json
from pathlib import Path
import sys

from .client import JevEmbed
from .config import ModelConfig
from .errors import JevEmbedError


def main(argv=None):
    parser = argparse.ArgumentParser(description="JevEmbed local embedding decisions")
    parser.add_argument("--config", action="append", required=True, help="YAML config; repeat to register multiple models")
    parser.add_argument("--input", default="-", help="Request JSON or - for stdin")
    parser.add_argument("--output", default="-", help="Result JSON or - for stdout")
    parser.add_argument("--explain", action="store_true", help="Compile only, without loading weights")
    parser.add_argument("--trace", action="store_true", help="Return response and diagnostic trace")
    parser.add_argument("--serve", action="store_true")
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
    try:
        client = JevEmbed()
        for path in args.config:
            client.register(ModelConfig.load(path))
        if args.serve:
            import uvicorn
            from .server import HTTPServiceLimits, create_app
            limits = HTTPServiceLimits(max_body_bytes=args.max_request_bytes,
                                       max_questions=args.max_questions,
                                       max_embedding_inputs=args.max_embedding_inputs,
                                       max_concurrent_requests=args.max_concurrent_requests,
                                       body_read_timeout_seconds=args.body_read_timeout_seconds)
            uvicorn.run(create_app(client, limits=limits), host=args.host, port=args.port)
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
