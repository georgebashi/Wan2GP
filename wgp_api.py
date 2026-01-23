#!/usr/bin/env python3
"""
WanGP HTTP API Server

A persistent HTTP server for WanGP video generation that allows:
- Loading models once at startup
- Processing generation requests quickly with pre-loaded models
- REST API interface for external applications

Usage:
    python wgp_api.py [options]

Examples:
    # Start server with default settings
    python wgp_api.py

    # Start server and preload a model
    python wgp_api.py --preload t2v

    # Start server on custom port with specific memory profile
    python wgp_api.py --port 8080 --preload i2v --profile 3

Options:
    --host HOST         Server host (default: 0.0.0.0)
    --port PORT         Server port (default: 8000)
    --preload MODEL     Preload model on startup (e.g., 't2v', 'i2v')
    --profile PROFILE   Memory profile for preloaded model (0-5, default: -1 for auto)

API Endpoints:
    GET  /health        - Health check and server info
    GET  /models        - List available models
    POST /models/load   - Load a model
    POST /generate      - Generate video/image
    GET  /files/{path}  - Download generated files
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="WanGP HTTP API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wgp_api.py                          # Start with defaults
  python wgp_api.py --preload t2v            # Preload text-to-video model
  python wgp_api.py --port 8080 --preload i2v --profile 3
        """
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--preload",
        type=str,
        default=None,
        help="Model to preload on startup (e.g., 't2v', 'i2v')"
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=-1,
        help="Memory profile (0-5), -1 for auto (default: -1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Set environment variables for the API server
    if args.preload:
        os.environ["WANGP_PRELOAD_MODEL"] = args.preload
    os.environ["WANGP_PROFILE"] = str(args.profile)

    # Import uvicorn here to allow --help without dependencies
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install API dependencies:")
        print("  pip install -r requirements-api.txt")
        sys.exit(1)

    print("=" * 60)
    print("WanGP HTTP API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    if args.preload:
        print(f"Preload model: {args.preload}")
        print(f"Memory profile: {args.profile if args.profile >= 0 else 'auto'}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
