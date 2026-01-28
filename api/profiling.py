"""ProfileManager for pyinstrument profiling with S3 upload support."""

import os
import time
import uuid

import boto3
from botocore.config import Config
from pyinstrument import Profiler
from pyinstrument.renderers import SpeedscopeRenderer


class ProfileManager:
    """Singleton class for managing pyinstrument profiling sessions."""

    _instance = None

    def __init__(self):
        """Initialize ProfileManager with environment configuration."""
        enable_str = os.environ.get("ENABLE_PROFILING", "false").lower()
        self.enabled = enable_str == "true"
        # Use RunPod pod ID for startup profile, fall back to random UUID
        self.startup_id = os.environ.get("RUNPOD_POD_ID", uuid.uuid4().hex[:8])
        self.profiler = None
        self.current_profile_name = None
        self.current_profile_id = None

    @classmethod
    def get_instance(cls):
        """Get or create the singleton ProfileManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, name: str, profile_id: str):
        """Start a new profiling session.

        Args:
            name: The profile name (e.g., "startup", "generate")
            profile_id: Unique identifier for this profile session
        """
        if not self.enabled:
            return

        if self.profiler is not None:
            raise RuntimeError("Profiler already active")

        self.profiler = Profiler(async_mode="enabled")
        self.current_profile_name = name
        self.current_profile_id = profile_id
        self.profiler.start()

    def stop(self):
        """Stop the current profiling session and upload results."""
        if not self.enabled or self.profiler is None:
            return

        self.profiler.stop()
        self._upload_profile()
        self.profiler = None
        self.current_profile_name = None
        self.current_profile_id = None

    def start_startup_profile(self):
        """Start profiling the startup phase."""
        self.start("startup", self.startup_id)

    def stop_startup_profile(self):
        """Stop profiling the startup phase."""
        self.stop()

    def _upload_profile(self):
        """Render and upload profile data to S3.

        Uses the same S3 configuration as RunPod's rp_upload utility:
        - BUCKET_ENDPOINT_URL
        - BUCKET_ACCESS_KEY_ID
        - BUCKET_SECRET_ACCESS_KEY
        - Bucket name defaults to current month-year (e.g., "01-26")
        """
        renderer = SpeedscopeRenderer()
        output = self.profiler.output(renderer)

        s3_key = f"profiles/{self.current_profile_name}/{self.current_profile_id}.speedscope"

        endpoint_url = os.environ.get("BUCKET_ENDPOINT_URL")
        access_key_id = os.environ.get("BUCKET_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
        # Match RunPod's bucket naming convention (month-year)
        bucket_name = os.environ.get("BUCKET_NAME", time.strftime("%m-%y"))

        if not all([endpoint_url, access_key_id, secret_access_key]):
            raise RuntimeError(
                "Missing S3 credentials. Required env vars: "
                "BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY"
            )

        boto_config = Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"}
        )

        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=boto_config,
        )

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=output,
            ContentType="application/json",
        )

        print(f"[profiling] Uploaded s3://{bucket_name}/{s3_key}", flush=True)
