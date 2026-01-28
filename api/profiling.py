"""ProfileManager for pyinstrument profiling with S3 upload support."""

import os
import uuid

import boto3
from pyinstrument import Profiler
from pyinstrument.renderers import SpeedscopeRenderer


class ProfileManager:
    """Singleton class for managing pyinstrument profiling sessions."""

    _instance = None

    def __init__(self):
        """Initialize ProfileManager with environment configuration."""
        enable_str = os.environ.get("ENABLE_PROFILING", "false").lower()
        self.enabled = enable_str == "true"
        self.startup_id = uuid.uuid4().hex[:8]
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
        """Render and upload profile data to S3."""
        renderer = SpeedscopeRenderer()
        output = self.profiler.output(renderer)

        s3_key = f"profiles/{self.current_profile_name}/{self.current_profile_id}.speedscope"

        endpoint_url = os.environ.get("BUCKET_ENDPOINT_URL")
        bucket_name = os.environ.get("BUCKET_NAME")
        access_key_id = os.environ.get("BUCKET_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY")

        if not all([endpoint_url, bucket_name, access_key_id, secret_access_key]):
            raise RuntimeError(
                "Missing S3 credentials. Required env vars: "
                "BUCKET_ENDPOINT_URL, BUCKET_NAME, BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY"
            )

        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=output,
            ContentType="application/json",
        )

        print(f"[profiling] Uploaded {s3_key}")
