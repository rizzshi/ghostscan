"""Upload GhostScan application and checkpoints to Hugging Face Spaces.

This script uses `huggingface_hub` to create and populate a Hugging Face Space
repository with the GhostScan Gradio app, models in `checkpoints/`, and supporting code.
"""

import argparse
import os
import sys
from huggingface_hub import HfApi


def push_to_hub(repo_id: str, token: str = None, private: bool = False) -> None:
    """Upload checkpoints, app.py, and dependencies to a Hugging Face Space.

    Args:
        repo_id: Hugging Face repository identifier in format 'username/space-name'.
        token: Hugging Face API access token with write permissions.
            Defaults to HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variables.
        private: Whether to set the created Space repository to private.
    """
    auth_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not auth_token:
        print("Error: No Hugging Face token provided.")
        print("Please pass --token <your_token> or set the HF_TOKEN environment variable.")
        sys.exit(1)

    api = HfApi(token=auth_token)

    print(f"Ensuring Hugging Face Space repository exists: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
        )
        print("Space repository verified.")
    except Exception as e:
        print(f"Notice during repository creation: {e}")

    # Core files required to run the Space
    files_to_upload = [
        "app.py",
        "contracts.py",
        "victim_model.py",
        "detector.py",
        "attacks.py",
        "explain.py",
        "requirements.txt",
        "README.md",
    ]

    print("\nUploading application code files...")
    for filename in files_to_upload:
        if os.path.exists(filename):
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="space",
            )
        else:
            print(f"Warning: {filename} not found locally, skipping.")

    # Upload checkpoints directory
    if os.path.exists("checkpoints") and os.listdir("checkpoints"):
        print("\nUploading model checkpoints from checkpoints/...")
        api.upload_folder(
            folder_path="checkpoints",
            path_in_repo="checkpoints",
            repo_id=repo_id,
            repo_type="space",
        )
        print("Checkpoints uploaded successfully.")
    else:
        print("Warning: 'checkpoints/' directory is missing or empty.")

    # Upload sample images directory
    if os.path.exists("samples") and os.listdir("samples"):
        print("\nUploading sample images from samples/...")
        api.upload_folder(
            folder_path="samples",
            path_in_repo="samples",
            repo_id=repo_id,
            repo_type="space",
        )
        print("Samples uploaded successfully.")

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nGhostScan successfully deployed to Hugging Face Spaces: {space_url}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy GhostScan to a Hugging Face Space repository."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face Space repo ID (e.g., 'your-username/ghostscan')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hugging Face Space private upon creation",
    )
    args = parser.parse_args()
    push_to_hub(repo_id=args.repo_id, token=args.token, private=args.private)


if __name__ == "__main__":
    main()
