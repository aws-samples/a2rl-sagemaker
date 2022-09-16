"""Submit a job to train A2RL simulators on SageMaker training containers.

.. code-block:: bash

    # Train with default settings
    python3 submit_train.py

    # When submitting from outside a SageMaker notebook instance (or Studio notebook), must
    # specify the execution role.
    python3 submit_train.py --role arn:aws:iam::111122223333:role/my-sagemaker-execution-role

    # or, just set the execution role in environment variable SAGEMAKER_EXECUTION_ROLE
    export SAGEMAKER_EXECUTION_ROLE= arn:aws:iam::111122223333:role/my-sagemaker-execution-role
    python3 submit_train.py

    # Handy for dev: verify what goes into the training code tarball starting a training job.
    python3 submit_train.py --generate-tarball-only

    # Submit a really short training job (~5min).
    # NOTE: the submit script will passthrough unknown CLI args as hyperparameters directly to the
    #       entrypoint script. In this example, "--reward-function revenue_0_20" is passed as-is to
    #       flight_sales/run_exp.py.
    python3 submit_train.py --reward-function revenue_0_20
"""
from __future__ import annotations

import smepu

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from loguru import logger
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session


@dataclass
class Default:
    value: Any
    metadata: None | Any = None


def get_defaults() -> argparse.Namespace:
    """Best efforts to auto-detect a bunch of default values."""
    defaults = argparse.Namespace()

    # Change me!
    defaults.sagemaker_s3_prefix = f"s3://{Session().default_bucket()}/a2rl/sagemaker-training-jobs"

    # default.role
    if "SAGEMAKER_EXECUTION_ROLE" in os.environ:
        defaults.role = Default(
            os.environ["SAGEMAKER_EXECUTION_ROLE"],
            metadata="$SAGEMAKER_EXECUTION_ROLE",
        )
    else:
        try:
            import sagemaker

            defaults.role = Default(
                sagemaker.get_execution_role(),
                "sagemaker.get_execution_role()",
            )
        except ImportError:
            defaults.role = Default("")

    # defaults.flight_sales
    try:
        import flight_sales

        defaults.flight_sales = Default(
            str(Path(flight_sales.__file__).parent),
            metadata="Python path",
        )
    except ImportError:
        defaults.flight_sales = Default(
            str(Path(__file__).parent / "flight_sales"),
            metadata="submit script's directory",
        )

    # defaults.requirements_txt
    defaults.requirements_txt = Default(
        Path(__file__).parent / "requirements-pytorch-1.12-dlc-with-internet.txt",
        metadata="submit script's directory",
    )

    return defaults


def get_parser(defaults) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser("sagemaker-train-submitter")
    parser.add_argument(
        "--instance-type",
        default="ml.c5.4xlarge",
        help="Instance type (default: ml.c5.4xlarge)",
    )
    parser.add_argument(
        "--role",
        default=defaults.role.value,
        help=(
            f"SageMaker execution role (default: {defaults.role.value} from "
            f"{defaults.role.metadata})"
        ),
    )
    parser.add_argument(
        "--flight-sales",
        default=defaults.flight_sales.value,
        help=(
            f"Path to the flight_sales module (default: {defaults.flight_sales.value} in "
            f"{defaults.flight_sales.metadata})"
        ),
    )
    parser.add_argument(
        "--requirements-txt",
        type=str,
        default=defaults.requirements_txt.value,
        help=(
            "Path to requirements file, where 'None' means file not needed (default: "
            f"{defaults.requirements_txt.value} in {defaults.requirements_txt.metadata})"
        ),
    )
    parser.add_argument(
        "--generate-tarball-only",
        action="store_true",
        help="Generate sourcedir.tar.gz on Amazon S3, then exit.",
    )

    args, script_args = parser.parse_known_args()
    if not args.role:
        raise ValueError("Failed to autodetect SageMaker execution role. Please specify one.")

    if args.requirements_txt == "None":
        args.requirements_txt = None

    return args, script_args


def generate_entrypoint(
    entrypoint_body: str,
    requirements_txt: None | str | Path = None,
) -> tuple[str, str, TemporaryDirectory]:
    """Generate ``source_dir/*`` on-the-fly."""
    tmp = TemporaryDirectory(ignore_cleanup_errors=True)
    source_dir = Path(tmp.name)

    # Generate SageMaker training entrypoint
    with (source_dir / "entrypoint.py").open("w") as f:
        f.write(entrypoint_body)

    # Requirements.txt
    if requirements_txt:
        shutil.copy(requirements_txt, source_dir / "requirements.txt")

        # Add some metadata
        abs_requirements_txt = Path(requirements_txt).absolute()
        with (source_dir / "requirements.yaml").open("w") as f:
            f.write(f"src: {str(abs_requirements_txt)}\n")

    return (
        "entrypoint.py",
        str(source_dir),  # SageMaker estimator only accepts strings, not Path objects :(
        tmp,  # Keep this ref around to prevent gc before estimator has a chance.
    )


def epilogue(estimator: PyTorch, defaults: argparse.Namespace, has_train_job: bool) -> None:
    """Show (hopefully) hopeful messages and hits to users after uploading code tarball or
    submitting a training job."""
    code_dir = estimator.uploaded_code.s3_prefix
    script = estimator.uploaded_code.script_name
    logger.success("(code_dir, script) = ({}, {})", code_dir, script)
    logger.info(
        "CLI to preview content: "
        "aws s3 cp {} - | tar --to-stdout -xzf - entrypoint.py"
        f"{' requirements.txt requirements.yaml' if args.requirements_txt else ''}",
        code_dir,
    )
    logger.info("CLI to list content: aws s3 cp {} - | tar -tzf -", code_dir)

    if not has_train_job:
        sys.exit(1)

    job_name = estimator._current_job_name
    logger.success("Training job name: {}", job_name)
    logger.info(
        (
            "CLI to describe training job: "
            "aws sagemaker describe-training-job --training-job-name {} | jq"
        ),
        job_name,
    )
    logger.info(
        "Model will be saved at: {}",
        f"{defaults.sagemaker_s3_prefix}/output/model.tar.gz",
    )
    logger.info(
        "Training output (incl. backtesting results) will be saved at: {}",
        f"{defaults.sagemaker_s3_prefix}/{job_name}/output/output.tar.gz",
    )
    logger.info(
        "URL: https://{}.console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}",
        estimator.sagemaker_session.boto_region_name,
        estimator.sagemaker_session.boto_region_name,
        job_name,
    )


if __name__ == "__main__":
    defaults = get_defaults()
    args, script_args = get_parser(defaults)
    logger.info("Default assumed: {}", {k: v for k, v in vars(defaults).items()})
    logger.info("CLI args: {}", vars(args))
    logger.info("Entrypoint args: {}", smepu.argparse.to_kwargs(script_args))

    # Meta-programming fun :) -- send remaining CLI args to entrypoint script (via hyperparameters).
    hyperparameters = smepu.argparse.to_kwargs(script_args)

    entry_point, source_dir, _ = generate_entrypoint(
        """from flight_sales.run_exp import main

main()
""",
        requirements_txt=args.requirements_txt,
    )
    logger.info("source_dir: {}", source_dir)

    estimator = PyTorch(
        base_job_name="a2rl-dyn-pricing",
        entry_point=entry_point,  # "entrypoint.py",
        source_dir=source_dir,  # str(args.source_dir),
        dependencies=[args.flight_sales],
        framework_version="1.12",
        py_version="py38",
        code_location=defaults.sagemaker_s3_prefix,
        output_path=defaults.sagemaker_s3_prefix,
        max_run=24 * 60 * 60,
        instance_count=1,
        instance_type=args.instance_type,
        # instance_type="ml.m5.2xlarge",  # 31min for a single reward function
        # instance_type="ml.g4dn.xlarge",  # 24min for a single reward function, low GPU util.
        # instance_type="local_gpu",
        # instance_type="local",
        role=args.role,
        hyperparameters=hyperparameters,
    )

    if args.generate_tarball_only:
        # For debugging: generage sourcedir.tar.gz on Amazon S3, without starting a training job.
        estimator._prepare_for_training()
    else:
        estimator.fit(wait=False)

    epilogue(estimator, defaults, has_train_job=(not args.generate_tarball_only))
