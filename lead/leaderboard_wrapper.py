#!/usr/bin/env python3
"""
==============================================================================
CARLA Leaderboard Evaluation Wrapper
==============================================================================

1. WHY THIS EXISTS:
----------------
Primary motivation: Python-based execution makes debugging easier!
- Set breakpoints and inspect variables (vs. bash scripts with only print statements)
- Step through evaluation pipeline with IDE debugger
- Proper debugging workflow instead of log file archaeology

Secondary benefits:
- Unified interface for multiple leaderboard variants (Standard/Bench2Drive/Autopilot)
- Automatic environment setup and path management
- Consistent CLI across all modes

2. WHAT IT DOES:
-------------
Provides a unified interface for running CARLA autonomous driving evaluations:
1. Detects workspace structure and sets up paths
2. Configures environment for the selected leaderboard variant
3. Executes the appropriate evaluator as a subprocess
4. Handles checkpoints, logging, and output organization

3. EVALUATION MODES:
-----------------
A "mode" = combination of agent, track type, and leaderboard variant.

Current modes:
• EXPERT MODE: Expert agent with privileged info (MAP track + AUTOPILOT leaderboard)
  - Use case: Generate training data, debug with perfect perception
  - CLI: --expert

• MODEL MODE: Learned driving policy with realistic sensors (SENSORS track)
  - Use case: Evaluate trained models on benchmarks
  - CLI: --checkpoint <model_dir>
  - Variants: --bench2drive flag switches to extended benchmark

Each mode bundles the right agent script, track type, and leaderboard together.

4. EXTENDING THIS WRAPPER:
-----------------------
To add a new evaluation mode (e.g., a different agent type):

1. Add constants to ModeConfig class (around line 95):
   - NEW_MODE_AGENT = "path/to/your/agent.py"
   - NEW_MODE_TRACK = "SENSORS" or "MAP"
   - NEW_MODE_LEADERBOARD = LeaderboardType.STANDARD (or create new type)

2. Update get_mode_config() method in ModeConfig:
   - Add new parameter for mode detection (e.g., is_new_mode)
   - Add conditional logic to return your mode's configuration

3. Update main() function:
   - Add CLI argument (e.g., --new-mode flag)
   - Pass the flag to get_mode_config()

4. Done! The rest of the pipeline handles the new mode automatically.

5. USAGE:
------
Model evaluation:
  python leaderboard_wrapper.py --checkpoint <model_dir> --routes <route.xml>

Expert evaluation:
  python leaderboard_wrapper.py --expert --routes <route.xml>

Bench2Drive:
  python leaderboard_wrapper.py --checkpoint <model_dir> --routes <route.xml> --bench2drive

==============================================================================
"""

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path

from lead.common.logging_config import setup_logging

setup_logging()
LOG = logging.getLogger(__name__)


class LeaderboardType(Enum):
    """Type of leaderboard to use."""

    STANDARD = "standard"
    BENCH2DRIVE = "bench2drive"
    AUTOPILOT = "autopilot"


# Mode-specific constants
class ModeConfig:
    """Configuration constants for different evaluation modes.

    All mode-specific settings are centralized here for easy maintenance.
    See top docstring for extension guide.
    """

    # Expert mode (privileged information for data generation)
    EXPERT_AGENT = "lead/expert/expert.py"
    EXPERT_TRACK = "MAP"
    EXPERT_LEADERBOARD = LeaderboardType.AUTOPILOT

    # Model mode (sensor-based evaluation)
    MODEL_AGENT = "lead/inference/sensor_agent.py"
    MODEL_TRACK = "SENSORS"

    # Default settings
    DEFAULT_PORT = 2000
    DEFAULT_TM_PORT = 8000
    DEFAULT_TM_SEED = 0
    DEFAULT_TIMEOUT = 600.0
    DEFAULT_REPETITIONS = 1
    DEFAULT_PLANNER = "only_traj"

    @staticmethod
    def get_mode_config(
        is_expert: bool, is_bench2drive: bool, checkpoint: str | None, routes: str
    ) -> tuple[LeaderboardType, str, str, str | None, str]:
        """Get mode configuration based on CLI arguments.

        Args:
            is_expert: Whether expert mode is selected
            is_bench2drive: Whether bench2drive variant is selected
            checkpoint: Model checkpoint path (None for expert)
            routes: Routes file path

        Returns:
            Tuple of (leaderboard_type, agent, agent_config, checkpoint_dir, track)
        """
        if is_expert:
            return (
                ModeConfig.EXPERT_LEADERBOARD,
                ModeConfig.EXPERT_AGENT,
                routes,
                None,
                ModeConfig.EXPERT_TRACK,
            )

        leaderboard = (
            LeaderboardType.BENCH2DRIVE if is_bench2drive else LeaderboardType.STANDARD
        )
        return (
            leaderboard,
            ModeConfig.MODEL_AGENT,
            checkpoint,
            checkpoint,
            ModeConfig.MODEL_TRACK,
        )


class LeaderboardWrapper:
    """Wrapper for running CARLA leaderboard evaluations."""

    def __init__(
        self, routes: str, leaderboard_type: LeaderboardType = LeaderboardType.STANDARD
    ):
        """
        Initialize the leaderboard wrapper.

        Args:
            routes: Path to routes XML file
            leaderboard_type: Type of leaderboard to use
        """
        self.routes = Path(routes)
        self.leaderboard_type = leaderboard_type

        # Resolve workspace root from environment variable
        self.workspace_root = Path(os.environ["LEAD_PROJECT_ROOT"]).resolve()

        # Auto-detect scenario type and route ID
        self.scenario_type = self.routes.parent.name
        self.route_id = self.routes.stem.split("_")[0]

    def get_leaderboard_evaluator_paths(self) -> dict:
        """Get paths to leaderboard evaluator components for subprocess execution. [Subprocess setup]

        Returns paths needed to locate and run the leaderboard evaluator:
        - Where to find the evaluator script
        - Where to find scenario runner dependencies
        - Where to find CARLA Python API (for AUTOPILOT mode)

        These paths are used to build PYTHONPATH and execute the evaluator subprocess.

        Returns:
            Dictionary containing paths:
            - leaderboard_root: Root directory of leaderboard code
            - scenario_runner_root: Root directory of scenario runner
            - evaluator_script: Path to main evaluator script
            - evaluator_module: Python module path (kept for compatibility)
            - carla_path: (AUTOPILOT only) Path to CARLA Python API
        """
        if self.leaderboard_type == LeaderboardType.BENCH2DRIVE:
            return {
                "leaderboard_root": self.workspace_root
                / "3rd_party/Bench2Drive/leaderboard",
                "scenario_runner_root": self.workspace_root
                / "3rd_party/Bench2Drive/scenario_runner",
                "evaluator_script": self.workspace_root
                / "3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py",
                "evaluator_module": "leaderboard.leaderboard_evaluator",
            }
        elif self.leaderboard_type == LeaderboardType.AUTOPILOT:
            return {
                "leaderboard_root": self.workspace_root
                / "3rd_party/leaderboard_autopilot",
                "scenario_runner_root": self.workspace_root
                / "3rd_party/scenario_runner_autopilot",
                "evaluator_script": self.workspace_root
                / "3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py",
                "evaluator_module": "leaderboard.leaderboard_evaluator_local",
                "carla_path": self.workspace_root
                / "3rd_party/CARLA_0915/PythonAPI/carla",
            }
        else:  # STANDARD
            return {
                "leaderboard_root": self.workspace_root / "3rd_party/leaderboard",
                "scenario_runner_root": self.workspace_root
                / "3rd_party/scenario_runner",
                "evaluator_script": self.workspace_root
                / "3rd_party/leaderboard/leaderboard/leaderboard_evaluator.py",
                "evaluator_module": "leaderboard.leaderboard_evaluator",
            }

    def _build_pythonpath(self, paths: dict) -> str:
        """Build PYTHONPATH string from leaderboard paths. [Subprocess environment]

        Constructs PYTHONPATH for leaderboard and scenario runner paths.
        LEAD package is assumed to be installed in the Python environment.

        Args:
            paths: Dictionary of leaderboard paths from get_leaderboard_evaluator_paths()

        Returns:
            Colon-separated PYTHONPATH string ready for environment variable
        """
        pythonpath_parts = [
            str(paths["leaderboard_root"]),
            str(paths["scenario_runner_root"]),
        ]
        if "carla_path" in paths:
            pythonpath_parts.insert(0, str(paths["carla_path"]))

        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)

        return ":".join(pythonpath_parts)

    def _determine_evaluation_output_dir(
        self, output_dir: Path | None, checkpoint_dir: str | None
    ) -> Path:
        """Determine where to save evaluation results. [Main process logic]

        Output directory logic:
        - If explicitly provided: use provided directory
        - Model evaluation mode: outputs/local_evaluation/{scenario}/{route_id}
        - Expert mode: data/expert_debug

        Args:
            output_dir: Explicitly provided output directory (takes precedence)
            checkpoint_dir: Model checkpoint directory (None for expert mode)

        Returns:
            Resolved output directory path
        """
        if output_dir is not None:
            return output_dir

        if checkpoint_dir:
            # Model evaluation: organize by scenario and route
            return (
                self.workspace_root
                / f"outputs/local_evaluation/{self.scenario_type}/{self.route_id}"
            )
        else:
            # Expert evaluation: debug directory
            return self.workspace_root / "data/expert_debug"

    def _get_common_leaderboard_env_vars(self, paths: dict) -> dict:
        """Get environment variables common to all leaderboard evaluations. [Subprocess environment]

        Sets up core variables needed by all leaderboard variants:
        - Python path configuration (PYTHONPATH)
        - Leaderboard/scenario runner paths
        - Route and scenario information
        - Leaderboard type flags (IS_BENCH2DRIVE)

        Args:
            paths: Dictionary of leaderboard paths from get_leaderboard_evaluator_paths()

        Returns:
            Common environment variables dictionary
        """
        return {
            "PYTHONPATH": self._build_pythonpath(paths),
            "SCENARIO_RUNNER_ROOT": str(paths["scenario_runner_root"]),
            "LEADERBOARD_ROOT": str(paths["leaderboard_root"]),
            "ROUTES": str(self.routes.absolute()),
            "SCENARIO_TYPE": self.scenario_type,
            "BENCHMARK_ROUTE_ID": self.route_id,
            "ROUTE_NUMBER": self.route_id,
            "PYTHONUNBUFFERED": "1",
            "IS_BENCH2DRIVE": "1"
            if self.leaderboard_type == LeaderboardType.BENCH2DRIVE
            else "0",
        }

    def _get_agent_mode_env_vars(
        self, output_dir: Path, checkpoint_dir: str | None
    ) -> dict:
        """Get environment variables specific to agent mode (expert vs model). [Subprocess environment]

        Model mode sets:
        - CHECKPOINT_DIR: Path to model checkpoint
        - SAVE_PATH: Output directory for evaluation results

        Expert mode sets:
        - SAVE_PATH: Data collection directory
        - DATAGEN: Flag to enable data generation
        - DEBUG_CHALLENGE: Debug mode flag
        - TEAM_CONFIG: Routes configuration

        Args:
            output_dir: Output directory path
            checkpoint_dir: Model checkpoint directory (None for expert mode)

        Returns:
            Agent mode-specific environment variables
        """
        env_vars = {
            "OUTPUT_DIR": str(output_dir),
            "EVALUATION_OUTPUT_DIR": str(output_dir),
        }

        if checkpoint_dir:
            # Model evaluation mode
            env_vars["CHECKPOINT_DIR"] = checkpoint_dir
            env_vars["SAVE_PATH"] = str(output_dir)
        else:
            # Expert mode
            env_vars["SAVE_PATH"] = str(output_dir / "data" / self.scenario_type)
            env_vars["DATAGEN"] = "1"
            env_vars["DEBUG_CHALLENGE"] = "0"
            env_vars["TEAM_CONFIG"] = str(self.routes.absolute())

        return env_vars

    def setup_leaderboard_environment(
        self,
        output_dir: Path | None = None,
        checkpoint_dir: str | None = None,
        extra_env: dict | None = None,
    ) -> dict:
        """Setup environment variables for leaderboard evaluator subprocess. [Subprocess environment]

        Configures ~15 environment variables needed by the leaderboard evaluator:
        PYTHONPATH, output directories, checkpoint paths, scenario info, etc.
        These are required for the evaluator subprocess to run correctly.

        Args:
            output_dir: Output directory (auto-generated if None)
            checkpoint_dir: Model checkpoint directory (None for expert mode)
            extra_env: Additional environment variables to merge

        Returns:
            Complete environment variables dictionary
        """
        paths = self.get_leaderboard_evaluator_paths()
        output_dir = self._determine_evaluation_output_dir(output_dir, checkpoint_dir)

        # Build environment variables
        env_vars = self._get_common_leaderboard_env_vars(paths)
        env_vars.update(self._get_agent_mode_env_vars(output_dir, checkpoint_dir))

        # Add extra environment variables
        if extra_env:
            env_vars.update(extra_env)

        # Apply to os.environ
        for key, value in env_vars.items():
            os.environ[key] = value

        return env_vars

    def clean_output_dir(self, output_dir: Path) -> None:
        """Clean and recreate output directory for fresh evaluation. [Main process]

        Removes existing output directory and recreates it with necessary
        subdirectories (logs). This ensures clean state for new evaluations.

        Args:
            output_dir: Output directory path to clean and recreate

        Note:
            Use with caution - this permanently deletes existing evaluation results.
        """
        if output_dir.exists():
            LOG.info(f"Cleaning previous output directory: {output_dir}")
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create log directory
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        LOG.info(f"Created output directory: {output_dir}")

    def _prepare_checkpoint_paths(self, output_path: Path) -> tuple[Path, Path]:
        """Create checkpoint directories and return checkpoint file paths. [Main process]

        Creates two types of checkpoint files:
        1. checkpoint_endpoint.json: Main evaluation checkpoint for resume
        2. debug_checkpoint_endpoint.txt: Debug checkpoint for detailed tracking

        Args:
            output_path: Base output directory for evaluation

        Returns:
            Tuple of (checkpoint_path, debug_checkpoint_path)
        """
        checkpoint_path = output_path / "checkpoint_endpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        debug_checkpoint_path = (
            output_path / "debug_checkpoint/debug_checkpoint_endpoint.txt"
        )
        debug_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        return checkpoint_path, debug_checkpoint_path

    def _run_evaluator(
        self, paths: dict, env_vars: dict, output_path: Path, **run_params
    ):
        """Run evaluator as subprocess. [Subprocess execution]

        Executes evaluator script as a subprocess with properly configured
        environment variables. Handles SIGINT (CTRL+C) gracefully by forwarding
        the signal to the subprocess and allowing time for cleanup.

        Args:
            paths: Dictionary of leaderboard paths from get_leaderboard_evaluator_paths()
            env_vars: Environment variables to set for subprocess
            output_path: Output directory path for logging
            **run_params: Evaluation parameters (agent, ports, timeouts, etc.)

        Returns:
            Subprocess result object
        """
        env = os.environ.copy()
        env.update(env_vars)

        self._print_config(env_vars, output_path)

        cmd = self._build_command(paths["evaluator_script"], **run_params)

        # Use Popen for better process control
        process = None
        try:
            process = subprocess.Popen(cmd, cwd=self.workspace_root, env=env)
            returncode = process.wait()
            return subprocess.CompletedProcess(cmd, returncode)

        except KeyboardInterrupt:
            LOG.info("\n" + "=" * 80)
            LOG.info("Received CTRL+C - initiating graceful shutdown...")
            LOG.info("=" * 80)

            if process:
                # Send SIGINT to subprocess to allow graceful cleanup
                try:
                    LOG.info("Sending interrupt signal to subprocess...")
                    process.send_signal(signal.SIGINT)

                    # Wait up to 30 seconds for graceful shutdown
                    LOG.info("Waiting for subprocess to clean up (max 30s)...")
                    for i in range(30):
                        if process.poll() is not None:
                            LOG.info(f"Subprocess exited cleanly after {i + 1} seconds")
                            break
                        time.sleep(1)
                    else:
                        # If still running after timeout, send SIGTERM
                        LOG.warning(
                            "Subprocess did not exit after 30s, sending SIGTERM..."
                        )
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                            LOG.info("Subprocess terminated successfully")
                        except subprocess.TimeoutExpired:
                            # Last resort: force kill
                            LOG.error("Subprocess did not terminate, forcing kill...")
                            process.kill()
                            process.wait()

                except Exception as e:
                    LOG.error(f"Error during cleanup: {e}")
                    if process and process.poll() is None:
                        process.kill()

            LOG.info("=" * 80)
            LOG.info("Shutdown complete")
            LOG.info("=" * 80)
            sys.exit(130)  # Standard exit code for SIGINT

        finally:
            # Ensure process is cleaned up
            if process and process.poll() is None:
                try:
                    process.kill()
                    process.wait()
                except:
                    pass

    def run(
        self,
        agent: str,
        agent_config: str | None = None,
        output_dir: Path | None = None,
        checkpoint_dir: str | None = None,
        port: int = ModeConfig.DEFAULT_PORT,
        traffic_manager_port: int = ModeConfig.DEFAULT_TM_PORT,
        traffic_manager_seed: int = ModeConfig.DEFAULT_TM_SEED,
        repetitions: int = ModeConfig.DEFAULT_REPETITIONS,
        timeout: float = ModeConfig.DEFAULT_TIMEOUT,
        resume: bool = True,
        debug: int = 0,
        track: str = ModeConfig.MODEL_TRACK,
        planner_type: str = ModeConfig.DEFAULT_PLANNER,
        clean: bool = True,
        extra_env: dict | None = None,
        extra_args: list | None = None,
    ):
        """Run the complete leaderboard evaluation pipeline. [Main process orchestration]

        Main entry point for running evaluations. Orchestrates:
        1. Environment setup
        2. Output directory preparation
        3. Checkpoint configuration
        4. Evaluator subprocess execution

        Args:
            agent: Relative path to agent script (e.g., 'lead/expert/expert.py')
            agent_config: Agent configuration (checkpoint dir or routes file)
            output_dir: Output directory (auto-generated if None)
            checkpoint_dir: Model checkpoint directory (None for expert mode)
            port: CARLA server port
            traffic_manager_port: Traffic manager port
            traffic_manager_seed: Seed for traffic manager (default: 0)
            repetitions: Number of route repetitions (default: 1)
            timeout: Timeout per route in seconds (default: 600.0)
            resume: Resume from checkpoint if exists (default: True)
            debug: Debug level 0-2 (default: 0)
            track: Track type - 'SENSORS' or 'MAP' (default: 'SENSORS')
            planner_type: Planner type for model evaluation (default: 'only_traj')
            clean: Clean output directory before running (default: True)
            extra_env: Additional environment variables
            extra_args: Additional CLI arguments for evaluator

        Returns:
            Subprocess result object

        Example:
            >>> wrapper = LeaderboardWrapper('data/routes/town01.xml')
            >>> wrapper.run(
            ...     agent='lead/expert/expert.py',
            ...     track='MAP',
            ...     debug=1
            ... )
        """
        # Setup environment for leaderboard subprocess
        env_vars = self.setup_leaderboard_environment(
            output_dir, checkpoint_dir, extra_env
        )
        output_path = Path(env_vars["OUTPUT_DIR"])

        if clean:
            self.clean_output_dir(output_path)

        # Prepare checkpoint paths
        checkpoint_path, debug_checkpoint_path = self._prepare_checkpoint_paths(
            output_path
        )

        # Get leaderboard paths
        paths = self.get_leaderboard_evaluator_paths()

        # Common run parameters
        run_params = {
            "agent": agent,
            "agent_config": agent_config,
            "checkpoint_path": checkpoint_path,
            "debug_checkpoint_path": debug_checkpoint_path,
            "port": port,
            "traffic_manager_port": traffic_manager_port,
            "traffic_manager_seed": traffic_manager_seed,
            "repetitions": repetitions,
            "timeout": timeout,
            "resume": resume,
            "debug": debug,
            "track": track,
            "extra_args": extra_args,
        }

        # Run evaluator as subprocess
        return self._run_evaluator(paths, env_vars, output_path, **run_params)

    def _get_agent_paths(
        self, agent: str, agent_config: str | None
    ) -> tuple[Path, str]:
        """Resolve agent script and configuration paths. [Main process logic]

        Converts relative agent path to absolute and resolves config path,
        using routes file as default if no config provided.

        Args:
            agent: Relative path to agent script from workspace root
            agent_config: Agent config path (checkpoint or routes), None for default

        Returns:
            Tuple of (absolute_agent_path, config_path)
        """
        agent_path = self.workspace_root / agent
        config = agent_config if agent_config else str(self.routes.absolute())
        return agent_path, config

    def _build_base_args(
        self,
        agent_path: Path,
        agent_config: str,
        checkpoint_path: Path,
        port: int,
        traffic_manager_port: int,
        traffic_manager_seed: int,
        repetitions: int,
        timeout: float,
        resume: bool,
        debug: int,
        track: str,
    ) -> list:
        """Build base argument list common to all evaluator variants. [Subprocess arguments]

        Constructs CLI arguments accepted by all leaderboard evaluators:
        routes, track, checkpoint, agent, ports, timing, etc.

        Args:
            agent_path: Absolute path to agent script
            agent_config: Agent configuration path
            checkpoint_path: Path to checkpoint file for resume
            port: CARLA server port
            traffic_manager_port: Traffic manager port
            traffic_manager_seed: Traffic manager random seed
            repetitions: Number of route repetitions
            timeout: Timeout per route in seconds
            resume: Whether to resume from checkpoint
            debug: Debug level (0-2)
            track: Track type ('SENSORS' or 'MAP')

        Returns:
            List of command-line arguments as strings
        """
        return [
            "--routes",
            str(self.routes.absolute()),
            "--track",
            track,
            "--checkpoint",
            str(checkpoint_path),
            "--agent",
            str(agent_path),
            "--agent-config",
            agent_config,
            "--debug",
            str(debug),
            "--resume",
            str(int(bool(resume))),
            "--port",
            str(port),
            "--traffic-manager-port",
            str(traffic_manager_port),
            "--traffic-manager-seed",
            str(traffic_manager_seed),
            "--repetitions",
            str(repetitions),
            "--timeout",
            str(timeout),
        ]

    def _build_args(
        self,
        agent: str,
        agent_config: str | None,
        checkpoint_path: Path,
        debug_checkpoint_path: Path,
        port: int,
        traffic_manager_port: int,
        traffic_manager_seed: int,
        repetitions: int,
        timeout: float,
        resume: bool,
        debug: int,
        track: str,
        extra_args: list | None,
    ) -> list:
        """Build complete argument list for leaderboard evaluator. [Subprocess arguments]

        Combines base arguments with leaderboard-specific arguments
        (like debug checkpoint) and custom extra arguments.

        Args:
            agent: Relative path to agent script
            agent_config: Agent config path or None for default
            checkpoint_path: Main checkpoint file path
            debug_checkpoint_path: Debug checkpoint file path
            port: CARLA server port
            traffic_manager_port: Traffic manager port
            traffic_manager_seed: Traffic manager seed
            repetitions: Number of route repetitions
            timeout: Timeout per route in seconds
            resume: Whether to resume from checkpoint
            debug: Debug level
            track: Track type
            extra_args: Additional custom arguments

        Returns:
            Complete list of command-line arguments
        """
        agent_path, config = self._get_agent_paths(agent, agent_config)

        args = self._build_base_args(
            agent_path,
            config,
            checkpoint_path,
            port,
            traffic_manager_port,
            traffic_manager_seed,
            repetitions,
            timeout,
            resume,
            debug,
            track,
        )

        # Add debug checkpoint if not autopilot
        if self.leaderboard_type != LeaderboardType.AUTOPILOT:
            args.extend(
                ["--debug-checkpoint", str(debug_checkpoint_path), "--record", "None"]
            )

        if extra_args:
            args.extend(extra_args)

        return args

    def _build_command(self, script_path: Path, **kwargs) -> list:
        """Build complete command list for subprocess execution. [Subprocess command]

        Constructs subprocess command by prepending Python executable
        and script path to the standard argument list.

        Args:
            script_path: Path to evaluator script to execute
            **kwargs: All arguments passed through to _build_args()

        Returns:
            Complete command list for subprocess.run()
            Format: [python_executable, script_path, arg1, arg2, ...]
        """
        # Reuse _build_args logic but prepend python executable and script
        args = self._build_args(**kwargs)
        return [sys.executable, str(script_path)] + args

    def _print_config(self, env_vars: dict, output_path: Path) -> None:
        """Print formatted evaluation configuration to log. [Main process logging]

        Displays key evaluation parameters in a formatted banner:
        - Leaderboard type
        - Routes file
        - Scenario type and route ID
        - Output directory
        - Checkpoint directory (if applicable)

        Args:
            env_vars: Environment variables dictionary
            output_path: Output directory path
        """
        LOG.info("\n" + "=" * 80)
        LOG.info(
            f"Starting CARLA Leaderboard Evaluation ({self.leaderboard_type.value})"
        )
        LOG.info("=" * 80)
        LOG.info(f"Routes: {self.routes}")
        LOG.info(f"Scenario Type: {self.scenario_type}")
        LOG.info(f"Route ID: {self.route_id}")
        LOG.info(f"Output Dir: {output_path}")
        if "CHECKPOINT_DIR" in env_vars:
            LOG.info(f"Checkpoint Dir: {env_vars['CHECKPOINT_DIR']}")
        LOG.info("=" * 80 + "\n")


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure CLI argument parser with all options.

    Sets up argument parser with:
    - Mode selection (--checkpoint vs --expert)
    - Required arguments (--routes)
    - Leaderboard type (--bench2drive)
    - CARLA connection settings (ports)
    - Evaluation settings (repetitions, timeout, resume, debug)
    - Model-specific settings (planner-type, gpu)
    - Output control (no-clean, output-dir)

    Returns:
        Configured argument parser with usage examples
    """
    parser = argparse.ArgumentParser(
        description="Run CARLA Leaderboard Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Evaluate model on Town13
  python $LEAD_PROJECT_ROOT/lead/leaderboard_wrapper.py --checkpoint outputs/checkpoints/tfv6_resnet34 --routes data/benchmark_routes/Town13/0.xml

  # Evaluate model on Bench2Drive
  python $LEAD_PROJECT_ROOT/lead/leaderboard_wrapper.py --checkpoint outputs/checkpoints/tfv6_resnet34 --routes data/benchmark_routes/bench2drive220routes/23687.xml --bench2drive

  # Evaluate expert agent
  python $LEAD_PROJECT_ROOT/lead/leaderboard_wrapper.py --expert --routes data/benchmark_routes/Town13/1.xml

  # Evaluate expert for data generation
  python $LEAD_PROJECT_ROOT/lead/leaderboard_wrapper.py --expert --routes data/data_routes/lead/noScenarios/short_route.xml
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint directory (for model evaluation)",
    )
    mode_group.add_argument(
        "--expert", action="store_true", help="Run expert agent (for expert evaluation)"
    )

    # Required arguments
    parser.add_argument(
        "--routes", type=str, required=True, help="Path to the routes XML file"
    )

    # Leaderboard type
    parser.add_argument(
        "--bench2drive", action="store_true", help="Use Bench2Drive leaderboard"
    )

    # CARLA settings
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--traffic-manager-port", type=int, default=8000, help="Traffic manager port"
    )
    parser.add_argument(
        "--traffic-manager-seed", type=int, default=0, help="Traffic manager seed"
    )

    # Evaluation settings
    parser.add_argument(
        "--repetitions", type=int, default=1, help="Number of repetitions per route"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Timeout in seconds"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="Resume from checkpoint"
    )
    parser.add_argument("--debug", type=int, default=0, help="Debug mode")
    parser.add_argument(
        "--planner-type",
        type=str,
        default="only_traj",
        help="Planner type (for model evaluation)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID (for model evaluation)"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean output directory before running",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (auto-generated if not specified)",
    )

    return parser


def main():
    """CLI interface for running CARLA leaderboard evaluations.

    Command-line entry point that:
    1. Parses CLI arguments
    2. Sets GPU device for model evaluation
    3. Configures evaluation mode (expert vs model)
    4. Creates LeaderboardWrapper instance
    5. Runs evaluation with specified parameters

    Exit codes:
        0: Success
        1: Error during evaluation
        130: Keyboard interrupt (Ctrl+C)

    Examples:
        Model evaluation:
            $ python leaderboard_wrapper.py \\
                --checkpoint outputs/checkpoints/model \\
                --routes data/routes/town01.xml

        Expert data generation:
            $ python leaderboard_wrapper.py \\
                --expert \\
                --routes data/routes/short_route.xml

        Bench2Drive evaluation:
            $ python leaderboard_wrapper.py \\
                --checkpoint outputs/checkpoints/model \\
                --routes data/routes/bench2drive.xml \\
                --bench2drive
    """

    parser = _create_argument_parser()
    args = parser.parse_args()

    # Set GPU for model evaluation
    if args.checkpoint:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Get mode configuration
    leaderboard_type, agent, agent_config, checkpoint_dir, track = (
        ModeConfig.get_mode_config(
            is_expert=args.expert,
            is_bench2drive=args.bench2drive,
            checkpoint=args.checkpoint,
            routes=args.routes,
        )
    )

    # Setup extra environment variables
    extra_env = {}
    if args.checkpoint:
        extra_env["PLANNER_TYPE"] = args.planner_type

    # Create wrapper and run
    wrapper = LeaderboardWrapper(routes=args.routes, leaderboard_type=leaderboard_type)
    output_dir = Path(args.output_dir) if args.output_dir else None

    wrapper.run(
        agent=agent,
        agent_config=agent_config,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        port=args.port,
        traffic_manager_port=args.traffic_manager_port,
        traffic_manager_seed=args.traffic_manager_seed,
        repetitions=args.repetitions,
        timeout=args.timeout,
        resume=args.resume,
        debug=args.debug,
        track=track,
        planner_type=args.planner_type if args.checkpoint else "only_traj",
        clean=not args.no_clean,
        extra_env=extra_env,
    )


if __name__ == "__main__":
    main()
